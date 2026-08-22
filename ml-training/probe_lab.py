"""One instrument for every head experiment on cached frozen features.

WHY A SHARED HARNESS
    The accuracy work has already produced two wrong conclusions from
    inconsistent protocols: "class-balanced weighting is the fix" (it was
    compensating for an underfit head) and "LDAM's margin helps" (the margin
    hurts; its s=30 scale was a learning-rate fix). Both came from comparing runs
    that differed in more than one thing. So every experiment now goes through
    this file, and a config dict is the only thing allowed to vary.

PROTOCOL (identical for every config, never overridable)
    * outer 5-fold StratifiedGroupKFold on near-duplicate groups, so the 96
      images that share a dHash group cannot straddle train and test
    * inner StratifiedGroupKFold on the training part for early stopping, so the
      outer test fold is never used to select an epoch
    * every preprocessing statistic (standardisation mu/sigma, per-class
      centroids) is fitted on the FIT split only and applied to val and test
    * temperature is fitted on pooled out-of-fold logits, never on test-only

    The leakage rules are enforced here rather than documented, because a probe
    that leaks reports a number that cannot be shipped.

USAGE
    python probe_lab.py --features F.npz --config '{"lr":0.1,"hidden":1024}'
    python probe_lab.py --features F.npz --grid grid.json --out results.json

    A config is a flat dict; unknown keys are rejected rather than ignored, so a
    typo cannot silently evaluate the default instead of what you asked for.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

# ─────────────────────────────────────────────────────────────────────────────
# Config surface. Anything not listed is a typo, not a feature.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    # "mlp" is the SGD-trained head. The rest are convex/closed-form probes,
    # which matter here for a specific reason: the SGD head was badly underfit
    # (fitted temperature 0.505, i.e. scaling had to double the logits), and a
    # convex solver cannot have that failure mode at all. The published best on
    # this dataset is also an SVM ensemble, so an SVM on DINOv2 features is the
    # obvious comparison and had never been run.
    "model": "mlp",             # mlp | logreg | linsvm | rbfsvm | lda | centroid | knn
    "C": 1.0,                   # logreg / SVM regularisation
    "gamma": "scale",           # rbfsvm kernel width
    "n_neighbors": 15,
    # Convex probes need no early-stopping split, so by default they are fitted
    # on the whole outer training fold. That is their legitimate advantage, but it
    # is ~25% more data than the MLP path sees, so set full_train=false to compare
    # architectures on identical data.
    "full_train": True,
    "whiten": 0,                # >0: PCA-whiten to this many components
    "power_norm": 0.0,          # >0: signed power normalisation x -> sign(x)|x|^p
    "loss": "plain",            # plain | class_balanced | ldam | scaled | focal | balanced_softmax
    "lr": 1e-1,
    "wd": 1e-4,
    "hidden": 0,                # 0 = linear probe
    "dropout": 0.2,
    "epochs": 150,
    "patience": 25,
    "batch": 128,
    "label_smoothing": 0.05,
    "standardise": False,       # per-fold z-scoring of features
    "l2norm_after": False,      # re-normalise after standardising
    "loss_scale": 1.0,          # multiply logits inside the loss (LDAM's 's')
    "cb_beta": 0.999,
    "ldam_max_m": 0.5,
    "focal_gamma": 2.0,
    "post": "none",             # none | prior (logit adjustment) | tau
    "tau": 1.0,
    "seeds": [123],             # CV seeds; results are averaged over them
    "ens_seeds": 1,             # heads per fold, logits averaged (>1 = ensemble)
    "select": "macro_f1",       # early-stopping criterion
    "view": "cls",              # which cached feature view to use
    "folds": 5,
    # False = plain StratifiedKFold that IGNORES near-duplicate groups. This is
    # NOT a better protocol; it exists to quantify how much published numbers on
    # this dataset are inflated by near-duplicate leakage, since the 87.82%
    # state-of-the-art paper uses an 80:20 split with no deduplication.
    "grouped": True,
}


class Head(nn.Module):
    """Dropout -> Linear, or Dropout -> Linear -> GELU -> Dropout -> Linear."""

    def __init__(self, dim, k, hidden=0, dropout=0.2):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, k))
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, k))

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────
def cb_weights(counts, beta):
    """Cui et al. 2019 effective number of samples."""
    eff = 1.0 - np.power(beta, np.maximum(counts, 1.0))
    w = (1.0 - beta) / eff
    return torch.tensor(w / w.mean() * len(counts) / len(counts), dtype=torch.float32)


class Scaled(nn.Module):
    """Plain CE on s*logits. Isolates LDAM's scale from LDAM's margin."""

    def __init__(self, s, ls):
        super().__init__()
        self.s, self.ls = s, ls

    def forward(self, z, y):
        return F.cross_entropy(self.s * z, y, label_smoothing=self.ls)


class LDAM(nn.Module):
    def __init__(self, counts, max_m, s, ls):
        super().__init__()
        m = 1.0 / np.sqrt(np.sqrt(np.maximum(counts, 1.0)))
        m = m * (max_m / m.max())
        self.register_buffer("m", torch.tensor(m, dtype=torch.float32))
        self.s, self.ls = s, ls

    def forward(self, z, y):
        adj = z - F.one_hot(y, z.shape[1]).float() * self.m[y][:, None]
        return F.cross_entropy(self.s * adj, y, label_smoothing=self.ls)


class Focal(nn.Module):
    def __init__(self, gamma, weight, ls):
        super().__init__()
        self.gamma, self.ls = gamma, ls
        self.register_buffer("w", weight if weight is not None else torch.tensor([]))

    def forward(self, z, y):
        w = self.w if self.w.numel() else None
        ce = F.cross_entropy(z, y, weight=w, reduction="none",
                             label_smoothing=self.ls)
        pt = torch.exp(-ce).clamp(1e-8, 1.0)
        return ((1 - pt) ** self.gamma * ce).mean()


class BalancedSoftmax(nn.Module):
    """Ren et al. 2020: add log(prior) to the logits during TRAINING."""

    def __init__(self, counts, ls):
        super().__init__()
        pri = counts / counts.sum()
        self.register_buffer("logpri", torch.tensor(np.log(pri + 1e-12),
                                                    dtype=torch.float32))
        self.ls = ls

    def forward(self, z, y):
        return F.cross_entropy(z + self.logpri[None, :], y, label_smoothing=self.ls)


def make_loss(cfg, counts):
    ls = cfg["label_smoothing"]
    kind = cfg["loss"]
    if kind == "plain":
        if cfg["loss_scale"] != 1.0:
            return Scaled(cfg["loss_scale"], ls)
        return nn.CrossEntropyLoss(label_smoothing=ls)
    if kind == "scaled":
        return Scaled(cfg["loss_scale"], ls)
    if kind == "class_balanced":
        return nn.CrossEntropyLoss(weight=cb_weights(counts, cfg["cb_beta"]),
                                   label_smoothing=ls)
    if kind == "ldam":
        return LDAM(counts, cfg["ldam_max_m"], cfg["loss_scale"] or 30.0, ls)
    if kind == "focal":
        return Focal(cfg["focal_gamma"], None, ls)
    if kind == "focal_cb":
        return Focal(cfg["focal_gamma"], cb_weights(counts, cfg["cb_beta"]), ls)
    if kind == "balanced_softmax":
        return BalancedSoftmax(counts, ls)
    raise ValueError(f"unknown loss {kind!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def fit_temperature(logits, y):
    lg = torch.tensor(logits, dtype=torch.float32)
    yy = torch.tensor(y, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=80)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / logT.exp(), yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(logT.exp().item(), 0.05, 10.0))


def ece(probs, y, bins=15):
    """Expected calibration error, equal-width confidence bins."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_one(Xf, yf, Xv, yv, k, cfg, lossf, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = Head(Xf.shape[1], k, cfg["hidden"], cfg["dropout"])
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    Xt = torch.tensor(Xf, dtype=torch.float32)
    yt = torch.tensor(yf, dtype=torch.long)
    Xvv = torch.tensor(Xv, dtype=torch.float32)

    best, best_state, bad = -1.0, None, 0
    for _ in range(cfg["epochs"]):
        head.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            lossf(head(Xt[idx]), yt[idx]).backward()
            opt.step()
        sch.step()
        head.eval()
        with torch.no_grad():
            pv = head(Xvv).argmax(1).numpy()
        sc = (f1_score(yv, pv, average="macro", zero_division=0)
              if cfg["select"] == "macro_f1"
              else balanced_accuracy_score(yv, pv))
        if sc > best:
            best, bad = sc, 0
            best_state = {kk: v.clone() for kk, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head


def apply_transforms(Xf, Xv, Xt, cfg):
    """Feature transforms, every statistic fitted on the FIT split only."""
    if cfg["power_norm"] > 0:
        p = cfg["power_norm"]
        pn = lambda A: np.sign(A) * np.abs(A) ** p
        Xf, Xv, Xt = pn(Xf), pn(Xv), pn(Xt)

    if cfg["standardise"]:
        mu = Xf.mean(0, keepdims=True)
        sg = Xf.std(0, keepdims=True) + 1e-6
        Xf, Xv, Xt = (Xf - mu) / sg, (Xv - mu) / sg, (Xt - mu) / sg

    if cfg["whiten"]:
        # PCA whitening equalises the variance the features carry per direction.
        # DINOv2 embeddings are strongly anisotropic (per-dim std spans 0.001 to
        # 0.037 here), so a linear probe spends most of its capacity on a handful
        # of high-variance directions unless this is done.
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(cfg["whiten"], Xf.shape[1], len(Xf)),
                  whiten=True, random_state=0).fit(Xf)
        Xf, Xv, Xt = pca.transform(Xf), pca.transform(Xv), pca.transform(Xt)

    if cfg["l2norm_after"]:
        nz = lambda A: A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        Xf, Xv, Xt = nz(Xf), nz(Xv), nz(Xt)

    return (np.ascontiguousarray(Xf, dtype=np.float32),
            np.ascontiguousarray(Xv, dtype=np.float32),
            np.ascontiguousarray(Xt, dtype=np.float32))


def fit_classical(Xf, yf, cfg, counts):
    """Convex / closed-form probes. Returns an object with decision_function."""
    kind = cfg["model"]
    # class_weight matters far less here than it did for the SGD head, but expose
    # it so "does balancing still help once the fit is right?" stays answerable.
    cw = "balanced" if cfg["loss"] == "class_balanced" else None
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=cfg["C"], max_iter=5000, class_weight=cw,
                               multi_class="multinomial", n_jobs=1)
    elif kind == "linsvm":
        from sklearn.svm import LinearSVC
        m = LinearSVC(C=cfg["C"], class_weight=cw, max_iter=20000, dual="auto")
    elif kind == "rbfsvm":
        from sklearn.svm import SVC
        m = SVC(C=cfg["C"], gamma=cfg["gamma"], class_weight=cw,
                decision_function_shape="ovr")
    elif kind == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif kind == "centroid":
        from sklearn.neighbors import NearestCentroid
        m = NearestCentroid()
    elif kind == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        m = KNeighborsClassifier(n_neighbors=cfg["n_neighbors"], weights="distance")
    else:
        raise ValueError(f"unknown model {kind!r}")
    m.fit(Xf, yf)
    return m


def classical_scores(m, X, k):
    """Uniform score matrix, whatever the estimator exposes."""
    if hasattr(m, "predict_proba"):
        p = m.predict_proba(X)
        return np.log(np.clip(p, 1e-12, 1.0))
    if hasattr(m, "decision_function"):
        d = m.decision_function(X)
        return d if d.ndim == 2 else np.stack([-d, d], axis=1)
    # NearestCentroid has neither: build a one-hot-like score from the prediction.
    pred = m.predict(X)
    out = np.full((len(X), k), -1.0)
    out[np.arange(len(X)), pred] = 1.0
    return out


def evaluate(X, y, groups, cfg):
    """Grouped CV for one config. Returns pooled out-of-fold metrics."""
    k = int(y.max()) + 1
    counts_all = np.bincount(y, minlength=k).astype(float)
    per_seed = []

    for sd in cfg["seeds"]:
        if cfg["grouped"]:
            skf = StratifiedGroupKFold(n_splits=cfg["folds"], shuffle=True,
                                       random_state=sd)
            splits = skf.split(X, y, groups=groups)
        else:
            from sklearn.model_selection import StratifiedKFold
            splits = StratifiedKFold(n_splits=cfg["folds"], shuffle=True,
                                     random_state=sd).split(X, y)
        f1s, bas, LG, YY = [], [], [], []
        for tr, te in splits:
            if cfg["grouped"]:
                inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)
                fi, vi = next(inner.split(X[tr], y[tr], groups=np.asarray(groups)[tr]))
            else:
                from sklearn.model_selection import StratifiedKFold as _SKF
                fi, vi = next(_SKF(n_splits=5, shuffle=True,
                                   random_state=sd).split(X[tr], y[tr]))
            fit, val = tr[fi], tr[vi]

            # A convex probe needs no early-stopping split, so it may train on
            # the whole outer training fold. The transform statistics then have
            # to be fitted on that same set, or the probe would be standardised
            # by statistics from a subset of its own training data.
            classical = cfg["model"] != "mlp"
            if classical and cfg["full_train"]:
                Xa, Xb, Xt2 = apply_transforms(X[tr], X[val], X[te], cfg)
                m = fit_classical(Xa, y[tr], cfg, counts_all)
                lg = classical_scores(m, Xt2, k)
            elif classical:
                Xf, Xv, Xt = apply_transforms(X[fit], X[val], X[te], cfg)
                m = fit_classical(Xf, y[fit], cfg, counts_all)
                lg = classical_scores(m, Xt, k)
            else:
                Xf, Xv, Xt = apply_transforms(X[fit], X[val], X[te], cfg)
                cf = np.bincount(y[fit], minlength=k).astype(float)
                lossf = make_loss(cfg, cf)
                # Ensemble over head seeds; logits averaged, not votes.
                acc_lg = None
                for e in range(cfg["ens_seeds"]):
                    head = train_one(Xf, y[fit], Xv, y[val], k, cfg, lossf,
                                     sd * 1000 + e)
                    with torch.no_grad():
                        lg = head(torch.tensor(Xt, dtype=torch.float32)).numpy()
                    acc_lg = lg if acc_lg is None else acc_lg + lg
                lg = acc_lg / cfg["ens_seeds"]

            if cfg["post"] == "prior":
                lg = lg - np.log(counts_all / counts_all.sum())[None, :]
            elif cfg["post"] == "tau" and not classical:
                W = head.net[-1].weight.detach().numpy()
                lg = lg / np.maximum(np.linalg.norm(W, axis=1) ** cfg["tau"], 1e-9)[None, :]

            pred = lg.argmax(1)
            f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
            bas.append(balanced_accuracy_score(y[te], pred))
            LG.append(lg)
            YY.append(y[te])

        lg = np.concatenate(LG)
        yy = np.concatenate(YY)
        pred = lg.argmax(1)
        T = fit_temperature(lg, yy)
        probs = torch.softmax(torch.tensor(lg / T), dim=1).numpy()
        per_seed.append(dict(
            macro_f1=float(np.mean(f1s)), macro_f1_fold_std=float(np.std(f1s)),
            acc=float((pred == yy).mean()), bal_acc=float(np.mean(bas)),
            temperature=T, ece=ece(probs, yy),
            per_class=f1_score(yy, pred, average=None, zero_division=0,
                               labels=range(k)).tolist(),
        ))

    agg = lambda kk: float(np.mean([r[kk] for r in per_seed]))
    return dict(
        macro_f1=agg("macro_f1"), acc=agg("acc"), bal_acc=agg("bal_acc"),
        temperature=agg("temperature"), ece=agg("ece"),
        macro_f1_seeds=[round(r["macro_f1"], 4) for r in per_seed],
        acc_seeds=[round(r["acc"], 4) for r in per_seed],
        seed_spread=float(np.max([r["macro_f1"] for r in per_seed])
                          - np.min([r["macro_f1"] for r in per_seed])),
        per_class=[round(x, 4) for x in np.mean([r["per_class"] for r in per_seed], axis=0)],
        fold_std=agg("macro_f1_fold_std"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature loading. Supports multi-view caches and concatenation across files.
# ─────────────────────────────────────────────────────────────────────────────
def load_features(paths, view):
    """paths: one or more npz. view: cached array name, or '+'-joined for concat.

    Concatenating across files requires identical y and groups; that is asserted
    rather than assumed, because silently mismatched row order would produce a
    high score that means nothing.
    """
    # Per-file views: "path::view1+view2". Different caches name their views
    # differently (the trainer writes X_eval, extract_views writes cls/mean, the
    # tiling pass writes tile_mean_cls), so combining across them needs the view
    # to travel with the path rather than being global.
    blocks, y, groups, classes = [], None, None, None
    for spec in paths:
        if "::" in str(spec):
            p, this_view = str(spec).split("::", 1)
        else:
            p, this_view = spec, view
        z = np.load(p, allow_pickle=True)
        yy = z["y"].astype(np.int64)
        gg = z["groups"]
        if y is None:
            y, groups = yy, gg
            classes = [str(c) for c in z["classes"].tolist()]
        else:
            if not np.array_equal(y, yy) or not np.array_equal(groups, gg):
                raise SystemExit(f"{p}: y/groups differ from the first file; "
                                 "these caches are not row-aligned.")
        for v in this_view.split("+"):
            key = v.strip()
            if key not in z.files:
                # The trainer names its single cached view X_eval; extract_views.py
                # names the same thing cls. Alias them so a cache from either
                # producer can be concatenated with one from the other.
                alias = {"cls": "X_eval", "X_eval": "cls"}.get(key)
                if alias in z.files:
                    key = alias
                else:
                    raise SystemExit(f"{p}: no view {key!r}. Available: {z.files}")
            blocks.append(z[key].astype(np.float32))
    X = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
    return X, y, groups, classes


def resolve(cfg_in):
    cfg = dict(DEFAULTS)
    unknown = set(cfg_in) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}. "
                         f"Valid: {sorted(DEFAULTS)}")
    cfg.update(cfg_in)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--config", default="{}", help="JSON dict")
    ap.add_argument("--grid", help="JSON: {name: {key: [values]}} cartesian product")
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    base = json.loads(args.config)
    jobs = []
    if args.grid:
        spec = json.loads(Path(args.grid).read_text(encoding="utf-8"))
        for name, axes in spec.items():
            keys = list(axes)
            for combo in itertools.product(*[axes[k] for k in keys]):
                c = dict(base)
                c.update(dict(zip(keys, combo)))
                tag = ",".join(f"{k}={v}" for k, v in zip(keys, combo))
                jobs.append((f"{name}[{tag}]", c))
    else:
        jobs.append((args.label or "config", base))

    results = []
    for name, c in jobs:
        cfg = resolve(c)
        X, y, groups, classes = load_features(args.features, cfg["view"])
        r = evaluate(X, y, groups, cfg)
        r["name"] = name
        r["config"] = {k: v for k, v in cfg.items() if v != DEFAULTS[k]}
        r["dim"] = int(X.shape[1])
        results.append(r)
        if not args.quiet:
            print(f"  {name:58s} macroF1 {r['macro_f1']:.4f}"
                  f"  acc {r['acc']:.4f}  T {r['temperature']:.3f}"
                  f"  ECE {r['ece']:.4f}  worst {min(r['per_class']):.3f}"
                  f"  d={r['dim']}", flush=True)

    results.sort(key=lambda r: -r["acc"])
    if not args.quiet and len(results) > 1:
        print("\n  ranked by accuracy:")
        for r in results[:12]:
            print(f"    acc {r['acc']:.4f}  macroF1 {r['macro_f1']:.4f}  {r['name']}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"classes": classes, "results": results}, indent=1), encoding="utf-8")
        if not args.quiet:
            print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
