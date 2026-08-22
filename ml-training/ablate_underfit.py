"""Is the head underfit, or is the dataset imbalance the binding constraint?

WHY THIS EXISTS
    The production pipeline used class-balanced re-weighting (macro-F1 0.7180),
    and a sweep found LDAM much better (0.8108), so LDAM became the default. A
    later analysis argued that both readings are misinterpretations of the same
    underlying fault: the head is UNDERFIT, and LDAM's apparent win comes from
    its s=30 logit scale acting as a learning-rate multiplier rather than from
    its margin.

    Two facts from our own artifacts are consistent with that:
      * fitted temperatures are 0.476-0.519 (mean 0.505). T < 1 means temperature
        scaling had to SHARPEN the logits ~2x, which is the signature of an
        under-confident, under-fit model. ECE 0.0401 looked healthy because
        scaling repairs confidence while leaving the argmax untouched.
      * the cached features are exactly unit-norm (std 4.5e-8) with a per-dim std
        of 0.024, so Linear(768,7) at default init emits logits around 0.03 and
        lr=1e-3 under cosine decay with early stopping never escapes the
        near-uniform-softmax regime.

    Consistent-with is not proof, so this script separates the candidate causes
    instead of arguing about them. Every arm is the same protocol on the same
    cached features, differing in one thing.

THE ARMS THAT DECIDE IT
    scale-only vs LDAM-full vs margin-only is the crux. If the margin is what
    matters, removing the scale should preserve most of the gain. If the SCALE is
    what matters, plain cross-entropy plus the scale should reproduce LDAM
    without any margin at all, and margin-without-scale should collapse.

NO-LEAKAGE NOTE
    Standardisation statistics are fitted on the inner FIT split only and applied
    to val and test. Fitting them on all of X would leak test distribution into
    training and inflate every standardised arm — the exact mistake that would
    make this whole comparison worthless.

    python ablate_underfit.py --features ../ml-service/artifacts/features_dinov2-base.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sw", HERE / "sweep_head.py")
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)

SEED = sw.SEED


class ScaledCE(nn.Module):
    """Plain cross-entropy on s * logits. No margin, no class weighting.

    This is LDAM with the margin deleted and only the scale kept. It exists to
    attribute LDAM's gain to one of its two mechanisms.
    """

    def __init__(self, s: float = 30.0, label_smoothing: float = 0.05):
        super().__init__()
        self.s, self.ls = s, label_smoothing

    def forward(self, logits, target):
        return F.cross_entropy(self.s * logits, target, label_smoothing=self.ls)


class MarginOnly(nn.Module):
    """LDAM's margin with s = 1, i.e. the margin without the gradient scale."""

    def __init__(self, counts: np.ndarray, max_m: float = 0.5):
        super().__init__()
        m = 1.0 / np.sqrt(np.sqrt(np.maximum(counts, 1.0)))
        m = m * (max_m / m.max())
        self.register_buffer("m", torch.tensor(m, dtype=torch.float32))

    def forward(self, logits, target):
        adj = logits - F.one_hot(target, logits.shape[1]).float() * self.m[target][:, None]
        return F.cross_entropy(adj, target)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single-parameter Guo et al. scaling, on held-out logits."""
    lg = torch.tensor(logits, dtype=torch.float32)
    yy = torch.tensor(y, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / logT.exp(), yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().item())


def run(name, X, y, groups, k, counts, make_loss, folds=5, standardise=False,
        lr=1e-3, hidden=0, seeds=(SEED,)):
    """One arm. Returns mean over CV seeds so single-split luck cannot decide it."""
    per_seed = []
    for sd in seeds:
        sw.set_seed(sd)
        skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=sd)
        f1s, bas, logits_all, y_all = [], [], [], []
        for tr, te in skf.split(X, y, groups=groups):
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd)
            fi, vi = next(inner.split(X[tr], y[tr], groups=np.asarray(groups)[tr]))
            fit, val = tr[fi], tr[vi]

            Xf, Xv, Xt = X[fit], X[val], X[te]
            if standardise:
                # Fitted on the FIT split only. See the no-leakage note above.
                mu = Xf.mean(0, keepdims=True)
                sg = Xf.std(0, keepdims=True) + 1e-6
                Xf, Xv, Xt = (Xf - mu) / sg, (Xv - mu) / sg, (Xt - mu) / sg

            head = sw.train_head(Xf, y[fit], Xv, y[val], k,
                                 make_loss(np.bincount(y[fit], minlength=k).astype(float)),
                                 lr=lr, hidden=hidden)
            with torch.no_grad():
                lg = head(torch.tensor(Xt, dtype=torch.float32)).numpy()
            pred = lg.argmax(1)
            f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
            bas.append(balanced_accuracy_score(y[te], pred))
            logits_all.append(lg)
            y_all.append(y[te])

        lg = np.concatenate(logits_all)
        yy = np.concatenate(y_all)
        pred = lg.argmax(1)
        per_seed.append(dict(
            macro_f1=float(np.mean(f1s)), acc=float((pred == yy).mean()),
            bal_acc=float(np.mean(bas)),
            T=fit_temperature(lg, yy),
            per_class=f1_score(yy, pred, average=None, zero_division=0,
                               labels=range(k)).tolist(),
        ))

    return dict(
        name=name,
        macro_f1=float(np.mean([r["macro_f1"] for r in per_seed])),
        macro_f1_seeds=[round(r["macro_f1"], 4) for r in per_seed],
        acc=float(np.mean([r["acc"] for r in per_seed])),
        bal_acc=float(np.mean([r["bal_acc"] for r in per_seed])),
        temperature=float(np.mean([r["T"] for r in per_seed])),
        per_class=[round(x, 3) for x in np.mean([r["per_class"] for r in per_seed], axis=0)],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="ablate_underfit.json")
    ap.add_argument("--seeds", type=int, default=1, help="CV seeds per arm (1 or 3)")
    args = ap.parse_args()

    z = np.load(args.features, allow_pickle=True)
    X = z["X_eval"].astype(np.float32)
    y = z["y"].astype(np.int64)
    names = [str(c) for c in z["classes"].tolist()]
    groups = z["groups"]
    k = int(y.max()) + 1
    counts = np.bincount(y, minlength=k).astype(float)

    seeds = (SEED,) if args.seeds == 1 else (SEED, 7, 123)
    print(f"  X {X.shape}  |  norm {np.linalg.norm(X, axis=1).mean():.6f}"
          f"  |  per-dim std {X.std(0).mean():.5f}")
    print(f"  classes {names}  counts {counts.astype(int).tolist()}")
    print(f"  CV seeds {seeds}\n")

    plain = lambda c: nn.CrossEntropyLoss(label_smoothing=0.05)
    cb999 = lambda c: nn.CrossEntropyLoss(
        weight=torch.as_tensor(sw.cb_weights(c, 0.999), dtype=torch.float32),
        label_smoothing=0.05)

    # Round 2: the fixes that survived, re-checked over 3 CV seeds, plus the
    # combinations of them. Round 1 established WHY the head was underfit; this
    # picks the configuration to ship.
    arms = [
        ("K  plain CE + stand. + MLP-1024, lr 1e-3   (round-1 winner)",
         plain, True, 1e-3, 1024),
        ("L  plain CE + stand. + MLP-1024, lr 1e-2", plain, True, 1e-2, 1024),
        ("M  plain CE + stand. + MLP-2048, lr 1e-3", plain, True, 1e-3, 2048),
        ("N  plain CE, raw, lr 0.1, MLP-1024", plain, False, 1e-1, 1024),
        ("O  plain CE + stand., linear, lr 1e-2", plain, True, 1e-2, 0),
        ("P  scale-only(30) + stand. + MLP-1024", lambda c: ScaledCE(30.0), True, 1e-3, 1024),
    ]

    results = []
    for name, mk, std, lr, hid in arms:
        r = run(name, X, y, groups, k, counts, mk, standardise=std, lr=lr,
                hidden=hid, seeds=seeds)
        results.append(r)
        print(f"  {name:52s} macroF1 {r['macro_f1']:.4f}  acc {r['acc']:.4f}"
              f"  T {r['temperature']:.3f}  worst-class {min(r['per_class']):.3f}")

    results.sort(key=lambda r: -r["macro_f1"])
    print(f"\n  ranked by macro-F1:")
    for r in results:
        print(f"    {r['macro_f1']:.4f}  acc {r['acc']:.4f}  T {r['temperature']:.3f}"
              f"  {r['name']}")
    if names:
        print(f"\n  per-class F1 of the winner ({results[0]['name'].strip()}):")
        for c, f in zip(names, results[0]["per_class"]):
            print(f"    {c:14s} {f:.3f}")

    Path(args.out).write_text(json.dumps(
        {"classes": names, "counts": counts.tolist(), "seeds": list(seeds),
         "results": results}, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
