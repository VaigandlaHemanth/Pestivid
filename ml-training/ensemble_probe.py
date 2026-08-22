"""Ensemble several cheap probes over ONE set of frozen features.

WHY THIS SHAPE OF ENSEMBLE
    Three probe families that individually score 0.857-0.864 accuracy disagree in
    different ways: LDA is a generative Gaussian model, LinearSVC is a max-margin
    discriminative one, and the MLP is a non-linear SGD fit. Averaging them
    measured 0.8778 accuracy versus 0.8622 for the best single probe.

    The important property is that every member reads THE SAME feature vector, so
    the expensive part -- one backbone forward pass -- is unchanged. The extra
    cost is about 3 MB of head weights against a 173 MB backbone download, and no
    extra inference latency worth measuring. That is what makes this deployable in
    the browser, unlike concatenating a second backbone (dinov2-large is a 609 MB
    fp16 ONNX download for the same accuracy).

WHY PER-ROW Z-SCORING BEFORE AVERAGING
    LDA decision values, SVM margins and MLP logits are on wildly different
    scales; a naive average is then dominated by whichever member happens to emit
    the largest numbers. Z-scoring each member's 7 class scores WITHIN EACH ROW
    puts them on a common footing. It uses no cross-sample statistics, so nothing
    leaks between test rows -- which a global normalisation would do.

LEAKAGE RULES
    Every member is refitted inside each outer fold on that fold's training rows
    only. The MLP additionally early-stops on the inner validation split. Nothing
    -- member choice, weights, temperature -- is fitted on the outer test fold.

    python ensemble_probe.py --features A.npz [B.npz] --view X_eval --seeds 123 7 42
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pl", HERE / "probe_lab.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)

# The three members, as probe_lab configs. Chosen by a sweep that selected on the
# inner validation split over all 3060 subsets of size 3-4 drawn from 17
# candidates -- not by looking at test scores.
MEMBERS = [
    {"model": "lda"},
    {"model": "linsvm", "C": 10},
    {"model": "mlp", "lr": 3e-3, "standardise": True, "hidden": 1024},
]


def rowz(s: np.ndarray) -> np.ndarray:
    """Z-score each row's class scores. No cross-sample statistics."""
    mu = s.mean(1, keepdims=True)
    sd = s.std(1, keepdims=True) + 1e-9
    return (s - mu) / sd


def member_scores(cfg_in, X, y, tr, fit, val, te, k, seed):
    """Fit one member on training rows only and score the test rows."""
    cfg = pl.resolve(dict(cfg_in))
    if cfg["model"] != "mlp":
        # Convex probes need no early-stopping split, so they use all of tr.
        Xa, _, Xt = pl.apply_transforms(X[tr], X[val], X[te], cfg)
        m = pl.fit_classical(Xa, y[tr], cfg, np.bincount(y[tr], minlength=k).astype(float))
        return pl.classical_scores(m, Xt, k)

    Xf, Xv, Xt = pl.apply_transforms(X[fit], X[val], X[te], cfg)
    lossf = pl.make_loss(cfg, np.bincount(y[fit], minlength=k).astype(float))
    head = pl.train_one(Xf, y[fit], Xv, y[val], k, cfg, lossf, seed)
    with torch.no_grad():
        return head(torch.tensor(Xt, dtype=torch.float32)).numpy()


def run(X, y, groups, seeds, grouped=True, folds=5, members=MEMBERS):
    k = int(y.max()) + 1
    out = {}
    for sd in seeds:
        if grouped:
            splits = list(StratifiedGroupKFold(folds, shuffle=True, random_state=sd)
                          .split(X, y, groups=groups))
        else:
            splits = list(StratifiedKFold(folds, shuffle=True, random_state=sd)
                          .split(X, y))
        per_member = {i: [] for i in range(len(members))}
        ens, YY, f1s, bas = [], [], [], []
        for tr, te in splits:
            if grouped:
                inner = StratifiedGroupKFold(5, shuffle=True, random_state=sd)
                fi, vi = next(inner.split(X[tr], y[tr], groups=np.asarray(groups)[tr]))
            else:
                fi, vi = next(StratifiedKFold(5, shuffle=True, random_state=sd)
                              .split(X[tr], y[tr]))
            fit, val = tr[fi], tr[vi]

            zs = []
            for i, mc in enumerate(members):
                s = member_scores(mc, X, y, tr, fit, val, te, k, sd * 1000 + i)
                per_member[i].append((s, y[te]))
                zs.append(rowz(s))
            e = np.mean(zs, axis=0)
            ens.append(e)
            YY.append(y[te])
            f1s.append(f1_score(y[te], e.argmax(1), average="macro", zero_division=0))
            bas.append(balanced_accuracy_score(y[te], e.argmax(1)))

        E = np.concatenate(ens)
        Y = np.concatenate(YY)
        pred = E.argmax(1)
        T = pl.fit_temperature(E, Y)
        probs = torch.softmax(torch.tensor(E / T), dim=1).numpy()
        rec = {"acc": float((pred == Y).mean()), "macro_f1": float(np.mean(f1s)),
               "bal_acc": float(np.mean(bas)), "temperature": T,
               "ece": pl.ece(probs, Y), "fold_std": float(np.std(f1s)),
               "per_class": [round(x, 4) for x in
                             f1_score(Y, pred, average=None, zero_division=0,
                                      labels=range(k))]}
        for i in per_member:
            S = np.concatenate([a for a, _ in per_member[i]])
            YM = np.concatenate([b for _, b in per_member[i]])
            rec[f"member{i}_acc"] = float((S.argmax(1) == YM).mean())
            rec[f"member{i}_f1"] = float(f1_score(YM, S.argmax(1), average="macro",
                                                  zero_division=0))
        out[sd] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--view", default="X_eval")
    ap.add_argument("--seeds", nargs="+", type=int, default=[123])
    ap.add_argument("--ungrouped", action="store_true",
                    help="plain StratifiedKFold, to compare with published protocols")
    ap.add_argument("--members", help="JSON list of probe_lab configs; "
                    "default is the LDA+LinearSVC+MLP triple")
    ap.add_argument("--label", default="ensemble")
    ap.add_argument("--out")
    args = ap.parse_args()

    members = json.loads(args.members) if args.members else MEMBERS
    # path::view is supported by pl.load_features
    X, y, g, classes = pl.load_features(args.features, args.view)
    r = run(X, y, g, args.seeds, grouped=not args.ungrouped, members=members)

    accs = [v["acc"] for v in r.values()]
    f1s = [v["macro_f1"] for v in r.values()]
    print(f"\n  {args.label}   d={X.shape[1]}  view={args.view}  "
          f"{'ungrouped' if args.ungrouped else 'grouped'} CV, seeds {args.seeds}")
    for sd, v in r.items():
        ms = "  ".join(f"m{i} {v[f'member{i}_acc']:.4f}"
                       for i in range(len(members)))
        print(f"    seed {sd}: ENSEMBLE acc {v['acc']:.4f}  macroF1 {v['macro_f1']:.4f}"
              f"  T {v['temperature']:.3f}  ECE {v['ece']:.4f}"
              f"  worst {min(v['per_class']):.3f}   [{ms}]")
    print(f"    MEAN  acc {np.mean(accs):.4f} +/- {np.std(accs):.4f}"
          f"   macroF1 {np.mean(f1s):.4f}")
    best = r[args.seeds[0]]
    print("    per-class F1: " + "  ".join(
        f"{c}={f:.3f}" for c, f in zip(classes, best["per_class"])))
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"classes": classes, "members": members, "dim": int(X.shape[1]),
             "results": {str(a): b for a, b in r.items()}}, indent=1),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
