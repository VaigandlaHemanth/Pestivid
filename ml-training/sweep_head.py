"""
Head-only experiment sweep on CACHED features. Seconds per run, no GPU, no
re-extraction.

Why this exists: the trained model scores macro-F1 0.7180, but the per-class
numbers say the problem is not simply "rare classes are hard":

    Nematode   prec 0.30  rec 0.88   n=68    <- massively OVER-predicted
    Healthy    prec 0.54  rec 0.91   n=201   <- over-predicted
    Pest       prec 0.80  rec 0.52   n=611   <- a LARGE class being MISSED

That is the signature of class-balanced weighting that has over-corrected: the
loss pays so much for a Nematode miss that the classifier floods that class, and
the mass has to come from somewhere -- Pest.

So the first thing to test is not "handle imbalance harder", it is "handle it
less hard, or handle it after training instead of during".

    python sweep_head.py --features ../ml-service/artifacts/features_dinov2-base.npz

Every configuration is evaluated with the SAME 5-fold StratifiedGroupKFold and
the same inner split as train_potato.py, so the numbers are comparable to the
0.7180 baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 123


def set_seed(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)


# ── loss variants ───────────────────────────────────────────────────────────
def cb_weights(counts, beta):
    """Cui et al. 2019 effective number. beta -> 1 means harsher re-weighting."""
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / np.maximum(eff, 1e-12)
    return w / w.sum() * len(counts)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma, self.weight = gamma, weight

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        lp = logp.gather(1, target[:, None]).squeeze(1)
        pt = p.gather(1, target[:, None]).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * lp
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin (Cao et al. 2019).

    Enforces a larger margin for rarer classes instead of simply upweighting
    them, which is exactly the distinction that matters here -- margins do not
    inflate a rare class's predicted frequency the way loss weights do.
    """

    def __init__(self, counts, max_m=0.5, s=30.0, weight=None):
        super().__init__()
        m = 1.0 / np.sqrt(np.sqrt(counts))
        m = m * (max_m / m.max())
        self.m = torch.tensor(m, dtype=torch.float32)
        self.s, self.weight = s, weight

    def forward(self, logits, target):
        m = self.m.to(logits.device)[target]
        oh = F.one_hot(target, logits.shape[1]).bool()
        adj = logits - oh.float() * m[:, None]
        return F.cross_entropy(self.s * adj, target, weight=self.weight)


class BalancedSoftmax(nn.Module):
    """Ren et al. 2020. Adds log(prior) to the logits DURING training, which is
    the principled version of the post-hoc prior correction below."""

    def __init__(self, counts):
        super().__init__()
        p = counts / counts.sum()
        self.log_prior = torch.tensor(np.log(p), dtype=torch.float32)

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.log_prior.to(logits.device), target)


# ── head ────────────────────────────────────────────────────────────────────
class Head(nn.Module):
    def __init__(self, d, k, hidden=0, p=0.2):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(p),
                                     nn.Linear(hidden, k))
        else:
            self.net = nn.Sequential(nn.Dropout(p), nn.Linear(d, k))

    def forward(self, x):
        return self.net(x)


def train_head(Xtr, ytr, Xva, yva, k, lossfn, epochs=150, lr=1e-3, wd=1e-4,
               hidden=0, patience=25, select="macro_f1"):
    head = Head(Xtr.shape[1], k, hidden)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            lossfn(head(Xt[idx]), yt[idx]).backward()
            opt.step()
        sch.step()
        head.eval()
        with torch.no_grad():
            pv = head(Xv).argmax(1).numpy()
        sc = (f1_score(yva, pv, average="macro", zero_division=0) if select == "macro_f1"
              else balanced_accuracy_score(yva, pv))
        if sc > best:
            best, bad = sc, 0
            best_state = {kk: v.clone() for kk, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head


def evaluate(name, X, y, groups, k, counts, make_loss, folds=5,
             post="none", hidden=0, select="macro_f1", tau=0.0):
    """post: 'none' | 'prior' (logit adjustment) | 'tau' (classifier normalisation)"""
    set_seed()
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    f1s, bas, all_logits, all_y = [], [], [], []
    for tr, te in skf.split(X, y, groups=groups):
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        fi, vi = next(inner.split(X[tr], y[tr], groups=np.asarray(groups)[tr]))
        fit, val = tr[fi], tr[vi]
        head = train_head(X[fit], y[fit], X[val], y[val], k,
                          make_loss(np.bincount(y[fit], minlength=k).astype(float)),
                          hidden=hidden, select=select)
        with torch.no_grad():
            lg = head(torch.tensor(X[te], dtype=torch.float32)).numpy()

        if post == "prior":
            # Menon et al. logit adjustment: subtract log(prior) at INFERENCE.
            # Rebalances without ever distorting the training objective.
            pri = counts / counts.sum()
            lg = lg - np.log(pri)[None, :]
        elif post == "tau":
            # Kang et al. tau-normalisation: shrink the classifier weight norms,
            # which are what actually inflate a re-weighted rare class.
            W = head.net[-1].weight.detach().numpy()
            nrm = np.linalg.norm(W, axis=1) ** tau
            lg = lg / np.maximum(nrm, 1e-9)[None, :]

        pred = lg.argmax(1)
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
        bas.append(balanced_accuracy_score(y[te], pred))
        all_logits.append(lg); all_y.append(y[te])

    lg = np.concatenate(all_logits); yy = np.concatenate(all_y)
    pred = lg.argmax(1)
    per = f1_score(yy, pred, average=None, zero_division=0, labels=range(k))
    return dict(name=name, macro_f1=float(np.mean(f1s)), std=float(np.std(f1s)),
                bal_acc=float(np.mean(bas)),
                acc=float((pred == yy).mean()), per_class=[round(float(x), 3) for x in per])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="sweep_results.json")
    ap.add_argument("--focus", default="", help="ldam = tune the winner")
    args = ap.parse_args()

    z = np.load(args.features, allow_pickle=True)
    X, y, groups = z["X_eval"], z["y"].astype(np.int64), z["groups"]
    classes = [str(c) for c in z["classes"]]
    k = len(classes)
    counts = np.bincount(y, minlength=k).astype(float)
    print(f"features {X.shape}  classes {classes}")
    print(f"counts   {counts.astype(int).tolist()}   imbalance {counts.max()/counts.min():.1f}:1\n")

    CFG = [
        # --- the current baseline, for comparison ---
        ("CB beta=0.999 (CURRENT)", lambda c: nn.CrossEntropyLoss(
            weight=torch.tensor(cb_weights(c, 0.999), dtype=torch.float32), label_smoothing=0.05), "none", 0, 0.0),
        # --- is the re-weighting too harsh? walk beta down ---
        ("CB beta=0.99", lambda c: nn.CrossEntropyLoss(
            weight=torch.tensor(cb_weights(c, 0.99), dtype=torch.float32), label_smoothing=0.05), "none", 0, 0.0),
        ("CB beta=0.9", lambda c: nn.CrossEntropyLoss(
            weight=torch.tensor(cb_weights(c, 0.9), dtype=torch.float32), label_smoothing=0.05), "none", 0, 0.0),
        ("sqrt-inv-freq", lambda c: nn.CrossEntropyLoss(
            weight=torch.tensor((1/np.sqrt(c))/(1/np.sqrt(c)).sum()*len(c), dtype=torch.float32),
            label_smoothing=0.05), "none", 0, 0.0),
        # --- the control: no weighting at all ---
        ("plain CE (no weights)", lambda c: nn.CrossEntropyLoss(label_smoothing=0.05), "none", 0, 0.0),
        # --- rebalance AFTER training instead of during ---
        ("plain CE + logit adj", lambda c: nn.CrossEntropyLoss(label_smoothing=0.05), "prior", 0, 0.0),
        ("plain CE + tau-norm 0.5", lambda c: nn.CrossEntropyLoss(label_smoothing=0.05), "tau", 0, 0.5),
        ("plain CE + tau-norm 1.0", lambda c: nn.CrossEntropyLoss(label_smoothing=0.05), "tau", 0, 1.0),
        # --- margin- and prior-based losses ---
        ("Balanced Softmax", lambda c: BalancedSoftmax(c), "none", 0, 0.0),
        ("LDAM", lambda c: LDAMLoss(c), "none", 0, 0.0),
        ("Focal g=2 (unweighted)", lambda c: FocalLoss(2.0), "none", 0, 0.0),
        ("CB-Focal beta=0.99", lambda c: FocalLoss(
            2.0, torch.tensor(cb_weights(c, 0.99), dtype=torch.float32)), "none", 0, 0.0),
        # --- capacity ---
        ("plain CE + MLP-512 + logit adj", lambda c: nn.CrossEntropyLoss(label_smoothing=0.05), "prior", 512, 0.0),
    ]

    if args.focus == "ldam":
        # LDAM won stage 1 by +0.094 macro-F1. Tune its two knobs (max margin and
        # the logit scale s), try it with an MLP head, and try stacking a post-hoc
        # prior correction on top -- these are the obvious follow-ups and each
        # costs seconds on cached features.
        CFG = [
            ("LDAM m=0.5 s=30 (stage-1 winner)", lambda c: LDAMLoss(c, 0.5, 30.0), "none", 0, 0.0),
            ("LDAM m=0.3 s=30", lambda c: LDAMLoss(c, 0.3, 30.0), "none", 0, 0.0),
            ("LDAM m=0.7 s=30", lambda c: LDAMLoss(c, 0.7, 30.0), "none", 0, 0.0),
            ("LDAM m=0.9 s=30", lambda c: LDAMLoss(c, 0.9, 30.0), "none", 0, 0.0),
            ("LDAM m=0.5 s=15", lambda c: LDAMLoss(c, 0.5, 15.0), "none", 0, 0.0),
            ("LDAM m=0.5 s=45", lambda c: LDAMLoss(c, 0.5, 45.0), "none", 0, 0.0),
            ("LDAM m=0.7 s=45", lambda c: LDAMLoss(c, 0.7, 45.0), "none", 0, 0.0),
            ("LDAM + MLP-512", lambda c: LDAMLoss(c, 0.5, 30.0), "none", 512, 0.0),
            ("LDAM m=0.7 + MLP-512", lambda c: LDAMLoss(c, 0.7, 30.0), "none", 512, 0.0),
            ("LDAM + MLP-1024", lambda c: LDAMLoss(c, 0.5, 30.0), "none", 1024, 0.0),
            ("LDAM + logit adj", lambda c: LDAMLoss(c, 0.5, 30.0), "prior", 0, 0.0),
            ("LDAM + MLP-512 + logit adj", lambda c: LDAMLoss(c, 0.5, 30.0), "prior", 512, 0.0),
            ("LDAM + CB w beta=0.99", lambda c: LDAMLoss(
                c, 0.5, 30.0, torch.tensor(cb_weights(c, 0.99), dtype=torch.float32)), "none", 0, 0.0),
        ]

    rows = []
    for name, mk, post, hidden, tau in CFG:
        r = evaluate(name, X, y, groups, k, counts, mk, args.folds, post, hidden, tau=tau)
        rows.append(r)
        print(f"  {name:<32} macroF1 {r['macro_f1']:.4f} +/-{r['std']:.4f}  "
              f"balAcc {r['bal_acc']:.4f}  acc {r['acc']:.4f}")

    rows.sort(key=lambda r: -r["macro_f1"])
    base = next((r for r in rows if "CURRENT" in r["name"] or "stage-1 winner" in r["name"]), rows[-1])

    print("\n" + "=" * 84)
    print(f"{'config':<32} {'macroF1':>9} {'vs base':>9} {'balAcc':>8} {'acc':>7}")
    print("=" * 84)
    for r in rows:
        d = r["macro_f1"] - base["macro_f1"]
        print(f"{r['name']:<32} {r['macro_f1']:>9.4f} {d:>+9.4f} {r['bal_acc']:>8.4f} {r['acc']:>7.4f}")

    best = rows[0]
    print("\nBEST:", best["name"])
    print(f"{'class':<14}{'base F1':>9}{'best F1':>9}{'delta':>8}   n")
    for i, c in enumerate(classes):
        print(f"  {c:<12}{base['per_class'][i]:>9.3f}{best['per_class'][i]:>9.3f}"
              f"{best['per_class'][i]-base['per_class'][i]:>+8.3f}   {int(counts[i])}")

    Path(args.out).write_text(json.dumps(
        {"classes": classes, "counts": counts.tolist(), "results": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
