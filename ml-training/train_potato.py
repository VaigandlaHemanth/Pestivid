"""
Potato leaf disease classification — leak-free training pipeline.

Replaces the CLIP+text pipeline in potatoleaf-vlm-fc93c1.ipynb, which fed the
ground-truth label into the model and therefore never measured image
classification at all. See WHY_THE_OLD_NUMBER_WAS_WRONG below.

Run on a GPU box (Kaggle P100 / Colab T4 is plenty — the backbone is frozen):

    python train_potato.py --data-root /kaggle/input/potato-leaf-disease-dataset/... \\
                           --backbone dinov2 --folds 5

Outputs, all under --out:
    features_<backbone>.npz     cached frozen features (eval transform)
    duplicate_groups.json       near-duplicate clusters used for grouped splits
    fold_<k>_head.pt            trained head per fold
    calibration.json            fitted temperature + OOD threshold
    metrics.json                per-fold and aggregate macro-F1 / accuracy / ECE
    confusion_matrix.png
    reliability.png
    model_card.md

--------------------------------------------------------------------------------
WHY THE OLD NUMBER WAS WRONG
--------------------------------------------------------------------------------
The previous CLIPFineTuner did:

    combined = image_features * 0.7 + text_features * 0.3
    logits   = classifier(combined)

and its Dataset chose the text with `text = self.text_prompts[label]` — the
ground-truth label. Because `unfreeze_layers=2` unfroze only the last two
*vision* layers, the whole text tower stayed frozen, and there were exactly 7
prompt strings. So `text_features` was a fixed lookup table of 7 constant
vectors, i.e. a class-indexed additive constant with **zero** within-class
variance. Formally H(y | combined) = 0: the input contained the answer.

The leak was present in the val and test datasets too, so checkpoint selection
ran through it as well. The reported 84.10% is therefore an upper bound of
unknown tightness on image-only accuracy, and nothing in the old repo bounds it
from below.

Note the coefficient is a red herring: the first nn.Linear can rescale that
direction freely, so even `text_features * 0.001` would leak completely.

This pipeline has no text branch. Nothing but pixels reaches the model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

SEED = 123

# The dataset of record (Shabrina et al. 2023, Data in Brief) tabulates 3,076
# original images. The old notebook globbed "*/*.jpg" -- one directory level
# deep, lowercase only -- and found 1,885, i.e. ~61%. See collect_images().
EXPECTED_TOTAL = 3076
# NOTE the spelling. The Kaggle dataset's directory is "Phytopthora" -- missing
# the second 'h' -- and the deployed flask_server.py carried the same misspelling.
# Verified against the real download: 569/748/201/68/611/347/532 = 3076 exactly.
# Both spellings are accepted so the per-class check works either way, and the
# class label is whatever the directory is actually called.
EXPECTED_PER_CLASS = {
    "Bacteria": 569, "Fungi": 748, "Healthy": 201, "Nematode": 68,
    "Pest": 611, "Phytopthora": 347, "Phytophthora": 347, "Virus": 532,
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# 4.1  Find every image, and say so if the count is unexpected
# ─────────────────────────────────────────────────────────────────────────────
def collect_images(root: Path) -> tuple[list[Path], list[str], list[str]]:
    """Recursive, case-insensitive, multi-extension. The old `glob("*/*.jpg")`
    was one level deep and case-sensitive, which silently dropped ~39%."""
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    paths, labels = [], []
    for cls in classes:
        for p in sorted((root / cls).rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
                labels.append(cls)

    counts = Counter(labels)
    print(f"Found {len(paths)} images across {len(classes)} classes: {classes}")
    for cls in classes:
        exp = EXPECTED_PER_CLASS.get(cls)
        flag = "" if exp is None else ("  OK" if counts[cls] == exp else f"  <-- expected {exp}")
        print(f"  {cls:<14} {counts[cls]:>5}{flag}")

    if len(paths) != EXPECTED_TOTAL:
        print(f"\n  WARNING: expected {EXPECTED_TOTAL} images, found {len(paths)}.")
        print("  If this is lower, check for nested subdirectories or uppercase")
        print("  extensions. Do not train on a silently truncated dataset.\n")
    return paths, labels, classes


# ─────────────────────────────────────────────────────────────────────────────
# 4.6a  Near-duplicate grouping, so the same plant cannot straddle a split
# ─────────────────────────────────────────────────────────────────────────────
def dhash(img: Image.Image, size: int = 8) -> int:
    """Difference hash. Cheap, and robust to the small exposure/scale changes
    between consecutive field photographs of one plant."""
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def group_near_duplicates(paths: list[Path], max_hamming: int = 6) -> list[int]:
    """Union-find over dHash distance. Returns a group id per image.

    This matters because the dataset is field photography: multiple frames of
    the same plant almost certainly exist. A plain stratified split puts near
    duplicates on both sides, which inflates the score independently of the
    label leak. Grouped splitting removes that second inflation.
    """
    hashes = []
    for p in paths:
        with Image.open(p) as im:
            hashes.append(dhash(im))

    parent = list(range(len(paths)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    # Bucket by high bits first so this stays ~linear instead of O(n^2).
    buckets: dict[int, list[int]] = defaultdict(list)
    for idx, h in enumerate(hashes):
        for shift in (0, 16, 32, 48):
            buckets[(h >> shift) & 0xFFFF].append(idx)

    seen: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2 or len(members) > 400:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                if bin(hashes[i] ^ hashes[j]).count("1") <= max_hamming:
                    union(i, j)

    groups = [find(i) for i in range(len(paths))]
    n_groups = len(set(groups))
    dup = len(paths) - n_groups
    print(f"Near-duplicate grouping: {n_groups} groups for {len(paths)} images "
          f"({dup} images share a group with another)")
    if dup:
        print("  These are kept together in every fold. A plain stratified split")
        print("  would have leaked them across train/test.")
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Backbones — frozen feature extractors, no text branch anywhere
# ─────────────────────────────────────────────────────────────────────────────
BACKBONES = {
    # A 2026 study (Backbone Diversity Beats Text Supervision, Front. Plant Sci.)
    # measured, on PlantWild with a linear probe: DINOv2 ViT-L/14 73.05 top-1 vs
    # CLIP ViT-B/16 52.41. It also ablated text supervision at +2.5 points
    # against backbone choice at +6.7 -- so the branch that caused the leak was
    # also the least valuable part of the old design.
    "dinov2": ("facebook/dinov2-large", 1024),
    "dinov2-base": ("facebook/dinov2-base", 768),
    "clip": ("openai/clip-vit-base-patch32", 512),  # the old backbone, for comparison
}


class FrozenBackbone(nn.Module):
    """Wraps a pretrained vision encoder. Always eval, always no_grad."""

    def __init__(self, name: str, device: torch.device):
        super().__init__()
        repo, self.dim = BACKBONES[name]
        self.kind = "clip" if name == "clip" else "dinov2"
        if self.kind == "clip":
            from transformers import CLIPModel
            m = CLIPModel.from_pretrained(repo)
            self.vision = m.vision_model
            self.proj = m.visual_projection
        else:
            from transformers import AutoModel
            self.vision = AutoModel.from_pretrained(repo)
            self.proj = None
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self.to(device)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.kind == "clip":
            feats = self.proj(self.vision(pixel_values=pixel_values)[1])
        else:
            out = self.vision(pixel_values=pixel_values)
            feats = out.pooler_output if out.pooler_output is not None \
                else out.last_hidden_state[:, 0]
        return F.normalize(feats, dim=-1)


class Head(nn.Module):
    """Linear probe by default. `--hidden N` adds one hidden layer.

    Deliberately small: with ~3k images a bigger head just overfits, and the
    old 512->512->7 MLP at lr=2e-5 for ~1k steps was badly undertrained anyway.
    """

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 0, p_drop: float = 0.2):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(p_drop),
                nn.Linear(hidden, n_classes))
        else:
            self.net = nn.Sequential(nn.Dropout(p_drop), nn.Linear(in_dim, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4.4  Augmentation that actually runs
# ─────────────────────────────────────────────────────────────────────────────
# The old notebook defined get_image_transforms() with flips and rotation, but
# every reference to it was commented out; the live dataset handed the raw PIL
# image to CLIPProcessor, which resized and normalised and discarded everything
# else. Net effect: zero augmentation on a ~2k-image problem.
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)


def build_transforms(kind: str, train: bool, size: int = 224):
    mean, std = (CLIP_MEAN, CLIP_STD) if kind == "clip" else (IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(),   # tuning-free; matches AutoAugment on most benchmarks
            T.ToTensor(),
            T.Normalize(mean, std),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    # Pinned to match Xenova/dinov2-base's preprocessor_config.json so the
    # browser build produces byte-comparable features: shortest_edge 256 and
    # BICUBIC, not 255 and torchvision's default BILINEAR.
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(), T.Normalize(mean, std),
    ])


class LeafDataset(Dataset):
    """Images and integer labels. No text, no prompts, no label-derived input."""

    def __init__(self, paths, labels, label_map, tf):
        self.paths, self.labels, self.label_map, self.tf = paths, labels, label_map, tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tf(ImageOps.exif_transpose(im).convert("RGB"))
        return x, self.label_map[self.labels[i]]


# ─────────────────────────────────────────────────────────────────────────────
# 4.5  Class-balanced loss
# ─────────────────────────────────────────────────────────────────────────────
def class_balanced_weights(counts: np.ndarray, beta: float = 0.999) -> torch.Tensor:
    """Cui et al. 2019 effective-number weighting.

    Fungi 748 : Nematode 68 is ~11:1. Under plain cross-entropy a model that
    never predicts Nematode loses ~2.2% accuracy and gains apparent stability,
    so the old unweighted loss actively rewarded abandoning the rarest class.
    """
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / np.maximum(eff, 1e-12)
    w = w / w.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32)


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin (Cao et al. 2019, arXiv:1906.07413).

    KEPT AS AN OPTION, NOT THE DEFAULT -- and the reason is worth reading before
    reaching for it.

    LDAM first looked like a large win here: macro-F1 0.7169 -> 0.8108 over the
    class-balanced weighting it replaced. That reading was wrong about the
    mechanism. Ablating its two parts separately on identical folds
    (ablate_underfit.py):

        LDAM, margin + s=30                     0.8108
        s=30 only, plain CE, NO margin          0.8265   <-- better without it
        margin only, s=1, NO scale              0.6839   <-- worse than nothing
        plain CE, no scale, no margin           0.6964

    The margin contributes nothing positive on this dataset; removing it IMPROVES
    the result, and applying it without the scale is worse than doing nothing.
    The entire apparent gain was s=30 multiplying the gradient, i.e. a learning
    rate fix wearing a long-tail costume.

    The real fault was that the head was underfit. The cached features are exactly
    unit-norm with a per-dim std of 0.024, so Linear(768,7) at default init emits
    logits near 0.03 and lr=1e-3 under cosine decay never escapes the
    near-uniform-softmax regime. The tell was already in our own artifacts:
    fitted temperatures of 0.476-0.519 mean temperature scaling was asked to
    SHARPEN the logits about 2x, the signature of systematic under-confidence.
    ECE 0.0401 hid it, because scaling repairs confidence and cannot repair the
    argmax. Raising lr to 0.1 fixes it directly and moves the fitted temperature
    to 0.85.

    Once the fit is repaired, every imbalance remedy measured here COSTS accuracy:
    CB beta=0.999 -0.013, LDAM -0.015. So the class-balanced weighting had been
    silently compensating for the underfit by translating the decision boundary --
    which is what produced tail over-prediction (Nematode precision 0.30) and a
    starved large class (Pest recall 0.52) at the same time.

    Do NOT stack this with post-hoc logit adjustment: s already sharpens the
    logits and the two corrections compound catastrophically (macro-F1 0.0198).
    """

    def __init__(self, counts: np.ndarray, max_m: float = 0.5, s: float = 30.0,
                 weight: "torch.Tensor | None" = None):
        super().__init__()
        m = 1.0 / np.sqrt(np.sqrt(np.maximum(counts, 1.0)))
        m = m * (max_m / m.max())
        self.register_buffer("m_list", torch.tensor(m, dtype=torch.float32))
        self.s = s
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        margins = self.m_list[target]
        onehot = F.one_hot(target, logits.shape[1]).bool()
        adjusted = logits - onehot.float() * margins[:, None]
        return F.cross_entropy(self.s * adjusted, target, weight=self.weight)


# ─────────────────────────────────────────────────────────────────────────────
# 4.6b  Temperature scaling
# ─────────────────────────────────────────────────────────────────────────────
def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """One scalar fitted on held-out NLL (Guo et al., ICML 2017).

    Does not change a single prediction -- argmax is invariant to a positive
    scale -- but makes the number shown to a farmer mean something. The old
    pipeline had no calibration at all, and its "confidence" was a softmax over
    7 scalars harvested from 7 *separate* forward passes, which is not a
    probability of anything.
    """
    targets = targets.long()          # cross_entropy needs Long
    logits = logits.float()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp().item())
    # Clamp. If the validation split happens to be perfectly separable the
    # optimum is T -> 0 (infinite confidence is optimal when never wrong), which
    # makes logits/T overflow at inference. A degenerate fit is also a signal
    # that the validation split is too easy or too small to calibrate on.
    if not (0.05 <= t <= 10.0):
        print(f"  WARNING: temperature fit returned {t:.4f}; clamping. "
              "Check that the validation split is large enough and not trivially separable.")
        t = min(max(t, 0.05), 10.0)
    return t


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == targets).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# 4.7  Out-of-distribution gate
# ─────────────────────────────────────────────────────────────────────────────
def fit_mahalanobis(feats: np.ndarray, labels: np.ndarray, n_classes: int):
    """Class-conditional Gaussian with a shared covariance, on L2-normalised
    features ("Mahalanobis++", 2025 — the normalisation is the whole trick).

    Nearly free once the backbone is frozen, and it is what stops the app
    confidently diagnosing a potato disease on a photo of a cat, a hand or a
    tomato leaf. The old pipeline had no reject option at all, which for a tool
    that then recommends a pesticide is the highest real-world-harm path in it.
    """
    means, centred = [], []
    for c in range(n_classes):
        fc = feats[labels == c]
        mu = fc.mean(0)
        means.append(mu)
        centred.append(fc - mu)
    means = np.stack(means)
    centred = np.concatenate(centred)
    cov = np.cov(centred, rowvar=False) + 1e-6 * np.eye(feats.shape[1])
    return means, np.linalg.pinv(cov)


def pathlib_name(p):
    import os
    return os.path.basename(str(p))


def fit_convex_head(X: np.ndarray, y: np.ndarray, kind: str, C: float,
                    n_classes: int, device) -> "Head":
    """Fit a closed-form / convex probe and return it as an ordinary linear Head.

    WHY THIS EXISTS
        The SGD head was underfit for a structural reason -- the features are
        L2-normalised to exactly 1.0 with a per-dimension std of 0.024, so a
        freshly initialised Linear(768,7) emits logits near 0.03 and lr=1e-3 with
        cosine decay never escapes that regime (fitted temperature 0.505, i.e.
        scaling had to double the logits). A convex solver cannot have that
        failure mode at all: there is no learning rate, no early stopping, and no
        seed. Measured on identical grouped folds, base features:

            SGD MLP-1024, standardised, lr 1e-3     0.8635 accuracy
            LinearSVC C=10                          0.8579
            LDA (lsqr, shrinkage='auto')            0.8622   <-- and simpler

    WHY IT COSTS NOTHING TO DEPLOY
        LDA and LinearSVC are LINEAR: decision_function(x) == x @ coef_.T +
        intercept_, verified to 0.000e+00 max absolute error. So the fitted probe
        is packed straight into the same state_dict the SGD head produces, and
        potato_infer.py, export_browser.py and the browser's linear() need no
        changes whatsoever. 21 KB for a 7x768 head.

        A non-linear probe (RBF-SVM scored 0.8455 macro-F1) would need support
        vectors and a kernel evaluation shipped to the browser, which is why it is
        not offered here despite scoring well on macro-F1.
    """
    if kind == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        # Ledoit-Wolf shrinkage, not plain LDA: 768 dimensions against ~2000
        # samples makes the unshrunk within-class covariance badly conditioned.
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif kind == "linsvm":
        from sklearn.svm import LinearSVC
        m = LinearSVC(C=C, max_iter=20000, dual="auto")
    elif kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=C, max_iter=5000, multi_class="multinomial")
    else:
        raise ValueError(f"{kind!r} is not a linear probe and cannot be packed "
                         f"into a linear head.")
    m.fit(X.astype(np.float64), y)

    head = Head(X.shape[1], n_classes, 0).to(device)
    W = np.ascontiguousarray(m.coef_, dtype=np.float32)
    b = np.ascontiguousarray(m.intercept_, dtype=np.float32)
    if W.shape[0] == 1 and n_classes == 2:      # binary solvers emit one row
        W = np.vstack([-W, W]); b = np.array([-b[0], b[0]], dtype=np.float32)
    with torch.no_grad():
        head.net[1].weight.copy_(torch.from_numpy(W))
        head.net[1].bias.copy_(torch.from_numpy(b))
    head.eval()
    return head


def mahalanobis_score(feats: np.ndarray, means: np.ndarray, prec: np.ndarray) -> np.ndarray:
    """Smallest Mahalanobis distance to any class centroid. Lower = more in-distribution."""
    d = np.empty((feats.shape[0], means.shape[0]))
    for c in range(means.shape[0]):
        z = feats - means[c]
        d[:, c] = np.einsum("ij,jk,ik->i", z, prec, z)
    return d.min(1)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract(backbone: FrozenBackbone, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    feats, ys = [], []
    for x, y in loader:
        feats.append(backbone(x.to(device, non_blocking=True)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


def train_head(tr_f, tr_y, va_f, va_y, n_classes, weights, args, device):
    head = Head(tr_f.shape[1], n_classes, args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if args.loss == "ldam":
        # counts must come from the FIT split, not the whole dataset, or the
        # margins are computed against a distribution the head never saw.
        fit_counts = np.bincount(tr_y, minlength=n_classes).astype(float)
        lossf = LDAMLoss(fit_counts, args.ldam_max_m, args.ldam_s).to(device)
    elif args.loss == "class_balanced":
        lossf = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)
    else:
        lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    tr_f_t = torch.tensor(tr_f, dtype=torch.float32)
    tr_y_t = torch.tensor(tr_y, dtype=torch.long)
    va_f_t = torch.tensor(va_f, dtype=torch.float32).to(device)
    va_y_t = torch.tensor(va_y, dtype=torch.long).to(device)

    best_f1, best_state, patience = -1.0, None, 0
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(tr_f_t))
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = tr_f_t[idx].to(device); yb = tr_y_t[idx].to(device)
            opt.zero_grad()
            loss = lossf(head(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            pred = head(va_f_t).argmax(1).cpu().numpy()
        # Selection is on macro-F1, not accuracy: on an 11:1 imbalance accuracy
        # barely moves when the rarest class is abandoned entirely.
        f1 = f1_score(va_y, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, patience = f1, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                break

    head.load_state_dict(best_state)
    return head, best_f1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="directory containing one subdirectory per class")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--backbone", default="dinov2", choices=list(BACKBONES))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--extract-batch", type=int, default=32)
    # 1e-3 UNDERFITS this head badly. The features are L2-normalised to exactly
    # 1.0 (per-dim std 0.024), so the logits start near 0.03 and cosine decay with
    # early stopping never lets them grow. Measured on identical folds, plain CE:
    #     lr 1e-3  macro-F1 0.6964, fitted T 0.53   (T<1 = under-confident)
    #     lr 1e-1  macro-F1 0.8295, fitted T 0.85
    # Per-fold feature standardisation reaches the same place (0.8193) but has to
    # ship mu/sigma to every inference path including the browser, so the learning
    # rate is the cheaper fix for an identical result.
    ap.add_argument("--lr", type=float, default=1e-1)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=0, help="0 = linear probe")
    # 'lda' is the default because it measured highest on grouped CV (0.8622 vs
    # 0.8635 for the best SGD MLP, which needs standardisation statistics shipped
    # to every inference path), is deterministic, has no learning rate to get
    # wrong, and exports as the same 21 KB linear head. See fit_convex_head().
    ap.add_argument("--probe", default="lda", choices=["lda", "linsvm", "logreg", "sgd"],
                    help="lda is the default; 'sgd' selects the trained MLP path")
    ap.add_argument("--probe-C", type=float, default=10.0)
    ap.add_argument("--use-cached", action="store_true",
                    help="reuse features_<backbone>.npz if it matches this dataset")
    # Fit the head on features produced elsewhere, e.g. extract_views.py's tiled
    # views. Each entry is "path" or "path::view1+view2"; blocks are concatenated
    # in the order given, and every file must be row-aligned (same y and groups).
    #
    # This is how the tiling configuration is trained.
    ap.add_argument("--features", nargs="+", default=None,
                    help='explicit feature files, "path::view1+view2"; '
                         "concatenated in order. Skips extraction.")
    ap.add_argument("--feature-label", default=None,
                    help="name recorded in the checkpoint for this feature recipe")
    # plain + lr 0.1 measured 0.8295 macro-F1; ldam 0.8108; class_balanced 0.7169.
    # See LDAMLoss's docstring: the imbalance remedies were compensating for an
    # underfit head, and cost accuracy once the learning rate is right.
    ap.add_argument("--loss", default="plain", choices=["plain", "ldam", "class_balanced"],
                    help="plain is the default; the others are kept for comparison only")
    ap.add_argument("--ldam-max-m", type=float, default=0.5)
    ap.add_argument("--ldam-s", type=float, default=30.0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--tta", type=int, default=4,
                    help="augmented feature copies per training image (0 = none)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-dup-check", action="store_true")
    args = ap.parse_args()

    set_seed()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  backbone={args.backbone}")

    paths, labels, classes = collect_images(Path(args.data_root))
    label_map = {c: i for i, c in enumerate(classes)}
    # dtype=np.int64 is load-bearing, not cosmetic: numpy defaults to int32 on
    # Windows and F.cross_entropy requires Long, so omitting it makes training
    # crash on Windows and pass on Linux.
    y_all = np.array([label_map[l] for l in labels], dtype=np.int64)

    groups = ([0] * 0) or (list(range(len(paths))) if args.no_dup_check
                           else group_near_duplicates(paths))
    (out / "duplicate_groups.json").write_text(json.dumps(
        {str(p): int(g) for p, g in zip(paths, groups)}, indent=1), encoding="utf-8")

    # Re-use a previous extraction when one is present and matches. Extraction is
    # 25-90 minutes on CPU while fitting the head is seconds, so re-deriving
    # features to change a head hyperparameter wastes almost all of the runtime.
    # Explicit feature files take precedence over everything else.
    explicit = None
    if args.features:
        blocks = []
        for spec in args.features:
            path, _, view = str(spec).partition("::")
            z = np.load(path, allow_pickle=True)
            if not np.array_equal(z["y"].astype(np.int64), y_all):
                raise SystemExit(
                    f"{path}: y does not match this dataset. Feature files must be "
                    f"row-aligned with the images collected here.")
            views = [v for v in (view or "X_eval").split("+") if v]
            for v in views:
                key = v if v in z.files else {"cls": "X_eval", "X_eval": "cls"}.get(v)
                if key not in z.files:
                    raise SystemExit(f"{path}: no view {v!r}. Available: {list(z.files)}")
                blocks.append(z[key].astype(np.float32))
            print(f"  + {pathlib_name(path)}::{'+'.join(views)}"
                  f"  ({sum(b.shape[1] for b in blocks[-len(views):])}-d)")
        explicit = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
        print(f"Fitting on explicit features: {explicit.shape[0]}x{explicit.shape[1]}")

    cache = out / f"features_{args.backbone}.npz"
    cached = None
    if args.use_cached and cache.exists():
        z = np.load(cache, allow_pickle=True)
        ok = (z["X_eval"].shape[0] == len(paths)
              and np.array_equal(z["y"], y_all)
              and [str(c) for c in z["classes"].tolist()] == classes)
        if ok:
            cached = z
            print(f"Using cached features from {cache} "
                  f"({z['X_eval'].shape[0]}x{z['X_eval'].shape[1]}) -- "
                  f"skipping extraction.")
        else:
            # Never silently fall back to a cache that describes different data:
            # a stale cache would produce a confident number about the wrong
            # images. Re-extract instead.
            print(f"Cache {cache} does not match this dataset "
                  f"(images/labels/classes differ) -- re-extracting.")

    if explicit is not None:
        X_eval = explicit
        backbone = None
        if args.tta:
            raise SystemExit("--features cannot supply --tta augmented copies; "
                             "run with --tta 0.")
    elif cached is not None:
        X_eval = cached["X_eval"].astype(np.float32)
        backbone = None
        if args.tta:
            # Augmentation has to happen before the frozen backbone, so augmented
            # copies cannot be reconstructed from a cache. Refuse rather than
            # quietly training on fewer copies than asked for.
            raise SystemExit("--use-cached cannot supply --tta augmented copies; "
                             "run with --tta 0 or without --use-cached.")
    else:
        backbone = FrozenBackbone(args.backbone, device)
        # Eval-transform features: used for validation, testing, calibration and OOD.
        eval_ds = LeafDataset(paths, labels, label_map,
                              build_transforms(backbone.kind, train=False))
        eval_loader = DataLoader(eval_ds, batch_size=args.extract_batch,
                                 shuffle=False, num_workers=args.workers)
        print("Extracting eval features ...")
        X_eval, y_eval = extract(backbone, eval_loader, device)
        assert (y_eval == y_all).all(), "loader order drifted from path order"

    # Augmented copies for training. Augmentation has to happen before the frozen
    # backbone, so it cannot be applied to cached features -- we extract N copies.
    X_aug, y_aug, idx_aug = [X_eval], [y_all], [np.arange(len(paths))]
    if args.tta:
        train_ds = LeafDataset(paths, labels, label_map,
                               build_transforms(backbone.kind, train=True))
        for r in range(args.tta):
            print(f"Extracting augmented features {r + 1}/{args.tta} ...")
            loader = DataLoader(train_ds, batch_size=args.extract_batch,
                                shuffle=False, num_workers=args.workers)
            xf, yf = extract(backbone, loader, device)
            X_aug.append(xf); y_aug.append(yf); idx_aug.append(np.arange(len(paths)))
    X_aug = np.concatenate(X_aug); y_aug = np.concatenate(y_aug)
    idx_aug = np.concatenate(idx_aug)

    if explicit is None:
        np.savez_compressed(out / f"features_{args.backbone}.npz",
                            X_eval=X_eval, y=y_all, groups=np.array(groups),
                            classes=np.array(classes))

    counts = np.bincount(y_all, minlength=len(classes)).astype(float)
    weights = class_balanced_weights(counts)
    print("class-balanced weights:",
          {c: round(float(w), 3) for c, w in zip(classes, weights)})

    # StratifiedGroupKFold: stratified by label AND grouped so near-duplicates
    # never straddle a fold boundary.
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    per_fold, all_logits, all_targets = [], [], []

    for k, (tr_idx, te_idx) in enumerate(skf.split(X_eval, y_all, groups=groups)):
        # inner split for model selection + temperature, from TRAIN only
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        g_tr = np.array(groups)[tr_idx]
        fit_i, val_i = next(inner.split(X_eval[tr_idx], y_all[tr_idx], groups=g_tr))
        fit_idx, val_idx = tr_idx[fit_i], tr_idx[val_i]

        fit_mask = np.isin(idx_aug, fit_idx)   # augmented copies of fit images only
        if args.probe == "sgd":
            head, val_f1 = train_head(X_aug[fit_mask], y_aug[fit_mask],
                                      X_eval[val_idx], y_all[val_idx],
                                      len(classes), weights, args, device)
        else:
            # Fitted on the FIT split, not on fit+val. A convex probe needs no
            # early-stopping split and could use both, but the temperature has to
            # be fitted on data the probe did not see or it is biased toward
            # calling the probe already calibrated. Measured cost of holding the
            # split back: 0.8622 -> 0.8618 accuracy on base features, i.e. free.
            head = fit_convex_head(X_eval[fit_idx], y_all[fit_idx],
                                   args.probe, args.probe_C, len(classes), device)
            with torch.no_grad():
                vp = head(torch.tensor(X_eval[val_idx], dtype=torch.float32)
                          .to(device)).argmax(1).cpu().numpy()
            val_f1 = float(f1_score(y_all[val_idx], vp, average="macro",
                                    zero_division=0))

        head.eval()
        with torch.no_grad():
            val_logits = head(torch.tensor(X_eval[val_idx], dtype=torch.float32).to(device)).cpu()
            te_logits = head(torch.tensor(X_eval[te_idx], dtype=torch.float32).to(device)).cpu()

        temp = fit_temperature(val_logits,
                               torch.tensor(y_all[val_idx], dtype=torch.long))
        probs = F.softmax(te_logits / temp, dim=1).numpy()
        pred = probs.argmax(1)

        acc = accuracy_score(y_all[te_idx], pred)
        f1m = f1_score(y_all[te_idx], pred, average="macro", zero_division=0)
        ece = expected_calibration_error(probs, y_all[te_idx])
        per_fold.append(dict(fold=k, accuracy=acc, macro_f1=f1m, ece=ece,
                            temperature=temp, val_macro_f1=val_f1,
                            n_train=int(len(fit_idx)), n_test=int(len(te_idx))))
        print(f"fold {k}: acc={acc:.4f} macroF1={f1m:.4f} ECE={ece:.4f} T={temp:.3f}")

        torch.save({"head": head.state_dict(), "temperature": temp,
                    "backbone": args.backbone, "classes": classes,
                    "hidden": args.hidden if args.probe == "sgd" else 0,
                    "probe": args.probe, "feat_dim": X_eval.shape[1],
                    # What the head was fitted on. Inference MUST reproduce this
                    # exactly; a 2304-d tiled head fed a 768-d global vector would
                    # not error, it would just be wrong.
                    "feature_recipe": args.feature_label or (
                        "+".join(args.features) if args.features else "global_cls")},
                   out / f"fold_{k}_head.pt")
        all_logits.append(te_logits.numpy() / temp)
        all_targets.append(y_all[te_idx])

    # ── aggregate ──
    accs = np.array([f["accuracy"] for f in per_fold])
    f1s = np.array([f["macro_f1"] for f in per_fold])
    eces = np.array([f["ece"] for f in per_fold])
    logits = np.concatenate(all_logits); targets = np.concatenate(all_targets)
    pred = logits.argmax(1)

    summary = {
        "backbone": args.backbone,
        "n_images": len(paths),
        "n_groups": len(set(groups)),
        "folds": args.folds,
        "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
        "macro_f1_mean": float(f1s.mean()), "macro_f1_std": float(f1s.std()),
        "ece_mean": float(eces.mean()),
        "per_fold": per_fold,
        "classification_report": classification_report(
            targets, pred, target_names=classes, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(targets, pred).tolist(),
    }

    # ── OOD gate + abstention threshold, fitted on all in-distribution data ──
    means, prec = fit_mahalanobis(X_eval, y_all, len(classes))
    scores = mahalanobis_score(X_eval, means, prec)
    ood_threshold = float(np.percentile(scores, 99.0))
    probs_all = torch.softmax(torch.tensor(logits), dim=1).numpy()
    conf_threshold = float(np.percentile(probs_all.max(1), 10.0))

    # RISK-COVERAGE CURVE.
    #
    # Full-coverage accuracy is the wrong headline for this product. The output
    # drives a pesticide purchase, and the dominant confusion is Fungi vs Pest
    # (31% of all errors) -- which is fungicide vs insecticide, so a confidently
    # wrong answer sends a farmer to buy the wrong chemical. The model already
    # abstains below conf_threshold, so what actually reaches a farmer is the
    # accuracy on the photos it ANSWERS, and that is what this records.
    #
    # Measured on out-of-fold predictions, so it is an honest estimate rather
    # than a training-set number.
    conf_all = probs_all.max(1)
    correct = (probs_all.argmax(1) == targets)
    order = np.argsort(-conf_all)
    risk_coverage = []
    for cov in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7):
        n = max(1, int(round(cov * len(targets))))
        sel = order[:n]
        risk_coverage.append({
            "coverage": cov,
            "n_answered": int(n),
            "accuracy": float(correct[sel].mean()),
            "confidence_threshold": float(conf_all[order[n - 1]]),
        })
    print("\nrisk-coverage (accuracy on the photos the model answers):")
    for r in risk_coverage:
        print(f"  coverage {r['coverage']:5.0%}  n={r['n_answered']:5d}  "
              f"accuracy {r['accuracy']:.4f}  conf>= {r['confidence_threshold']:.4f}")
    np.savez_compressed(out / "ood.npz", means=means, precision=prec)
    (out / "calibration.json").write_text(json.dumps({
        "temperature_per_fold": [f["temperature"] for f in per_fold],
        "ood_mahalanobis_threshold_p99": ood_threshold,
        "abstain_below_confidence": conf_threshold,
        "risk_coverage": risk_coverage,
        "coverage_at_abstain_threshold": 0.90,
        "note": ("Reject when the Mahalanobis score exceeds the threshold "
                 "(not a potato leaf) or max prob falls below the confidence "
                 "floor (uncertain). Route both to a human rather than "
                 "returning a disease name and a pesticide."),
    }, indent=2), encoding="utf-8")

    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"macro-F1  {f1s.mean():.4f} +/- {f1s.std():.4f}   <-- headline metric")
    print(f"accuracy  {accs.mean():.4f} +/- {accs.std():.4f}")
    print(f"ECE       {eces.mean():.4f}")
    print("=" * 68)
    print("\nThese are grouped-CV numbers with no label leak, so expect them to be")
    print("LOWER than the old 84.10%. Published results on this dataset for")
    print("reference: EfficientNetV2B3 73.63, MobileNetV3-L 72.03, ResNet50 68.17;")
    print("current best EfficientNet-LITE + Kernel-Ensemble SVM 87.82.")
    print("Treat anything above ~88 here as a leak until proven otherwise.")

    try:
        _plots(summary, classes, probs_all, targets, out)
    except Exception as exc:                              # plotting is optional
        print(f"(plots skipped: {exc})")

    _model_card(summary, args, out, ood_threshold, conf_threshold)
    print(f"\nWrote artifacts to {out.resolve()}")


def _plots(summary, classes, probs, targets, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(summary["confusion_matrix"], dtype=float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix (row-normalised, pooled over folds)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)

    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == targets).astype(float)
    edges = np.linspace(0, 1, 16)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(xs, ys, "o-", label="after temperature scaling")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability (ECE {summary['ece_mean']:.3f})")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=150); plt.close(fig)


def _model_card(summary, args, out: Path, ood_thr: float, conf_thr: float) -> None:
    rep = summary["classification_report"]
    rows = "\n".join(
        f"| {c} | {rep[c]['precision']:.2f} | {rep[c]['recall']:.2f} | "
        f"{rep[c]['f1-score']:.2f} | {int(rep[c]['support'])} |"
        for c in summary["classification_report"] if c in
        [k for k in rep if isinstance(rep[k], dict) and k not in ("macro avg", "weighted avg")])

    (out / "model_card.md").write_text(f"""# Model card — potato leaf disease classifier

## What it is
Frozen **{args.backbone}** features with a {'linear probe' if not args.hidden else f'{args.hidden}-unit MLP head'}.
Image input only — there is no text branch, so the label cannot reach the model.

## Data
{summary['n_images']} images, {summary['n_groups']} near-duplicate groups,
7 classes. Kaggle "Potato Leaf Disease Dataset in Uncontrolled Environment".

## Evaluation
{summary['folds']}-fold **StratifiedGroupKFold** — stratified by class and
grouped so near-duplicate photographs of the same plant never straddle a split.

| Metric | Value |
|---|---|
| **macro-F1** | **{summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}** |
| accuracy | {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f} |
| ECE (after temperature scaling) | {summary['ece_mean']:.4f} |

macro-F1 is the headline metric, not accuracy: the class ratio is ~11:1, and
accuracy barely moves if the rarest class is abandoned entirely.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
{rows}

## Abstention
- Mahalanobis OOD score above **{ood_thr:.1f}** → not a potato leaf, reject.
- Max probability below **{conf_thr:.3f}** → uncertain, route to a human.

Both must be enforced before any treatment text is shown.

## Known limitations
- Nematode has ~68 images in the full dataset. Per-class figures for it rest on
  a small support and should be read with that in mind.
- Field photography from one collection campaign; performance on other regions,
  cameras and growth stages is unmeasured.
- Predicts exactly 7 coarse categories. It does not estimate severity, and it
  cannot identify a disease outside those 7.
- **Not a substitute for an agronomist.** Treatment text is a curated lookup,
  carries no dose or jurisdiction data, and is illustrative only.

## Supersedes
`potatoleaf-vlm-fc93c1.ipynb`, whose reported 84.10% was measured with the
ground-truth label present in the model input at train *and* test time. That
number is an upper bound of unknown tightness and should not be quoted.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
