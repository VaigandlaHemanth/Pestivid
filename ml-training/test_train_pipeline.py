"""
Integration test for train_potato.py — runs the WHOLE pipeline on CPU in seconds.

Why this exists: the training script had only been statically checked and
unit-tested per function. Every individual piece passed, but the orchestration
had never executed once — and that is exactly where the bugs live (index
alignment between augmented copies and fold indices, StratifiedGroupKFold
failing on small groups, artifact shapes, the inner-split logic).

How it avoids needing a GPU or the real dataset:
  * synthetic images written to a temp dir, with deliberate near-duplicates so
    the grouping code has something to find
  * the frozen backbone is REPLACED by a stub that returns deterministic
    pseudo-features derived from the image, so no weights are downloaded and no
    real inference runs. The features are class-separable, so the head can
    actually learn and the metrics are meaningful as a smoke signal.

This does not validate accuracy. It validates that the pipeline runs, that the
folds are grouped correctly, and that every artifact is written in the shape
potato_infer.py expects.

    python test_train_pipeline.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent

CLASSES = ["Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytophthora", "Virus"]
# Small but enough that StratifiedGroupKFold(5) has something to work with, and
# the rarest class stays rare so the class-balancing path is exercised.
PER_CLASS = {"Bacteria": 12, "Fungi": 14, "Healthy": 8, "Nematode": 6,
             "Pest": 12, "Phytophthora": 9, "Virus": 11}
FEAT_DIM = 64


def load_module():
    spec = importlib.util.spec_from_file_location("tp", HERE / "train_potato.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_dataset(root: Path) -> int:
    """Class-correlated images, with a near-duplicate pair in every class."""
    rng = np.random.default_rng(7)
    total = 0
    for ci, cls in enumerate(CLASSES):
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        n = PER_CLASS[cls]
        for i in range(n):
            # Class signal in the mean colour, plus noise.
            base = np.full((64, 64, 3), 30 + ci * 30, dtype=np.int16)
            base = base + rng.integers(-18, 18, (64, 64, 3))
            arr = np.clip(base, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{cls}_{i:03d}.jpg", quality=95)
            total += 1
            # every class gets one near-duplicate so grouping has work to do
            if i == 0:
                Image.fromarray(arr).save(d / f"{cls}_{i:03d}_copy.JPEG", quality=95)
                total += 1
    return total


class StubBackbone(torch.nn.Module):
    """Deterministic, class-separable features. No download, no real model."""

    def __init__(self, dim=FEAT_DIM):
        super().__init__()
        self.dim = dim
        self.kind = "dinov2"

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Mean channel intensity carries the class signal we baked in above.
        b = pixel_values.shape[0]
        mean = pixel_values.mean(dim=(2, 3))                      # (b, 3)

        # Generate on the CPU with a seeded generator, then move to whatever
        # device the batch is on.
        #
        # torch.Generator() is a CPU generator, so seeding it and calling
        # torch.randn without a device produced CPU tensors -- fine while this
        # only ever ran on CPU, and a hard failure the moment a machine with CUDA
        # ran it, because train_potato.py moves the batch to the detected device:
        #   RuntimeError: Expected all tensors to be on the same device,
        #   but found at least two devices, cuda:0 and cpu!
        #
        # Seeding a CUDA generator instead would make the stub's features depend
        # on the device, and several checks below compare feature values across
        # runs. CPU-then-move keeps them bit-identical everywhere.
        dev = pixel_values.device
        g = torch.Generator().manual_seed(1234)
        proj = torch.randn(3, self.dim, generator=g).to(dev)
        noise = torch.randn(b, self.dim, generator=g).to(dev)
        feats = mean @ proj
        feats = feats + 0.05 * noise
        return torch.nn.functional.normalize(feats, dim=-1)


def main() -> int:
    tp = load_module()
    tmp = Path(tempfile.mkdtemp(prefix="pestivid_pipe_"))
    data, out = tmp / "data", tmp / "artifacts"
    checks, failed = [], 0

    def ck(label, cond, extra=""):
        nonlocal failed
        if not cond:
            failed += 1
        checks.append((label, cond, extra))

    try:
        n_written = make_dataset(data)
        print(f"synthetic dataset: {n_written} images in {len(CLASSES)} classes\n")

        # ---- 1. collection ----
        paths, labels, classes = tp.collect_images(data)
        ck("collect_images finds every file, incl. .JPEG uppercase",
           len(paths) == n_written, f"{len(paths)} vs {n_written}")
        ck("classes discovered in sorted order", classes == sorted(CLASSES))

        # ---- 2. near-duplicate grouping ----
        groups = tp.group_near_duplicates(paths)
        ck("grouping returns one id per image", len(groups) == len(paths))
        ck("grouping actually merged the planted duplicates",
           len(set(groups)) < len(paths),
           f"{len(set(groups))} groups for {len(paths)} images")

        # ---- 3. class-balanced weights ----
        y = np.array([classes.index(l) for l in labels])
        counts = np.bincount(y, minlength=len(classes)).astype(float)
        w = tp.class_balanced_weights(counts).numpy()
        ck("rarest class gets the largest weight",
           int(np.argmax(w)) == int(np.argmin(counts)))

        # ---- 4. transforms ----
        tf_tr = tp.build_transforms("dinov2", train=True)
        tf_ev = tp.build_transforms("dinov2", train=False)
        with Image.open(paths[0]) as im:
            xt, xe = tf_tr(im.convert("RGB")), tf_ev(im.convert("RGB"))
        ck("train transform yields 3x224x224", tuple(xt.shape) == (3, 224, 224))
        ck("eval transform yields 3x224x224", tuple(xe.shape) == (3, 224, 224))
        ck("train transform is stochastic (augmentation live)",
           not torch.allclose(tf_tr(Image.open(paths[0]).convert("RGB")), xt))

        # ---- 5. run main() with the backbone stubbed out ----
        tp.FrozenBackbone = lambda name, device: StubBackbone()
        argv = sys.argv
        sys.argv = ["train_potato.py", "--data-root", str(data), "--out", str(out),
                    "--backbone", "dinov2", "--folds", "3", "--epochs", "12",
                    "--tta", "1", "--workers", "0", "--extract-batch", "16",
                    "--patience", "5"]
        try:
            tp.main()
        finally:
            sys.argv = argv

        # ---- 6. artifacts ----
        heads = sorted(out.glob("fold_*_head.pt"))
        ck("one head per fold written", len(heads) == 3, f"{len(heads)} heads")
        ck("metrics.json written", (out / "metrics.json").exists())
        ck("calibration.json written", (out / "calibration.json").exists())
        ck("ood.npz written", (out / "ood.npz").exists())
        ck("model_card.md written", (out / "model_card.md").exists())
        ck("duplicate_groups.json written", (out / "duplicate_groups.json").exists())

        m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        ck("macro_f1_mean present and in [0,1]",
           0.0 <= m["macro_f1_mean"] <= 1.0, f"{m['macro_f1_mean']:.3f}")
        ck("per-fold results recorded", len(m["per_fold"]) == 3)
        ck("confusion matrix is 7x7",
           len(m["confusion_matrix"]) == 7 and len(m["confusion_matrix"][0]) == 7)
        ck("learned something on separable features (macro-F1 > chance 0.14)",
           m["macro_f1_mean"] > 0.14, f"macro-F1 {m['macro_f1_mean']:.3f}")

        cal = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
        ck("temperature fitted per fold", len(cal["temperature_per_fold"]) == 3)
        ck("every temperature within the clamp [0.05, 10]",
           all(0.05 <= t <= 10.0 for t in cal["temperature_per_fold"]),
           str([round(t, 3) for t in cal["temperature_per_fold"]]))
        ck("OOD threshold present", "ood_mahalanobis_threshold_p99" in cal)
        ck("abstention floor present", "abstain_below_confidence" in cal)

        # ---- 7. checkpoint shape must match what potato_infer.py loads ----
        ckpt = torch.load(heads[0], map_location="cpu", weights_only=True)
        for key in ("head", "temperature", "backbone", "classes", "hidden", "feat_dim"):
            ck(f"checkpoint carries '{key}' (potato_infer.py reads it)", key in ckpt)
        ck("checkpoint classes match the dataset", list(ckpt["classes"]) == sorted(CLASSES))
        ck("feat_dim matches the stub backbone", ckpt["feat_dim"] == FEAT_DIM)

        # ---- 8. the head actually reloads and predicts ----
        head = tp.Head(ckpt["feat_dim"], len(ckpt["classes"]), ckpt["hidden"])
        head.load_state_dict(ckpt["head"])
        head.eval()
        with torch.no_grad():
            logits = head(torch.randn(4, ckpt["feat_dim"]))
        ck("reloaded head produces (4, 7) logits", tuple(logits.shape) == (4, 7))

        # ---- 9. OOD scorer works on the saved parameters ----
        z = np.load(out / "ood.npz")
        means, prec = z["means"], z["precision"]
        ck("OOD means shape is (7, feat_dim)", means.shape == (7, FEAT_DIM))
        in_d = tp.mahalanobis_score(means, means, prec)          # centroids themselves
        far = tp.mahalanobis_score(np.ones((3, FEAT_DIM)) * 9.0, means, prec)
        ck("OOD: far-away points score higher than centroids",
           far.mean() > in_d.mean(), f"{far.mean():.1f} vs {in_d.mean():.1f}")

        # ---- 10. no label leak reachable in the trained path ----
        src = (HERE / "train_potato.py").read_text(encoding="utf-8")
        code = src.split('"""', 2)[-1]                # drop the module docstring
        ck("no text_prompts[label] in executable code", "text_prompts[" not in code)
        ck("no image/text feature mixing in executable code",
           "text_features" not in code)

    finally:
        print()
        for label, ok, extra in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   [{extra}]" if extra else ""))
        print()
        print(f"{len(checks) - failed}/{len(checks)} checks passed")
        shutil.rmtree(tmp, ignore_errors=True)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
