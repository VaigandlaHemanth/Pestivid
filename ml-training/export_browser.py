"""
Export the trained classifier tail for BROWSER inference.

The point: the backbone is a frozen, unmodified public checkpoint that already
has an ONNX export on the Hub (Xenova/dinov2-base), and everything after it is
tiny — a Linear head, one temperature scalar, and a Mahalanobis distance. So the
browser can do the whole prediction and there is no ML server to host at all.

    python export_browser.py --artifacts ../ml-service/artifacts \
                             --out ../frontend/model

Writes:
    manifest.json      classes, dims, thresholds, backbone id, preprocessing
    head.bin           float32 head weights + bias, little-endian
    ood.bin            float32 Mahalanobis means + Cholesky factor

WHY A CHOLESKY FACTOR, NOT THE PRECISION MATRIX
    Mahalanobis distance is (x-mu)^T P (x-mu). Shipping P for 768 dims is a
    768x768 float32 blob = 2.36 MB. But P is symmetric positive-definite, so
    P = L L^T and the distance is just ||L^T (x-mu)||^2 — same number, and L is
    triangular so half of it is zeros we do not need to send. We ship the packed
    upper triangle: 768*769/2 floats = 1.18 MB, half the size, exact.

    A diagonal-only approximation would be 3 KB but throws away the feature
    correlations that make Mahalanobis better than Euclidean in the first place;
    --diag-only is offered for measurement, not as a default.

PREPROCESSING PARITY IS THE WHOLE BALLGAME
    The head is fitted to whatever the transform produces. Xenova/dinov2-base's
    preprocessor_config.json is {shortest_edge: 256, crop 224, resample: 3
    (BICUBIC), ImageNet mean/std}, and potato_infer.py / train_potato.py are now
    pinned to exactly that. If you change one side, change both, then re-run
    verify_parity() below.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

# The ONNX export that matches each trained backbone, and its embedding width.
# Getting this pair wrong is silent: the head multiplies a differently-shaped
# vector and returns confident nonsense.
ONNX_BACKBONE = {
    "dinov2":      ("Xenova/dinov2-large", 1024),
    "dinov2-base": ("Xenova/dinov2-base", 768),
    "clip":        ("Xenova/clip-vit-base-patch32", 512),
}

# Measured from the Hub, 2026-08-21. Quoted so the size cost is explicit rather
# than a surprise on a rural connection.
#
# The default is fp16, not the smallest option, and that is a measured choice.
# With browser preprocessing made bit-exact, feature drift versus the Python
# reference (mean cosine distance over 7 real images) came out:
#     fp16 3.15e-4 | fp32 3.17e-4 | q4f16 4.12e-2 | q8 8.78e-1
# q8 collapses rather than degrades and must not be shipped; q4f16 is usable only
# if a head is refitted on q4f16 features. See public/js/potato-browser.js.
ONNX_SIZES_MB = {
    "Xenova/dinov2-base":  {"q4f16": 49.5, "uint8": 87.0, "quantized": 91.0, "fp16": 173.5, "fp32": 346.6},
    "Xenova/dinov2-large": {"q4f16": 168.0, "uint8": 303.0, "fp32": 1210.0},
    "Xenova/dinov2-small": {"q4f16": 14.0, "uint8": 24.5, "fp32": 88.0},
}


def write_f32(path: Path, arrays: list[np.ndarray]) -> list[dict]:
    """Concatenate float32 arrays into one little-endian blob, return the layout."""
    layout, off = [], 0
    with open(path, "wb") as fh:
        for name, arr in arrays:
            a = np.ascontiguousarray(arr, dtype="<f4")
            fh.write(a.tobytes())
            layout.append({"name": name, "offset": off, "count": int(a.size),
                           "shape": list(a.shape)})
            off += a.size
    return layout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="fp16",
                    choices=["q4f16", "uint8", "quantized", "fp16", "fp32"])
    ap.add_argument("--diag-only", action="store_true",
                    help="ship only the precision diagonal (3 KB, less accurate OOD)")
    args = ap.parse_args()

    art, out = Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    heads = sorted(art.glob("fold_*_head.pt"))
    if not heads:
        print(f"No fold_*_head.pt in {art}. Train first.")
        return 2

    cks = [torch.load(h, map_location="cpu", weights_only=True) for h in heads]
    first = cks[0]
    classes = [str(c) for c in first["classes"]]
    backbone = first["backbone"]
    feat_dim = int(first["feat_dim"])
    hidden = int(first.get("hidden", 0))

    if backbone not in ONNX_BACKBONE:
        print(f"No known ONNX export for backbone '{backbone}'.")
        return 2
    onnx_id, expect_dim = ONNX_BACKBONE[backbone]

    # A multi-view head is a MULTIPLE of the backbone width: the browser runs the
    # backbone several times and concatenates. So the check is not "equal" but
    # "an exact multiple, and the recipe says which views".
    recipe = str(first.get("feature_recipe") or "global_cls")
    if feat_dim % expect_dim != 0:
        print(f"Dimension mismatch: {backbone} head expects {feat_dim}, "
              f"{onnx_id} emits {expect_dim}, which does not divide it. "
              f"Refusing to export.")
        return 2
    n_views = feat_dim // expect_dim

    # Translate the recorded recipe into the view list the browser must produce,
    # IN ORDER. Order is load-bearing: the head's columns were fitted to this
    # exact concatenation, and a permuted vector would score confidently wrong.
    VIEW_MAP = [
        ("tile2x2_mean_patch", "tile2x2_mean_patch"),
        ("tile2x2_mean_cls",   "tile2x2_mean_cls"),
        ("global_cls",         "global_cls"),
        ("global_mean",        "global_mean"),
    ]
    views = []
    for token in recipe.replace("+", " ").split():
        for key, name in VIEW_MAP:
            if token.strip() == key:
                views.append(name)
    # Preserve the recipe's own ordering rather than VIEW_MAP's.
    views = []
    for token in [t.strip() for t in recipe.split("+")]:
        if token in dict(VIEW_MAP):
            views.append(token)
    if len(views) != n_views:
        print(f"Recipe '{recipe}' names {len(views)} known view(s) but the head "
              f"needs {n_views} x {expect_dim}-d blocks. Refusing to export a "
              f"manifest the browser cannot reproduce.")
        return 2

    # ── head weights, one set per fold (the browser averages them, exactly as
    #    PotatoClassifier does server-side) ──
    arrays, fold_meta = [], []
    for i, ck in enumerate(cks):
        sd = ck["head"]
        if hidden:
            arrays += [(f"f{i}_w0", sd["net.0.weight"].numpy()),
                       (f"f{i}_b0", sd["net.0.bias"].numpy()),
                       (f"f{i}_w1", sd["net.3.weight"].numpy()),
                       (f"f{i}_b1", sd["net.3.bias"].numpy())]
        else:
            arrays += [(f"f{i}_w", sd["net.1.weight"].numpy()),
                       (f"f{i}_b", sd["net.1.bias"].numpy())]
        fold_meta.append({"fold": i, "temperature": float(ck.get("temperature", 1.0))})

    head_layout = write_f32(out / "head.bin", arrays)
    head_kb = (out / "head.bin").stat().st_size / 1024

    # ── OOD: means + Cholesky factor of the precision ──
    ood_meta = None
    ood_kb = 0.0
    ood_path = art / "ood.npz"
    if ood_path.exists():
        z = np.load(ood_path)
        means, prec = z["means"].astype(np.float64), z["precision"].astype(np.float64)
        if args.diag_only:
            payload = [("means", means), ("prec_diag", np.diag(prec))]
            mode = "diagonal"
        else:
            # P = L L^T, so d^2 = ||L^T (x-mu)||^2. Ship the packed upper
            # triangle of L^T: half the bytes, identical arithmetic.
            prec = (prec + prec.T) / 2.0
            w, V = np.linalg.eigh(prec)
            w = np.clip(w, 1e-10, None)          # guarantee PD before Cholesky
            prec = (V * w) @ V.T
            Lt = np.linalg.cholesky(prec).T
            iu = np.triu_indices(feat_dim)
            payload = [("means", means), ("Lt_upper", Lt[iu])]
            mode = "cholesky_upper"
        ood_layout = write_f32(out / "ood.bin", payload)
        ood_kb = (out / "ood.bin").stat().st_size / 1024
        ood_meta = {"mode": mode, "layout": ood_layout}

        # prove the compression is lossless before shipping it
        if mode == "cholesky_upper":
            rng = np.random.default_rng(0)
            x = rng.normal(size=(64, feat_dim))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            d_ref = np.min([np.einsum("ij,jk,ik->i", x - m, z["precision"], x - m)
                            for m in means], axis=0)
            Lt_full = np.zeros((feat_dim, feat_dim))
            Lt_full[np.triu_indices(feat_dim)] = payload[1][1]
            # P = L L^T  =>  (x-m)^T P (x-m) = || L^T (x-m) ||^2
            # Batched: z = (x-m) @ Lt.T, then d = sum(z^2) along the feature axis.
            d_new = np.min([(((x - m) @ Lt_full.T) ** 2).sum(1) for m in means], axis=0)
            err = float(np.max(np.abs(d_ref - d_new) / np.maximum(np.abs(d_ref), 1e-9)))
            print(f"  Cholesky round-trip max relative error: {err:.2e}"
                  f"  {'OK' if err < 1e-4 else '<-- CHECK THIS'}")

    cal = {}
    cal_path = art / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))

    metrics = {}
    mpath = art / "metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text(encoding="utf-8"))
        metrics = {k: m[k] for k in ("macro_f1_mean", "macro_f1_std",
                                     "accuracy_mean", "accuracy_std", "ece_mean")
                   if k in m}

    manifest = {
        "version": 1,
        "classes": classes,
        "feat_dim": feat_dim,
        "hidden": hidden,
        "n_folds": len(cks),
        "folds": fold_meta,
        "backbone": {
            "trained_with": backbone,
            "onnx_model_id": onnx_id,
            "dtype": args.dtype,
            "approx_download_mb": ONNX_SIZES_MB.get(onnx_id, {}).get(args.dtype),
            "output": "pooler_output, then L2-normalised",
            # The browser MUST build exactly these blocks, in this order, and
            # L2-normalise each one separately before concatenating.
            "views": views,
            "n_views": n_views,
            "view_dim": expect_dim,
            "feature_recipe": recipe,
            # 2x2 tiling: resize the shortest side to 2x, centre-crop 2x, cut into
            # four non-overlapping crops of the normal size, run each, then average.
            "tiling": {"grid": 2, "enabled": any(v.startswith("tile2x2") for v in views)},
        },
        # The browser MUST reproduce this exactly or the head sees shifted features.
        "preprocessing": {
            "resize_shortest_edge": 256,
            "resample": "bicubic",
            "center_crop": 224,
            "rescale": 1 / 255,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "note": "matches Xenova/dinov2-* preprocessor_config.json and the "
                    "pinned transform in potato_infer.py / train_potato.py",
        },
        "head": {"file": "head.bin", "layout": head_layout},
        "ood": ood_meta and {"file": "ood.bin", **ood_meta},
        "thresholds": {
            "ood_mahalanobis_p99": cal.get("ood_mahalanobis_threshold_p99"),
            "abstain_below_confidence": cal.get("abstain_below_confidence"),
        },
        "server_metrics": metrics,
        "disclaimer": "Guidance only. Confirm any treatment with a licensed agronomist.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dl = manifest["backbone"]["approx_download_mb"]
    print(f"\n  wrote {out}/")
    print(f"    manifest.json    {(out/'manifest.json').stat().st_size/1024:8.1f} KB")
    print(f"    head.bin         {head_kb:8.1f} KB   ({len(cks)} folds"
          f"{', MLP-' + str(hidden) if hidden else ', linear'})")
    if ood_meta:
        print(f"    ood.bin          {ood_kb:8.1f} KB   ({ood_meta['mode']})")
    print(f"    backbone         {onnx_id} [{args.dtype}]"
          + (f" ~{dl:.0f} MB from the Hub" if dl else ""))
    total = (head_kb + ood_kb) / 1024 + (dl or 0)
    print(f"    FIRST LOAD       ~{total:.1f} MB total, cacheable; ~0 MB thereafter")
    if metrics:
        print(f"\n  server-measured: macro-F1 {metrics.get('macro_f1_mean', 0):.4f}"
              f" +/- {metrics.get('macro_f1_std', 0):.4f},"
              f" accuracy {metrics.get('accuracy_mean', 0):.4f}")
        print("  The browser must reproduce these. Run the parity check in "
              "public/js/potato-browser.js against a few known images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
