"""Dump the FULL multi-view feature vector for a few images, per view.

Why a separate script from dump_parity.py: that one dumps the single 768-d global
CLS vector, which was the whole recipe when the head was single-view. The head is
now fitted to a concatenation (global CLS + 2x2 tile mean CLS + 2x2 tile mean
patch), and each block has to be checked SEPARATELY -- a mismatch in one view is
invisible in the concatenated cosine but changes the verdict.

The reference comes from potato_infer.PotatoClassifier._features, i.e. the actual
server path, so the browser is compared against what production really computes
rather than against a reimplementation of it.

    python dump_parity_tiled.py <dataset-root> <out-dir>
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ML = REPO / "ml-service"          # potato_infer.py, artifacts/
WEB = REPO / "frontend"            # parity fixtures the browser also serves


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    pi = load(ML / "potato_infer.py", "pi")
    clf = pi.PotatoClassifier(ML / "artifacts")
    print(f"  recipe: {' + '.join(clf.views)}  ({clf.feat_dim}-d)")

    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    records = []
    for c in classes:
        files = sorted(p for p in (root / c).iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            continue
        src = files[len(files) // 2]
        dest = out / f"parity_{c}{src.suffix.lower()}"
        shutil.copy(src, dest)

        with Image.open(src) as im:
            vec = clf._features(im).cpu().numpy()[0].astype(np.float64)

        # Split into per-view blocks so a single bad view is identifiable.
        blocks = {}
        for i, v in enumerate(clf.views):
            b = vec[i * 768:(i + 1) * 768]
            blocks[v] = [round(float(t), 7) for t in b]

        # The OOD Mahalanobis score and the predicted class.
        #
        # Per-view cosine distance alone is NOT a sufficient parity check, and
        # this was demonstrated rather than assumed: reintroducing the
        # normalisation-order bug (L2 each tile, then average, instead of average
        # then one L2) moved the worst per-view distance only to 6.4e-4 -- inside
        # any tolerance loose enough to accommodate fp16 -- while in production
        # that same bug pushed the Mahalanobis score to 3285 against a p99
        # threshold of 2702 and made the gate reject genuine potato leaves.
        #
        # So the reference carries the quantity that actually broke. A small
        # cosine shift that moves the vector out of the fitted distribution is
        # caught here even when the cosine check shrugs.
        # predict() takes a PATH, and Verdict is a dict subclass.
        pred = {}
        try:
            pred = clf.predict(dest)
        except Exception as e:  # noqa: BLE001
            print(f"    (predict failed for {c}: {e})")

        records.append({
            "file": dest.name,
            "class": c,
            "norm": float(np.linalg.norm(vec)),
            "ood_score": (float(pred["ood_score"])
                          if pred.get("ood_score") is not None else None),
            "status": pred.get("status"),
            "predicted": pred.get("disease"),
            "confidence": (float(pred["confidence"])
                           if pred.get("confidence") is not None else None),
            "views": blocks,
        })
        print(f"  {c:14s} {dest.name:28s} |v|={np.linalg.norm(vec):.4f}")

    (out / "parity_tiled.json").write_text(json.dumps({
        "recipe": clf.views,
        "feat_dim": int(clf.feat_dim),
        "view_dim": 768,
        "tiling_grid": clf.tiling_grid,
        "transform": ("global: Resize(256,BICUBIC)+CenterCrop(224); "
                      "tiles: Resize(512,BICUBIC)+CenterCrop(448) cut into 4x224; "
                      "each view L2-normalised AFTER averaging across tiles"),
        "records": records,
    }, indent=1), encoding="utf-8")
    print(f"\n  wrote {out / 'parity_tiled.json'}  ({len(records)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
