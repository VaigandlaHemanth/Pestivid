"""Dump the exact torchvision preprocessing result for a few images.

This isolates the part of the pipeline that the browser reimplements -- decode ->
Resize -> CenterCrop -> ToTensor -> Normalize -- from the ONNX backbone, so it can
be checked in Node without WASM and without a browser.

It exists because the tiled parity page compares FEATURES, which requires 5
backbone passes per image and takes ~10s each. A preprocessing disagreement is
cheaper and more precise to catch here: if the input tensors match bit-for-bit,
any remaining feature difference is the backbone, not the transform.

Writes, per image:
  <name>.rgb.bin    the decoded RGB bytes (H*W*3, uint8) -- the browser's input,
                    so JPEG decoder differences are excluded from the comparison
  <name>.ref.bin    the preprocessed CHW float32 tensor torchvision produces
  index.json        dimensions and the crop offsets, for the Node side to assert

    python dump_preprocess_ref.py <parity-dir> <out-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T


def main() -> int:
    src_dir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    # Exactly the global-view transform from potato_infer.py.
    tf = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    records = []
    files = sorted(p for p in src_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for f in files:
        with Image.open(f) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size

            # The browser receives decoded pixels, not a JPEG. Hand Node the same
            # bytes so a libjpeg-vs-browser decoder difference cannot be mistaken
            # for a transform difference.
            rgb = np.asarray(im, dtype=np.uint8)
            assert rgb.shape == (h, w, 3), rgb.shape
            (out / f"{f.stem}.rgb.bin").write_bytes(rgb.tobytes())

            ten = tf(im).numpy().astype(np.float32)   # (3, 224, 224)
            (out / f"{f.stem}.ref.bin").write_bytes(ten.tobytes())

        # Reproduce the offsets torchvision will have used, for the assertion.
        if w < h:
            ow, oh = 256, int(256 * h / w)
        else:
            oh, ow = 256, int(256 * w / h)
        if (w <= h and w == 256) or (h <= w and h == 256):
            ow, oh = w, h

        records.append({
            "stem": f.stem,
            "file": f.name,
            "inW": w, "inH": h,
            "resizedW": ow, "resizedH": oh,
            "cropTop": int(round((oh - 224) / 2.0)),
            "cropLeft": int(round((ow - 224) / 2.0)),
            # True when (side - crop) is odd, i.e. the offset lands on an exact
            # .5 and the rounding RULE decides the answer.
            "tie": bool(((oh - 224) % 2) or ((ow - 224) % 2)),
            "square": w == h,
        })
        r = records[-1]
        print(f"  {f.name:28s} {w}x{h} -> {ow}x{oh}  "
              f"offset=({r['cropTop']},{r['cropLeft']})  tie={r['tie']}")

    (out / "index.json").write_text(json.dumps({
        "transform": "Resize(256,BICUBIC) + CenterCrop(224) + ToTensor + Normalize",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "records": records,
    }, indent=1), encoding="utf-8")

    ties = sum(1 for r in records if r["tie"])
    print(f"\n  {len(records)} images, {ties} exercising the rounding tie")
    if ties == 0:
        print("  WARNING: no image hits the tie, so this set cannot detect a")
        print("  round-half-up vs round-half-to-even mismatch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
