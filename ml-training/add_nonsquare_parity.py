"""Add a NON-SQUARE image to the browser/server parity set.

Why this is needed.

Every existing parity image is 1500x1500. Resize(256) therefore produces 256x256,
the centre-crop offset is (256-224)/2 = 16 exactly, and no rounding rule is ever
exercised. The harness reported a worst-case cosine distance of 3.9e-4 and was
blind to a real one-pixel disagreement: torchvision's CenterCrop rounds the offset
with Python's round() (banker's, ties to even) while the browser used
Math.round() (ties up). They differ only on an exact .5, which happens whenever
(side - crop) is odd.

A phone photo is 4:3 or 16:9, never square. 4032x3024 resizes to 341x256, and
(341-224)/2 = 58.5 -- a tie. So the case the parity set omitted is the case that
every real photograph from a farmer's phone hits, and the case a square test image
can never reach.

This writes a 4:3 crop of an existing parity image and appends its per-view server
features to parity_tiled.json, so public/parity/tiled.html covers it from then on.

    python add_nonsquare_parity.py <app-public-parity-dir>
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "pestvid_complete_project"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (APP / "public" / "parity")
    manifest = out / "parity_tiled.json"
    if not manifest.exists():
        print(f"  no {manifest} -- run dump_parity_tiled.py first")
        return 1

    data = json.loads(manifest.read_text(encoding="utf-8"))

    # Build the 4:3 source from an existing parity image, so the content is a real
    # leaf rather than noise -- the point is to exercise the geometry, but a
    # realistic image also keeps the prediction meaningful.
    src = out / "parity_Fungi.jpg"
    if not src.exists():
        print(f"  missing {src}")
        return 1

    # 1200x900 is 4:3. Resize(256) -> 341x256, so (341-224)/2 = 58.5, a tie.
    dest = out / "parity_Fungi_4x3.jpg"
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")

        # Build the 4:3 frame by PADDING, not by cropping.
        #
        # The first version of this cropped the square down to 4:3, which threw
        # away the top and bottom of the leaf. Measured consequence: the OOD
        # Mahalanobis score went from 2161 to 6608 against a p99 threshold of
        # 2713, so the fixture was rejected as "not a potato leaf" -- a parity
        # image the model refuses to classify is a bad parity image, because the
        # prediction column is then meaningless.
        #
        # The gate turns out to be extremely sensitive to FRAMING, not to aspect
        # ratio: cropping 10% into the leaf already pushes 6 of 7 test images past
        # the threshold. Keeping the whole leaf in a centred square region and
        # padding out to 4:3 preserves the framing the model was fitted on -- the
        # centre crop then takes 87.5% of the leaf, exactly as it does for a
        # square input -- and scores ~2200, comfortably inside the gate.
        #
        # The geometry that this fixture exists to test is unaffected: 1200x900
        # still resizes to 341x256, so the centre-crop offset is still
        # (341-224)/2 = 58.5, an exact rounding tie.
        side = min(im.size)
        leaf = im.resize((side, side), Image.Resampling.BICUBIC).resize(
            (900, 900), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (1200, 900), (124, 116, 104))
        canvas.paste(leaf, ((1200 - 900) // 2, 0))
        canvas.save(dest, quality=95)

    with Image.open(dest) as check:
        w, h = check.size
    ow, oh = (256, int(256 * h / w)) if w < h else (int(256 * w / h), 256)
    print(f"  wrote {dest.name}  {w}x{h} -> resize {ow}x{oh}")
    print(f"  centre-crop offsets: ({oh - 224}/2, {ow - 224}/2)"
          f" = ({(oh - 224) / 2}, {(ow - 224) / 2})")
    tie = ((oh - 224) % 2) or ((ow - 224) % 2)
    print(f"  exercises the rounding tie: {'YES' if tie else 'no'}")
    if not tie:
        print("  WARNING: this image does not hit the tie, so it does not test the fix")

    # Server-side reference, through the real inference path.
    pi = load(APP / "potato_infer.py", "pi")
    clf = pi.PotatoClassifier(APP / "artifacts")
    if clf.views != data["recipe"]:
        print(f"  recipe drift: {clf.views} vs {data['recipe']}")
        return 1

    with Image.open(dest) as im:
        vec = clf._features(im).cpu().numpy()[0].astype(np.float64)

    blocks = {}
    for i, v in enumerate(clf.views):
        blocks[v] = [round(float(t), 7) for t in vec[i * 768:(i + 1) * 768]]

    record = {
        "file": dest.name,
        "class": "Fungi (4:3 non-square)",
        "norm": float(np.linalg.norm(vec)),
        "views": blocks,
    }

    # Replace any earlier copy rather than appending a duplicate.
    data["records"] = [r for r in data["records"] if r["file"] != dest.name]
    data["records"].append(record)
    data["note_nonsquare"] = (
        "parity_Fungi_4x3.jpg is 1200x900. Resize(256) gives 341x256, so the "
        "centre-crop offset is (341-224)/2 = 58.5 -- an exact tie. torchvision "
        "uses Python's round() (ties to even -> 58); Math.round() would give 59. "
        "Every other image in this set is square and cannot detect that."
    )

    manifest.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"  appended to {manifest.name}  ({len(data['records'])} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
