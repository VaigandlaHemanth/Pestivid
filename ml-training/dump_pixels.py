"""Dump PIL's exact intermediate pixels so the JS resize can be checked
byte-for-byte, not just "looks close".

For each parity image we write:
  <name>.src.bin     tightly packed RGB of the ORIGINAL decoded image
  <name>.r256.bin    tightly packed RGB after PIL Resize(shortest=256, BICUBIC)
  <name>.tensor.bin  float32 CHW 3x224x224 after CenterCrop + /255 + normalise
plus a JSON index with the shapes.

If JS reproduces .r256.bin exactly, the antialiasing replication is correct and
any residual feature difference must come from JPEG decoding or the ONNX graph.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

PUB = Path(r"C:/Users/ASUS/Desktop/pestivid orginal/p_pro/pestvid_complete_project/public/parity")
OUT = PUB / "pixels"
OUT.mkdir(parents=True, exist_ok=True)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
resize = T.Resize(256, interpolation=T.InterpolationMode.BICUBIC)
tail = T.Compose([T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])

index = []
for f in sorted(PUB.glob("parity_*.jpg")):
    with Image.open(f) as im:
        rgb = im.convert("RGB")
        src = np.asarray(rgb, dtype=np.uint8)          # H, W, 3
        r256 = resize(rgb)
        r256_arr = np.asarray(r256, dtype=np.uint8)
        tensor = tail(r256).numpy().astype(np.float32)  # 3, 224, 224

    stem = f.stem
    (OUT / f"{stem}.src.bin").write_bytes(src.tobytes())
    (OUT / f"{stem}.r256.bin").write_bytes(r256_arr.tobytes())
    (OUT / f"{stem}.tensor.bin").write_bytes(tensor.tobytes())
    index.append({
        "name": stem,
        "file": f.name,
        "src": {"h": int(src.shape[0]), "w": int(src.shape[1])},
        "r256": {"h": int(r256_arr.shape[0]), "w": int(r256_arr.shape[1])},
        "tensor_dims": [1, 3, 224, 224],
    })
    print(f"  {stem:22s} src {src.shape[1]}x{src.shape[0]}"
          f"  -> r256 {r256_arr.shape[1]}x{r256_arr.shape[0]}"
          f"  tensor mean {tensor.mean():+.5f}")

(OUT / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
total = sum(p.stat().st_size for p in OUT.glob("*.bin")) / 1024 / 1024
print(f"\nwrote {len(index)} image sets to {OUT}  ({total:.1f} MB)")
