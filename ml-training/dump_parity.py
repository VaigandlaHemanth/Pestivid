"""Dump Python-side DINOv2 features for a handful of real images so the browser
module can be checked against them. Uses the SAME pinned transform as training."""
import importlib.util, json, shutil, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

HERE = Path(r"C:/Users/ASUS/Desktop/pestivid orginal/p_pro/Pestivid")
spec = importlib.util.spec_from_file_location("tp", HERE / "train_potato.py")
tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)

root = Path(sys.argv[1])
out_dir = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)

# one image per class, deterministic pick
classes = sorted(d.name for d in root.iterdir() if d.is_dir())
picks = []
for c in classes:
    files = sorted(p for p in (root / c).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if files:
        picks.append((c, files[len(files) // 2]))

tf = tp.build_transforms("dinov2-base", train=False)
bb = tp.FrozenBackbone("dinov2-base", "cpu")

records = []
for cls, path in picks:
    dest = out_dir / f"parity_{cls}{path.suffix.lower()}"
    shutil.copy(path, dest)
    with Image.open(path) as im:
        x = tf(im.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        f = bb(x)                                  # already L2-normalised in FrozenBackbone
    v = f.squeeze(0).numpy().astype(np.float64)
    records.append({"file": dest.name, "class": cls,
                    "norm": float(np.linalg.norm(v)),
                    "features": [round(float(t), 7) for t in v]})
    print(f"  {cls:14s} {dest.name:28s} norm={np.linalg.norm(v):.6f}  f[0:3]={v[:3].round(5)}")

(out_dir / "parity.json").write_text(json.dumps({
    "backbone": "dinov2-base",
    "onnx_model_id": "Xenova/dinov2-base",
    "transform": "Resize(256,BICUBIC) -> CenterCrop(224) -> ToTensor -> ImageNet norm -> L2",
    "feat_dim": int(len(records[0]["features"])),
    "records": records,
}, indent=1), encoding="utf-8")
print(f"\nwrote {out_dir/'parity.json'}  ({len(records)} images, dim={len(records[0]['features'])})")
