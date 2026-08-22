"""Add the server's OOD score and prediction to the existing parity reference.

Per-view cosine distance is not a sufficient parity check, and that was measured
rather than assumed. Reintroducing the normalisation-order bug -- L2-normalise
each tile and then average, instead of averaging the raw vectors and normalising
once -- moved the worst per-view cosine distance only to 6.4e-4. That is inside
any tolerance loose enough to accommodate fp16 inference against an fp32
reference, so the check passed.

The same bug in production pushed the Mahalanobis OOD score to 3285 against a p99
threshold of 2712 and made the gate reject genuine potato leaves as "not a potato
leaf". The decision changed; the cosine barely moved.

So the reference needs the quantity that actually breaks. This reads the images
already in public/parity/, runs the real server predict() over them, and writes
ood_score, status, disease and confidence into parity_tiled.json. It needs no
dataset -- only the images that are already there.

    python add_ood_to_parity.py [parity-dir]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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
    pdir = Path(sys.argv[1]) if len(sys.argv) > 1 else (WEB / "parity")
    manifest = pdir / "parity_tiled.json"
    if not manifest.exists():
        print(f"  no {manifest}")
        return 1

    data = json.loads(manifest.read_text(encoding="utf-8"))

    pi = load(ML / "potato_infer.py", "pi")
    clf = pi.PotatoClassifier(ML / "artifacts")
    print(f"  ood threshold (p99): {clf.ood_threshold:.2f}")

    updated = 0
    for rec in data["records"]:
        img = pdir / rec["file"]
        if not img.exists():
            print(f"  MISSING {rec['file']}")
            continue
        try:
            v = clf.predict(img)
        except Exception as e:  # noqa: BLE001
            print(f"  predict failed for {rec['file']}: {e}")
            continue

        rec["ood_score"] = (float(v["ood_score"])
                            if v.get("ood_score") is not None else None)
        rec["status"] = v.get("status")
        rec["predicted"] = v.get("disease")
        rec["confidence"] = (float(v["confidence"])
                             if v.get("confidence") is not None else None)
        updated += 1
        print(f"  {rec['file']:30s} ood={rec['ood_score']:.1f}  "
              f"{rec['status']:<10s} {str(rec['predicted']):<14s} "
              f"conf={rec['confidence'] if rec['confidence'] is not None else 0:.3f}")

    data["ood_threshold_p99"] = float(clf.ood_threshold)
    data["note_ood"] = (
        "ood_score is the Mahalanobis distance the SERVER computes. Cosine "
        "distance per view is not enough on its own: the normalisation-order bug "
        "moved the worst per-view cosine only to 6.4e-4 while pushing this score "
        "past the threshold and flipping the verdict to not_a_leaf. Compare this "
        "number, and the status, not only the cosines."
    )
    manifest.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"\n  updated {updated}/{len(data['records'])} records in {manifest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
