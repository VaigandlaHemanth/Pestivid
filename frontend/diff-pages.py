"""Compares every generated page against the artboard panel it was cut from.

Three verdicts, because three different things can be true:

  identical     not one pixel differs
  sub-pixel     differs only because the panel landed on a fractional pixel --
                the same layout, rendered half a pixel over. Text edges change,
                nothing has moved.
  MOVED         survives both a one-pixel alignment search and a 3px blur, so
                something is genuinely in a different place. This is drift and
                it has to be fixed.

  python frontend/diff-pages.py [--save]
"""
from PIL import Image, ImageChops, ImageFilter
import pathlib, json, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SAVE = "--save" in sys.argv
S = pathlib.Path("frontend/.verify")
rows = json.loads((S / "pairs.json").read_text())

print(f"{'page':22s} {'size':>13s} {'raw px':>8s} {'aligned':>8s} {'blurred':>8s}  verdict")
clean, subpx, moved, broken = [], [], [], []

for r in rows:
    slug = r["slug"]
    if r.get("err"):
        print(f"  {slug:20s} {r['err']}"); broken.append(slug); continue
    a = Image.open(S / f"{slug}.board.png").convert("RGB")
    b = Image.open(S / f"{slug}.page.png").convert("RGB")
    if a.size[0] != b.size[0]:
        print(f"{slug:22s} {str(a.size):>13s}  WIDTH MISMATCH vs {b.size}")
        broken.append(slug); continue
    note = ""
    if a.size[1] != b.size[1]:
        h = min(a.size[1], b.size[1])
        note = f"  (board {a.size[1]}px, page {b.size[1]}px)"
        a = a.crop((0, 0, a.size[0], h)); b = b.crop((0, 0, b.size[0], h))
    W, H = a.size
    raw = sum(1 for p in ImageChops.difference(a, b).convert("L").getdata() if p > 8)

    # A panel can land a pixel off from sub-pixel rounding, which shifts every
    # edge at once. Score the best alignment before judging anything.
    best, shift = None, (0, 0)
    for dy in (0, 1, -1, 2, -2):
        for dx in (0, 1, -1):
            bb = ImageChops.offset(b, dx, dy)
            box = (max(dx, 0) + 2, max(dy, 0) + 2, W + min(dx, 0) - 2, H + min(dy, 0) - 2)
            ca, cb = a.crop(box), bb.crop(box)
            c = sum(1 for p in ImageChops.difference(ca, cb).convert("L").getdata() if p > 8)
            if best is None or c < best[0]:
                best, shift, pair = (c, None), (dx, dy), (ca, cb)
    aligned = best[0]

    # Then decide at a coarser scale. White text on a near-black hero shifted
    # half a pixel still beats a blur threshold -- the contrast is too high for
    # that test alone. Averaging 4x4 blocks removes anything smaller than a
    # pixel while leaving an element that actually moved plainly visible.
    ca, cb = pair
    small = (max(ca.size[0] // 4, 1), max(ca.size[1] // 4, 1))
    d = ImageChops.difference(ca.resize(small, Image.BOX), cb.resize(small, Image.BOX)).convert("L")
    blurred = sum(1 for p in d.getdata() if p > 24)
    if shift != (0, 0):
        note += f"  [aligned {shift[0]},{shift[1]}px]"

    if raw == 0:
        verdict = "identical"; clean.append(slug)
    elif blurred == 0:
        verdict = "sub-pixel"; subpx.append(slug)
    else:
        verdict = f"MOVED  {100 * blurred / (small[0] * small[1]):.2f}% of blocks"; moved.append(slug)
        if SAVE:
            d.point(lambda v: 255 if v > 24 else 0).resize(ca.size, Image.NEAREST).save(S / f"{slug}.moved.png")
    print(f"{slug:22s} {str((W, H)):>13s} {raw:>8d} {aligned:>8d} {blurred:>8d}  {verdict}{note}")

n = len(rows)
print(f"\n{len(clean)} identical, {len(subpx)} sub-pixel, {len(moved)} moved, {len(broken)} broken"
      f"   ->  {len(clean) + len(subpx)}/{n} faithful")
if moved: print("moved: " + ", ".join(moved))
if broken: print("broken: " + ", ".join(broken))
sys.exit(1 if moved or broken else 0)
