"""Extract several feature VIEWS per image in as few forward passes as possible.

WHY
    Concatenating dinov2-base and dinov2-large CLS features measured 0.8755
    accuracy versus 0.8622 for base alone -- even though large ALONE was no better
    than base (0.8615). The gain came from representation diversity, not from
    capacity. That makes cheap extra views the most promising remaining lever.

    The key economy: a ViT forward pass already computes last_hidden_state, so the
    CLS token and the mean of the patch tokens are two different views of the SAME
    computation. train_potato.py threw the patch tokens away (it kept only
    pooler_output), so the mean-patch view has never been measured here and costs
    nothing beyond storage. A horizontal flip is a genuinely second pass, so
    --flip doubles the time.

VIEWS WRITTEN
    cls          CLS token, L2-normalised          (what training used)
    mean         mean over patch tokens, L2-norm   (free, never yet measured)
    cls_flip     CLS of the horizontally flipped image      (--flip)
    mean_flip    mean-patch of the flipped image            (--flip)
    cls_tta      L2-norm of (cls + cls_flip)/2               (--flip)
    mean_tta     L2-norm of (mean + mean_flip)/2             (--flip)

    Every view is L2-normalised individually, matching how the head was fitted.

ROW ALIGNMENT IS LOAD-BEARING
    y and groups are produced by importing train_potato.collect_images and
    group_near_duplicates, so the row order, labels and near-duplicate group ids
    are IDENTICAL to the existing caches. probe_lab.py asserts this when it
    concatenates across files; without it, a silently permuted cache would
    produce a meaningless score rather than an error.

    python extract_views.py --data-root <dataset> --backbone dinov2-base \
        --out art_views_base.npz --flip
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tp", HERE / "train_potato.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)

HF_ID = {
    "dinov2-base": "facebook/dinov2-base",
    "dinov2": "facebook/dinov2-large",
    "dinov2-large": "facebook/dinov2-large",
    "dinov2-small": "facebook/dinov2-small",
}

# timm-hosted backbones, loaded through a different path than transformers.
#
# WHY THESE TWO
#   dinov2-base was pretrained on LVD-142M web imagery. The failing distinctions
#   here are fungal lesion vs insect chewing vs late-blight water-soaking, which
#   is plant pathology, not web-image semantics. These two isolate the two
#   plausible fixes:
#     reg4        register tokens (Darcet et al., arXiv 2309.16588) remove the
#                 high-norm artifact tokens that pollute ViT patch features --
#                 tests whether FEATURE QUALITY was the problem
#     reg4-plant  the same architecture fine-tuned on 1.4M Pl@ntNet images over
#                 7806 species for PlantCLEF 2024 -- tests whether PLANT DOMAIN
#                 knowledge was the problem
#   Caveat: PlantCLEF fine-tuned for SPECIES identification, which rewards leaf
#   shape and venation rather than lesion texture. Plant-domain is not
#   automatically disease-domain, so this is a genuine open question.
#
# Both are ViT-B/14, 768-d, ~86M params -- the same size as dinov2-base, so a
# browser download stays around 175 MB in fp16.
TIMM_ID = {
    "reg4": ("vit_base_patch14_reg4_dinov2.lvd142m", None),
    "reg4-plant": ("vit_base_patch14_reg4_dinov2",
                   ("vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights",
                    "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all.safetensors")),
}


def load_timm(name, size):
    """Build a timm ViT at an explicit img_size, position embeddings interpolated.

    These checkpoints are native 518px. Forcing 224 keeps them directly
    comparable to the existing dinov2-base result at the same cost; running them
    at 518 would confound the backbone change with a 5.3x resolution change.
    """
    import timm
    arch, ckpt = TIMM_ID[name]
    if ckpt is None:
        return timm.create_model(arch, pretrained=True, num_classes=0,
                                 img_size=size).eval()

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import timm.layers

    m = timm.create_model(arch, pretrained=False, num_classes=0, img_size=size)
    sd = load_file(hf_hub_download(ckpt[0], ckpt[1]))
    sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    if "pos_embed" in sd and sd["pos_embed"].shape != m.pos_embed.shape:
        # A 518px checkpoint into a 224px model: resample the grid rather than
        # dropping it, or every patch receives the wrong positional signal.
        sd["pos_embed"] = timm.layers.resample_abs_pos_embed(
            sd["pos_embed"], new_size=list(m.patch_embed.grid_size),
            num_prefix_tokens=0)
    miss, unexp = m.load_state_dict(sd, strict=False)
    miss = [k for k in miss if not k.startswith("head.")]
    if miss:
        raise SystemExit(f"{name}: checkpoint is missing {miss[:6]} -- refusing to "
                         f"extract with a partially-initialised backbone.")
    if unexp:
        print(f"    note: ignored {len(unexp)} unexpected keys e.g. {unexp[:3]}")
    return m.eval()


class Images(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(ImageOps.exif_transpose(im).convert("RGB"))


class TiledImages(Dataset):
    """Yields a (n*n, 3, crop, crop) stack of non-overlapping tiles per image.

    WHY TILING RATHER THAN A BIGGER CROP
        A 448px centre crop and a 2x2 grid of 224px tiles taken from that same
        448px region are pixel-identical in field of view, in ground sampling
        distance, and in total pixels (200,704). They differ only in how the
        transformer sees them, and tiling is the cheaper and safer of the two:

            448 single pass   1025 tokens, attention is quadratic -> ~4.6x a 224 pass
            2x2 tiles at 224  4 x 257 tokens                       -> ~4.0x a 224 pass

        For the browser the difference is larger than 4.6 vs 4.0 suggests: tiles
        keep the ONNX input shape at exactly 224x224, the shape already being run,
        so a phone makes four cheap identical calls with flat peak memory instead
        of one call with a 1025x1025 attention matrix. No position-embedding
        interpolation is needed either.

        The motivation is specific, not general: 53.5% of this model's errors are
        inside {Fungi, Pest, Phytopthora}, which are texture distinctions --
        lesion margins versus chewing damage. At 224 from a 1500px source we feed
        the model 14.9% linear resolution and throw away ~98% of the pixels.
    """

    def __init__(self, paths, shortest, crop, n=2, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        import torchvision.transforms as T
        self.paths, self.n, self.crop = paths, n, crop
        # Resize/crop n times larger, then cut into n*n tiles of exactly `crop`.
        self.pre = T.Compose([
            T.Resize(shortest * n, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(crop * n),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.pre(ImageOps.exif_transpose(im).convert("RGB"))          # (3, crop*n, crop*n)
        c, n = self.crop, self.n
        return torch.stack([x[:, r * c:(r + 1) * c, k * c:(k + 1) * c]
                            for r in range(n) for k in range(n)])


def tokens(model, x, is_timm, n_prefix):
    """Token sequence as (B, T, D), plus how many leading tokens are not patches.

    timm reg4 models order the sequence [CLS, reg0..reg3, patches...], so the
    patch-mean must skip 5 tokens, not 1. Averaging the register tokens into the
    patch mean would fold in exactly the artifact-absorbing tokens that the
    register design exists to keep separate.
    """
    if is_timm:
        return model.forward_features(x), n_prefix
    return model(pixel_values=x).last_hidden_state, 1


@torch.no_grad()
def extract(model, loader, flip: bool, device, log_every=10, is_timm=False,
            n_prefix=1, ckpt: "Path | None" = None, ckpt_every=320):
    """Returns (cls, mean, cls_flip, mean_flip); the flip pair is None if off.

    CHECKPOINTING IS NOT OPTIONAL HERE. A 3076-image CPU extraction runs for
    25-90 minutes, and one of these was killed at image 2416 of 3076 by a shell
    teardown, losing 45 minutes of compute with nothing to show. Partial results
    are flushed to <out>.partial.npz every ckpt_every images and reloaded on the
    next run, so a kill costs at most one checkpoint interval.

    The resume is keyed on the row COUNT only, which is safe purely because the
    loader is constructed with shuffle=False over a fixed path list -- row i is
    always the same image. If that ever changes, this resume becomes silently
    wrong, so do not enable shuffling here.
    """
    cls, mean, clsf, meanf = [], [], [], []
    done = 0
    if ckpt is not None and ckpt.exists():
        # np.load on an .npz is LAZY and holds the file open. On Windows,
        # os.replace over a path with a live handle raises WinError 5, so the
        # first flush after a resume used to crash. Copy what we need inside the
        # context manager and let the handle close before any flush happens.
        with np.load(ckpt) as z:
            same = bool(z["flip"]) == flip
            if same:
                cls = [torch.from_numpy(z["cls"].copy())]
                mean = [torch.from_numpy(z["mean"].copy())]
                if flip:
                    clsf = [torch.from_numpy(z["cls_flip"].copy())]
                    meanf = [torch.from_numpy(z["mean_flip"].copy())]
                done = int(z["n"])
            else:
                had = bool(z["flip"])
        if same:
            print(f"    resuming from {ckpt.name}: {done} images already done",
                  flush=True)
        else:
            print(f"    ignoring {ckpt.name}: written with flip={had}, "
                  f"this run wants flip={flip}", flush=True)

    def flush(n):
        if ckpt is None:
            return
        cat = lambda L: torch.cat(L).numpy().astype(np.float32)
        d = {"cls": cat(cls), "mean": cat(mean), "n": n, "flip": flip}
        if flip:
            d["cls_flip"] = cat(clsf)
            d["mean_flip"] = cat(meanf)
        tmp = ckpt.with_suffix(".tmp.npz")
        try:
            np.savez(tmp, **d)
            tmp.replace(ckpt)      # atomic: never leave a half-written checkpoint
        except OSError as e:
            # A checkpoint is an optimisation. Losing one is not worth losing the
            # 20-90 minutes of extraction that is already in memory.
            print(f"    checkpoint write failed ({e}); continuing without it",
                  flush=True)

    t0, seen, since = time.time(), 0, 0
    total = len(loader.dataset)
    for bi, x in enumerate(loader):
        b = len(x)
        # Skip batches already covered by the checkpoint. The forward pass is what
        # costs money; loading and discarding the tensors is comparatively free.
        if seen + b <= done:
            seen += b
            continue
        x = x.to(device)
        out, npre = tokens(model, x, is_timm, n_prefix)     # (B, npre+P, D)
        cls.append(out[:, 0].float().cpu())
        mean.append(out[:, npre:].mean(1).float().cpu())
        if flip:
            # Horizontal flip only. A vertical flip is not a plausible photo of a
            # leaf held up to a camera, and TTA over implausible views tends to
            # hurt rather than help.
            o2, _ = tokens(model, torch.flip(x, dims=[3]), is_timm, n_prefix)
            clsf.append(o2[:, 0].float().cpu())
            meanf.append(o2[:, npre:].mean(1).float().cpu())
        seen += b
        since += b
        if since >= ckpt_every:
            flush(seen)
            since = 0
        if bi % log_every == 0:
            el = time.time() - t0
            new = seen - done
            rate = new / max(el, 1e-6)
            print(f"    {seen}/{total}  {rate:.1f} img/s  "
                  f"eta {(total - seen) / max(rate, 1e-6) / 60:.1f} min", flush=True)
    flush(seen)
    j = lambda L: torch.cat(L).numpy().astype(np.float32) if L else None
    return j(cls), j(mean), j(clsf), j(meanf)


def l2(a):
    return (a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


@torch.no_grad()
def extract_tiles(model, loader, device, is_timm, n_prefix, log_every=10,
                  ckpt=None, ckpt_every=320):
    """Per image: mean and max over its tiles, for both CLS and patch-mean.

    Max-pooling over tiles matters as much as mean here: a lesion may occupy one
    tile out of four, and averaging dilutes it by 4x while max keeps it. Both are
    saved so the probe can use either or both.
    """
    mc, xc, mm, xm = [], [], [], []
    done = 0
    if ckpt is not None and ckpt.exists():
        # Same open-handle trap as in extract(); see the comment there.
        with np.load(ckpt) as z:
            mc = [torch.from_numpy(z["tile_mean_cls"].copy())]
            xc = [torch.from_numpy(z["tile_max_cls"].copy())]
            mm = [torch.from_numpy(z["tile_mean_mean"].copy())]
            xm = [torch.from_numpy(z["tile_max_mean"].copy())]
            done = int(z["n"])
        print(f"    resuming from {ckpt.name}: {done} images done", flush=True)

    def flush(n):
        if ckpt is None:
            return
        cat = lambda L: torch.cat(L).numpy().astype(np.float32)
        tmp = ckpt.with_suffix(".tmp.npz")
        np.savez(tmp, tile_mean_cls=cat(mc), tile_max_cls=cat(xc),
                 tile_mean_mean=cat(mm), tile_max_mean=cat(xm), n=n)
        tmp.replace(ckpt)

    t0, seen, since = time.time(), 0, 0
    total = len(loader.dataset)
    for bi, xb in enumerate(loader):
        b, ntile = xb.shape[0], xb.shape[1]
        if seen + b <= done:
            seen += b
            continue
        flat = xb.reshape(-1, *xb.shape[2:]).to(device)
        out, npre = tokens(model, flat, is_timm, n_prefix)
        cls = out[:, 0].float().reshape(b, ntile, -1)
        pmean = out[:, npre:].mean(1).float().reshape(b, ntile, -1)
        mc.append(cls.mean(1).cpu())
        xc.append(cls.max(1).values.cpu())
        mm.append(pmean.mean(1).cpu())
        xm.append(pmean.max(1).values.cpu())
        seen += b
        since += b
        if since >= ckpt_every:
            flush(seen)
            since = 0
        if bi % log_every == 0:
            el = time.time() - t0
            rate = (seen - done) / max(el, 1e-6)
            print(f"    {seen}/{total}  {rate:.1f} img/s ({rate*ntile:.1f} tiles/s)"
                  f"  eta {(total-seen)/max(rate,1e-6)/60:.1f} min", flush=True)
    flush(seen)
    j = lambda L: torch.cat(L).numpy().astype(np.float32)
    return j(mc), j(xc), j(mm), j(xm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--backbone", default="dinov2-base",
                    choices=sorted(set(HF_ID) | set(TIMM_ID)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--flip", action="store_true", help="also extract the mirrored image (2x time)")
    ap.add_argument("--tiles", type=int, default=0,
                    help="n for an n x n tile grid (2 = 4 tiles, ~4x time). "
                         "Replaces the global pass with tile pooling.")
    ap.add_argument("--size", type=int, default=224, help="crop size; >224 interpolates pos-encodings")
    ap.add_argument("--shortest", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only-classes", default="",
                    help="comma-separated subset, e.g. Fungi,Pest,Phytopthora. "
                         "Produces a standalone cache with its own y/groups -- "
                         "NOT row-aligned with the full-dataset caches, so it "
                         "cannot be concatenated with them.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, labels, classes = tp.collect_images(Path(args.data_root))
    if args.only_classes:
        keep = [c.strip() for c in args.only_classes.split(",") if c.strip()]
        missing = [c for c in keep if c not in classes]
        if missing:
            raise SystemExit(f"--only-classes names {missing}, which are not in "
                             f"the dataset ({classes}).")
        sel = [i for i, l in enumerate(labels) if l in keep]
        paths = [paths[i] for i in sel]
        labels = [labels[i] for i in sel]
        classes = sorted(keep)
        print(f"  restricted to {classes}: {len(paths)} images")
    print(f"  {len(paths)} images, {len(classes)} classes: {classes}")

    # Same grouping call as the trainer, so group ids match the existing caches.
    groups = tp.group_near_duplicates(paths)
    y = np.array([classes.index(l) for l in labels], dtype=np.int64)

    import torchvision.transforms as T
    # Pinned to Xenova/dinov2-*'s preprocessor_config.json, the same transform
    # potato_infer.py and the browser use. Changing it here silently invalidates
    # parity with the deployed classifier.
    tf = T.Compose([
        T.Resize(args.shortest, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    is_timm = args.backbone in TIMM_ID
    n_prefix = 1
    if is_timm:
        print(f"  loading timm {TIMM_ID[args.backbone][0]} on {device} at "
              f"{args.size}px{' with flip' if args.flip else ''} ...")
        model = load_timm(args.backbone, args.size).to(device)
        n_prefix = 1 + int(getattr(model, "num_reg_tokens", 0) or 0)
        print(f"    prefix tokens (CLS + registers): {n_prefix}")
    else:
        from transformers import AutoModel
        print(f"  loading {HF_ID[args.backbone]} on {device} at {args.size}px"
              f"{' with flip' if args.flip else ''} ...")
        model = AutoModel.from_pretrained(HF_ID[args.backbone]).to(device).eval()

    ckpt = Path(str(args.out) + ".partial.npz")
    t0 = time.time()

    if args.tiles:
        ds = TiledImages(paths, args.shortest, args.size, n=args.tiles)
        # Each item is n*n images, so shrink the batch to keep memory comparable.
        bs = max(1, args.batch // (args.tiles * args.tiles))
        loader = DataLoader(ds, batch_size=bs, shuffle=False,
                            num_workers=args.workers)
        print(f"  {args.tiles}x{args.tiles} tiling: {args.tiles**2} tiles of "
              f"{args.size}px from a {args.size*args.tiles}px crop "
              f"(batch {bs} images = {bs*args.tiles**2} tiles)")
        tmc, txc, tmm, txm = extract_tiles(model, loader, device, is_timm,
                                           n_prefix, ckpt=ckpt)
        print(f"  extraction took {(time.time() - t0) / 60:.1f} min")
        views = {"tile_mean_cls": l2(tmc), "tile_max_cls": l2(txc),
                 "tile_mean_mean": l2(tmm), "tile_max_mean": l2(txm)}
        cls = tmc          # for the dimension print at the end
        np.savez_compressed(args.out, y=y,
                            groups=np.asarray(groups, dtype=np.int32),
                            classes=np.array(classes), **views)
        if ckpt.exists():
            ckpt.unlink()
        mb = Path(args.out).stat().st_size / 1024 / 1024
        print(f"  wrote {args.out}  ({mb:.1f} MB)")
        print(f"  views: {sorted(views)}  dim={cls.shape[1]}")
        for k, v in sorted(views.items()):
            cen = l2(np.stack([v[y == c].mean(0) for c in range(len(classes))]))
            sim = cen @ cen.T
            off = sim[~np.eye(len(classes), dtype=bool)]
            print(f"    {k:16s} mean off-diagonal centroid cosine {off.mean():.4f}")
        return 0

    loader = DataLoader(Images(paths, tf), batch_size=args.batch, shuffle=False,
                        num_workers=args.workers)
    cls, mean, clsf, meanf = extract(model, loader, args.flip, device,
                                     is_timm=is_timm, n_prefix=n_prefix,
                                     ckpt=ckpt)
    print(f"  extraction took {(time.time() - t0) / 60:.1f} min")

    views = {"cls": l2(cls), "mean": l2(mean)}
    if args.flip:
        views["cls_flip"] = l2(clsf)
        views["mean_flip"] = l2(meanf)
        # Average the raw vectors then renormalise -- averaging two already-unit
        # vectors and renormalising is the standard feature-space TTA.
        views["cls_tta"] = l2((cls + clsf) / 2.0)
        views["mean_tta"] = l2((mean + meanf) / 2.0)

    np.savez_compressed(args.out, y=y, groups=np.asarray(groups, dtype=np.int32),
                        classes=np.array(classes), **views)
    if ckpt.exists():
        ckpt.unlink()
    mb = Path(args.out).stat().st_size / 1024 / 1024
    print(f"  wrote {args.out}  ({mb:.1f} MB)")
    print(f"  views: {sorted(views)}  dim={cls.shape[1]}")
    # A quick separability signal so a bad run is obvious before any head is fit.
    for k, v in sorted(views.items()):
        cen = np.stack([v[y == c].mean(0) for c in range(len(classes))])
        cen = l2(cen)
        sim = cen @ cen.T
        off = sim[~np.eye(len(classes), dtype=bool)]
        print(f"    {k:10s} mean off-diagonal centroid cosine {off.mean():.4f}"
              f"  (lower = classes further apart)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
