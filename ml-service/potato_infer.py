"""
Inference for the potato leaf disease classifier.

Consumes the artifacts produced by ml-training/train_potato.py:

    artifacts/fold_0_head.pt      head weights + fitted temperature + class list
    artifacts/ood.npz             Mahalanobis means + precision
    artifacts/calibration.json    OOD threshold + abstention floor

This replaces the inference path in the old notebook, which was a different
computation from the one the model was trained on:

    for each of the 7 candidate classes c:
        build an input whose TEXT is prompt[c]
        forward, and read logits[0, label_map[c]]        # the 7x7 diagonal
    predicted = argmax over those 7 scalars
    confidence = softmax over 7 scalars from 7 SEPARATE forward passes

Training only ever presented matched (image, text_of_true_label) pairs, so every
off-diagonal entry was unconstrained extrapolation, and the "confidence" was a
softmax over numbers that are not on a comparable scale.

Here: one forward pass, image only, temperature-scaled probabilities, and an
explicit reject option.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_TORCH_OK = True
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from torchvision import transforms as T
except Exception as exc:                                  # pragma: no cover
    _TORCH_OK = False
    _IMPORT_ERROR = exc

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)

BACKBONES = {
    "dinov2": ("facebook/dinov2-large", 1024),
    "dinov2-base": ("facebook/dinov2-base", 768),
    "clip": ("openai/clip-vit-base-patch32", 512),
}


class Verdict(dict):
    """A prediction, or an explicit refusal to make one.

    `status` is one of:
        ok            a calibrated prediction
        not_a_leaf    out of distribution -- do NOT show a disease name
        uncertain     in distribution but below the confidence floor
    """


class PotatoClassifier:
    def __init__(self, artifacts_dir: str | Path = "artifacts", device: str | None = None):
        if not _TORCH_OK:
            raise RuntimeError(f"torch/torchvision/PIL unavailable: {_IMPORT_ERROR}")

        self.dir = Path(artifacts_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        head_files = sorted(self.dir.glob("fold_*_head.pt"))
        if not head_files:
            raise FileNotFoundError(
                f"No fold_*_head.pt in {self.dir.resolve()}. Train first:\n"
                "  python ml-training/train_potato.py --data-root <dataset> --backbone dinov2")

        # weights_only=True: a .pth is a pickle, and unpickling executes
        # arbitrary code. The old code called torch.load without it on a
        # checkpoint the docs told users to download from elsewhere.
        ckpts = [torch.load(f, map_location="cpu", weights_only=True) for f in head_files]
        first = ckpts[0]
        self.classes: list[str] = list(first["classes"])
        self.backbone_name: str = first["backbone"]
        self.feat_dim: int = int(first["feat_dim"])
        hidden = int(first.get("hidden", 0))

        # Ensemble the folds. Each was trained on a different grouped split, so
        # averaging their probabilities is free variance reduction at inference.
        self.heads = []
        self.temps = []
        for ck in ckpts:
            h = _build_head(self.feat_dim, len(self.classes), hidden)
            h.load_state_dict(ck["head"])
            h.eval().to(self.device)
            self.heads.append(h)
            self.temps.append(float(ck.get("temperature", 1.0)))

        self.backbone = _FrozenBackbone(self.backbone_name, self.device)
        self.tf = _eval_transform(self.backbone.kind)

        # The feature recipe, read from the checkpoint rather than hardcoded, so
        # the server cannot drift from what the head was actually fitted to. The
        # browser reads the same recipe out of the exported manifest.
        recipe = str(first.get("feature_recipe") or "global_cls")
        KNOWN = ("global_cls", "global_mean", "tile2x2_mean_cls", "tile2x2_mean_patch")
        self.views = [t.strip() for t in recipe.split("+") if t.strip() in KNOWN]
        if not self.views:
            self.views = ["global_cls"]
        self.tiling_grid = 2

        # Transform constants, kept in one place because the tiled transform has
        # to scale them and any mismatch is silent.
        self.shortest = 256
        self.crop = 224
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

        expected = len(self.views) * 768
        if self.feat_dim != expected:
            logger.warning(
                "feat_dim %d does not match %d views x 768 from recipe %r; "
                "inference will refuse rather than guess.",
                self.feat_dim, len(self.views), recipe)
        logger.info("feature recipe: %s (%d-d)", " + ".join(self.views), self.feat_dim)

        # OOD + abstention
        self.ood_means = self.ood_prec = None
        self.ood_threshold = float("inf")
        self.confidence_floor = 0.0
        ood_path, cal_path = self.dir / "ood.npz", self.dir / "calibration.json"
        if ood_path.exists():
            z = np.load(ood_path)
            self.ood_means, self.ood_prec = z["means"], z["precision"]
        if cal_path.exists():
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
            self.ood_threshold = float(cal.get("ood_mahalanobis_threshold_p99", float("inf")))
            self.confidence_floor = float(cal.get("abstain_below_confidence", 0.0))
        if self.ood_means is None:
            logger.warning("No ood.npz -- the is-this-a-potato-leaf gate is DISABLED. "
                           "Non-leaf images will be classified into one of %d diseases.",
                           len(self.classes))

        logger.info("PotatoClassifier ready: %s, %d folds, %d classes, device=%s",
                    self.backbone_name, len(self.heads), len(self.classes), self.device)

    # ── features ────────────────────────────────────────────────────────────
    @staticmethod
    def _upright(img: "Image.Image") -> "Image.Image":
        """Apply the EXIF orientation tag, matching what the browser already does.

        MEASURED DIVERGENCE. createImageBitmap() honours EXIF orientation by
        default -- a 4x2 JPEG tagged Orientation=6 decodes as 2x4 in Chrome --
        while PIL's Image.open() ignores the tag entirely. So the SAME phone photo
        was upright in the browser path and rotated 90 degrees in the server path,
        producing different features and therefore different verdicts from the two
        halves of the same product. Phone cameras set this tag constantly.

        Rotating here is safe rather than a distribution shift: of 400 training
        images sampled, ZERO carry an orientation tag, so exif_transpose is a
        no-op on the data the head was fitted to. Making user photos upright moves
        them TOWARDS that distribution, not away from it.
        """
        try:
            return ImageOps.exif_transpose(img)
        except Exception:
            # A malformed EXIF block must not stop a farmer getting a diagnosis.
            return img

    def _tile_transform(self, grid: int):
        """The tiled transform: resize/crop grid-times larger, then cut.

        Mirrors ml-training/extract_views.py TiledImages exactly, because that is
        what produced the features the head was fitted to.
        """
        import torchvision.transforms as T
        return T.Compose([
            T.Resize(self.shortest * grid, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.crop * grid),
            T.ToTensor(),
            T.Normalize(self.norm_mean, self.norm_std),
        ])

    @torch.no_grad()
    def _one_pass(self, x: torch.Tensor):
        """RAW (cls, patch_mean) for one batch. Deliberately NOT normalised here.

        Normalisation ORDER is load-bearing, and getting it wrong was silent.
        extract_views.py, which produced the training features, averages the RAW
        per-tile vectors and L2-normalises the result ONCE:

            mc.append(cls.mean(1))            # raw mean over tiles
            views["tile_mean_cls"] = l2(mc)   # one normalisation, afterwards

        Normalising each tile first and then averaging yields a different vector.
        That looked harmless and was not: the OOD gate began rejecting genuine
        training images as "not a potato leaf" -- Mahalanobis 3257 against a p99
        threshold of 2702 -- because the live vector no longer lay in the
        distribution the gate was fitted on. So raw here, normalise at the end.
        """
        # .vision is the HF model inside the wrapper; the wrapper's own forward()
        # returns only the pooled vector, and the tiled recipe needs patch tokens.
        out = self.backbone.vision(pixel_values=x).last_hidden_state
        return out[:, 0], out[:, 1:].mean(1)

    @torch.no_grad()
    def _features(self, img: "Image.Image") -> torch.Tensor:
        """Build exactly the vector the head was fitted to.

        THE TWO HALVES MUST AGREE. The browser builds this same vector in
        public/js/potato-browser.js from manifest.backbone.views. If the server
        and the browser disagree on which views, in which order, or how each is
        normalised, the same photo gets two different diagnoses and nothing
        errors -- so both sides read the recipe rather than hardcoding it.

        Measured: global CLS alone 0.8622 accuracy on grouped 5-fold CV;
        global + the two 2x2 tile views 0.8816, which beats adding a second
        609 MB backbone.
        """
        rgb = self._upright(img).convert("RGB")

        need_global = any(v.startswith("global") for v in self.views)
        need_tiles = any(v.startswith("tile") for v in self.views)

        g_cls = g_patch = None
        if need_global:
            x = self.tf(rgb).unsqueeze(0).to(self.device)
            raw_cls, raw_patch = self._one_pass(x)
            g_cls = torch.nn.functional.normalize(raw_cls, dim=-1)
            g_patch = torch.nn.functional.normalize(raw_patch, dim=-1)

        t_cls = t_patch = None
        if need_tiles:
            grid = int(self.tiling_grid or 2)
            big = self._tile_transform(grid)(rgb)          # (3, crop*grid, crop*grid)
            c = self.crop
            tiles = torch.stack([
                big[:, r * c:(r + 1) * c, k * c:(k + 1) * c]
                for r in range(grid) for k in range(grid)
            ]).to(self.device)
            cls, patch = self._one_pass(tiles)
            # Average over tiles, then renormalise -- the same order
            # extract_views.py used when it wrote the training features.
            t_cls = torch.nn.functional.normalize(cls.mean(0, keepdim=True), dim=-1)
            t_patch = torch.nn.functional.normalize(patch.mean(0, keepdim=True), dim=-1)

        blocks = []
        for v in self.views:
            if v == "global_cls":
                blocks.append(g_cls)
            elif v == "global_mean":
                blocks.append(g_patch)
            elif v == "tile2x2_mean_cls":
                blocks.append(t_cls)
            elif v == "tile2x2_mean_patch":
                blocks.append(t_patch)
            else:
                raise RuntimeError(f"Unknown feature view {v!r} in the checkpoint recipe.")

        feats = torch.cat(blocks, dim=-1)
        if feats.shape[-1] != self.feat_dim:
            raise RuntimeError(
                f"built {feats.shape[-1]}-d features but the head expects "
                f"{self.feat_dim}-d. Views {self.views} do not reproduce the "
                f"recipe the head was fitted to.")
        return feats

    # ── prediction ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, image_path: str | Path) -> Verdict:
        with Image.open(image_path) as im:
            feats = self._features(im)

        f_np = feats.cpu().numpy()

        # 1. Is this even a potato leaf?
        ood_score = None
        if self.ood_means is not None:
            ood_score = float(_mahalanobis(f_np, self.ood_means, self.ood_prec)[0])
            if ood_score > self.ood_threshold:
                return Verdict(
                    status="not_a_leaf",
                    # NOT "filling the frame". That was the advice here, and it is
                    # the single thing most likely to cause this refusal.
                    #
                    # Measured on the 7 parity images, aspect ratio held constant
                    # so framing is the only variable (mean Mahalanobis, p99
                    # threshold 2713):
                    #
                    #   crop 10% into the leaf   3976   6/7 rejected
                    #   as trained               2019   0/7 rejected
                    #   15% extra margin         4437   6/7 rejected
                    #   crop 30% into the leaf   6718   7/7 rejected
                    #
                    # The gate is knife-edge on framing in BOTH directions, and
                    # tighter framing is worse. Telling a farmer to fill the frame
                    # was telling them to do the thing that gets them refused, and
                    # then refusing them for it.
                    message=(
                        "This does not look like a potato leaf, so no diagnosis was "
                        "produced.\n\n"
                        "Two things cause this most often:\n"
                        "- Part of the leaf is outside the photo. Step back so the "
                        "WHOLE leaf is in view with a little space around it. Do not "
                        "fill the frame.\n"
                        "- It is a close-up of one spot. This model reads whole "
                        "leaves and cannot judge a zoomed-in patch.\n\n"
                        "Take the photo straight down over a single leaf, in daylight, "
                        "with the whole leaf visible."),
                    ood_score=ood_score, ood_threshold=self.ood_threshold)

        # 2. One forward pass per fold, temperature-scaled, then averaged.
        probs = torch.zeros(len(self.classes), device=self.device)
        for head, temp in zip(self.heads, self.temps):
            probs += F.softmax(head(feats).squeeze(0) / temp, dim=-1)
        probs = (probs / len(self.heads)).cpu().numpy()

        order = np.argsort(-probs)
        top = int(order[0])
        confidence = float(probs[top])
        all_probs = {c: float(p) for c, p in zip(self.classes, probs)}

        # 3. Confident enough to name a disease?
        if confidence < self.confidence_floor:
            return Verdict(
                status="uncertain",
                # Same correction as the not_a_leaf message above: "filling the
                # frame" is the advice most likely to make the next attempt worse.
                message=("Not confident enough to give a diagnosis. Retake the photo "
                         "in even daylight with the WHOLE leaf in view and a little "
                         "space around it -- do not fill the frame or zoom in. If it "
                         "is still uncertain, ask an agronomist."),
                confidence=confidence, confidence_floor=self.confidence_floor,
                all_probabilities=all_probs, ood_score=ood_score)

        return Verdict(
            status="ok",
            disease=self.classes[top],
            confidence=confidence,
            runner_up=self.classes[int(order[1])] if len(order) > 1 else None,
            runner_up_confidence=float(probs[order[1]]) if len(order) > 1 else None,
            all_probabilities=all_probs,
            ood_score=ood_score,
            calibrated=True,
            n_folds=len(self.heads))

    # ── explainability ──────────────────────────────────────────────────────
    def heatmap(self, image_path: str | Path, out_path: str | Path,
                alpha: float = 0.5) -> Optional[str]:
        """Attention-rollout overlay showing where the backbone looked.

        The cheapest possible check on whether the model reads lesions or reads
        background -- which is the failure mode the whole plant-disease
        literature warns about. Also the thing that makes a diagnosis
        reviewable: an agronomist will not act on a bare class name, but will
        look at a highlighted lesion and agree or disagree.
        """
        try:
            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
                x = self.tf(rgb).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    attn = self.backbone.attention_map(x)      # (h, w) in [0,1]
                if attn is None:
                    return None
                heat = Image.fromarray((attn * 255).astype(np.uint8)).resize(
                    rgb.size, Image.BILINEAR)
                heat_rgb = _colourise(np.asarray(heat, dtype=np.float32) / 255.0)
                base = np.asarray(rgb, dtype=np.float32) / 255.0
                blend = (1 - alpha) * base + alpha * heat_rgb
                Image.fromarray((np.clip(blend, 0, 1) * 255).astype(np.uint8)).save(out_path)
            return str(out_path)
        except Exception as exc:
            logger.warning("heatmap failed: %s", exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
class _Head(nn.Module):
    """MUST mirror train_potato.Head exactly, including the attribute name.

    This previously returned a bare nn.Sequential, so its state_dict keys were
    "1.weight" while the trainer saved "net.1.weight" -- meaning inference could
    never load a checkpoint that training produced. Both files compiled and both
    passed their own unit tests; only running the handoff caught it. If you change
    the head architecture in train_potato.py, change it here in the same commit.
    """

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 0, p_drop: float = 0.2):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(p_drop),
                nn.Linear(hidden, n_classes))
        else:
            self.net = nn.Sequential(nn.Dropout(p_drop), nn.Linear(in_dim, n_classes))

    def forward(self, x):
        return self.net(x)


def _build_head(in_dim: int, n_classes: int, hidden: int) -> "nn.Module":
    return _Head(in_dim, n_classes, hidden)


def _eval_transform(kind: str, size: int = 224):
    """MUST match Xenova/dinov2-base's preprocessor_config.json exactly, because
    the browser build (transformers.js) uses that config and the trained head is
    fitted to whatever features the transform produces.

    That config is: {"size": {"shortest_edge": 256}, "crop_size": 224,
    "resample": 3 (BICUBIC), "image_mean": ImageNet, "image_std": ImageNet}.

    This previously used Resize(int(224 * 1.14)) = 255 with torchvision's DEFAULT
    interpolation, which is BILINEAR. Both differences are silent: features shift
    slightly, nothing errors, and browser predictions drift from server
    predictions with no signal. 256 + BICUBIC is now pinned explicitly.
    """
    mean, std = (CLIP_MEAN, CLIP_STD) if kind == "clip" else (IMAGENET_MEAN, IMAGENET_STD)
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class _FrozenBackbone(nn.Module):
    def __init__(self, name: str, device):
        super().__init__()
        repo, self.dim = BACKBONES[name]
        self.kind = "clip" if name == "clip" else "dinov2"
        if self.kind == "clip":
            from transformers import CLIPModel
            m = CLIPModel.from_pretrained(repo)
            self.vision, self.proj = m.vision_model, m.visual_projection
        else:
            from transformers import AutoModel
            self.vision, self.proj = AutoModel.from_pretrained(repo), None
        for p in self.parameters():
            p.requires_grad = False
        self.eval().to(device)

    @torch.no_grad()
    def forward(self, pixel_values):
        if self.kind == "clip":
            feats = self.proj(self.vision(pixel_values=pixel_values)[1])
        else:
            out = self.vision(pixel_values=pixel_values)
            feats = out.pooler_output if out.pooler_output is not None \
                else out.last_hidden_state[:, 0]
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def attention_map(self, pixel_values):
        """Mean last-layer CLS attention over patches, reshaped to a grid."""
        out = self.vision(pixel_values=pixel_values, output_attentions=True)
        if not getattr(out, "attentions", None):
            return None
        a = out.attentions[-1][0].mean(0)          # (tokens, tokens), heads averaged
        cls_to_patch = a[0, 1:]                    # CLS row, drop the CLS column
        n = cls_to_patch.numel()
        side = int(round(n ** 0.5))
        if side * side != n:
            return None
        m = cls_to_patch.reshape(side, side).float().cpu().numpy()
        rng = m.max() - m.min()
        return (m - m.min()) / rng if rng > 1e-12 else np.zeros_like(m)


def _mahalanobis(feats: np.ndarray, means: np.ndarray, prec: np.ndarray) -> np.ndarray:
    d = np.empty((feats.shape[0], means.shape[0]))
    for c in range(means.shape[0]):
        z = feats - means[c]
        d[:, c] = np.einsum("ij,jk,ik->i", z, prec, z)
    return d.min(1)


def _colourise(g: np.ndarray) -> np.ndarray:
    """Small blue->green->red ramp, so the module needs no matplotlib."""
    r = np.clip(1.5 * g - 0.5, 0, 1)
    b = np.clip(1.0 - 2.0 * g, 0, 1)
    gr = np.clip(1.0 - np.abs(2.0 * g - 1.0), 0, 1)
    return np.stack([r, gr, b], axis=-1)


_singleton: Optional[PotatoClassifier] = None


def get_classifier(artifacts_dir: str | Path = "artifacts") -> Optional[PotatoClassifier]:
    """Load once, or return None if the artifacts are absent.

    Returning None is deliberate: the caller must then answer 503, never guess.
    """
    global _singleton
    if _singleton is None:
        try:
            _singleton = PotatoClassifier(artifacts_dir)
        except Exception as exc:
            logger.warning("Classifier unavailable: %s", exc)
            return None
    return _singleton
