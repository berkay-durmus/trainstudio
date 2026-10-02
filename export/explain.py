"""Explaining one prediction: Grad-CAM, LIME and SHAP.

Each method answers "which parts of the image made the model say this?" in its
own way:

- **Grad-CAM** weighs the last spatial feature map by the gradient of the class
  score. One forward and one backward pass; it needs a layer with a spatial
  layout, which is found from the model's structure (`_target_layer`).
- **LIME** splits the image into superpixels, switches random subsets of them off,
  and fits a linear model to how the score responds.
- **SHAP** (the Partition explainer) blurs regions in a hierarchy and shares the
  change in score out among them as Shapley values.

LIME and SHAP only ever call `ScoreFn`: images in, the score of one class out.
That is what lets them explain every backend the same way — a segmentation model
included (Mask2Former too), whose score is the class's mean probability over the
region predicted for it. Everything works on the image as the model sees it, resized to the
training size, and is scaled back to the input for display.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from core.schemas import Backend, Task

METHODS: dict[str, dict] = {
    "gradcam": {"label": "Grad-CAM",
                "about": "Where the last feature map responds to the class — weighted by the "
                         "gradient of its score. Instant."},
    "lime": {"label": "LIME",
             "about": "Which superpixels raise (green) or lower (red) the class score, from a "
                      "linear model fitted to hundreds of copies with regions switched off."},
    "shap": {"label": "SHAP",
             "about": "How much each region adds to (red) or takes from (blue) the class score, "
                      "by blurring regions in turn and sharing out the difference."},
}
# LIME samples · SHAP evaluations
BUDGETS = {"fast": (300, 300), "balanced": (800, 1000), "thorough": (2000, 3000)}
MISSING = {"gradcam": "grad-cam", "lime": "lime", "shap": "shap"}


@dataclass
class Explanation:
    method: str
    overlay: np.ndarray          # RGB uint8, the size of the input image
    heat: np.ndarray             # float32 H×W: [0, 1] for Grad-CAM, [-1, 1] otherwise
    elapsed_s: float
    note: str = ""


class ExplainError(RuntimeError):
    pass


def evaluations(method: str, budget: str) -> int:
    """How many model evaluations a method makes: what its cost scales with."""
    if method == "gradcam":
        return 1
    return BUDGETS[budget][0 if method == "lime" else 1]


def seconds_per_image(loaded, batch: int = 8) -> float:
    """Measured model throughput, to say beforehand how long LIME or SHAP will take."""
    m = _Model(loaded)
    images = np.zeros((batch, m.size, m.size, 3), dtype=np.uint8)
    m.logits(images[:1])                            # the first call pays for warm-up
    t0 = time.time()
    m.logits(images, batch=batch)
    return (time.time() - t0) / batch


# ─────────────────────────────────────────────────────────────────────────────
# The model as a plain function of a batch
# ─────────────────────────────────────────────────────────────────────────────


class _Logits(nn.Module):
    """Whatever the model returns, as one logits tensor (B, K) or (B, K, H, W)."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from export.inference import model_logits

        return model_logits(self.net(x))


class _Model:
    """The network, its input size and its preprocessing, the same for every backend."""

    def __init__(self, loaded) -> None:
        self.loaded = loaded
        self.size = int(loaded.cfg.hp.img_size)
        self.device = loaded.device
        if loaded.backend == Backend.ULTRALYTICS:
            net = loaded.model.model
            self.channels = 3
            self._tf = None
        else:
            from data.datasets_2d import build_transforms

            net = loaded.model
            cfg = loaded.cfg
            self.channels = cfg.dataset.channels if cfg.dataset.channels in (1, 3) else 3
            self._tf = build_transforms(cfg.aug, self.size, train=False, task=cfg.dataset.task,
                                        modality=cfg.dataset.modality, channels=self.channels)
        self.raw = net
        self.net = _Logits(net).eval().to(self.device)

    def prep(self, images: np.ndarray) -> torch.Tensor:
        """N×S×S×3 RGB (any numeric dtype) → the model's input batch."""
        images = np.clip(images, 0, 255).astype(np.uint8)
        if self._tf is None:                        # Ultralytics: /255, no normalisation
            x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        else:
            if self.channels == 1:
                images = images[..., :1]
            x = torch.stack([self._tf(image=im)["image"] for im in images])
        return x.to(self.device)

    @torch.no_grad()
    def logits(self, images: np.ndarray, batch: int = 32) -> torch.Tensor:
        outs = [self.net(self.prep(images[i:i + batch])).float().cpu()
                for i in range(0, len(images), batch)]
        return torch.cat(outs)


def _rgb(image: np.ndarray) -> np.ndarray:
    rgb = image if image.ndim == 3 else image[..., None]
    if rgb.shape[-1] == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    rgb = rgb[..., :3]
    return rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)


def _work_image(image: np.ndarray, size: int) -> np.ndarray:
    """RGB uint8 at the training size, which is how the model sees it."""
    import cv2

    rgb = _rgb(image)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA
                      if max(rgb.shape[:2]) > size else cv2.INTER_LINEAR)


class ScoreFn:
    """images (N×S×S×3) → (N, 1): the score of the class being explained.

    Classification: its probability. Segmentation: its mean probability over the
    pixels predicted as that class in the unaltered image (all pixels when it was
    not predicted anywhere) — "how sure is the model about this region".
    """

    def __init__(self, m: _Model, work: np.ndarray, target: int) -> None:
        self.m, self.target = m, target
        self.region = None
        if m.loaded.task == Task.SEGMENTATION:
            probs = torch.softmax(m.logits(work[None]), dim=1)[0]
            region = probs.argmax(0) == target
            self.region = region if region.any() else torch.ones_like(region)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        probs = torch.softmax(self.m.logits(np.asarray(images)), dim=1)
        if self.region is None:
            score = probs[:, self.target]
        else:
            p = probs[:, self.target]
            region = self.region
            if p.shape[-2:] != region.shape:
                region = torch.nn.functional.interpolate(
                    region[None, None].float(), size=p.shape[-2:], mode="nearest")[0, 0] > 0
            score = p[:, region].mean(1)
        return score.numpy()[:, None].astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Availability
# ─────────────────────────────────────────────────────────────────────────────


def available(loaded) -> dict[str, tuple[bool, str]]:
    """Per method: can it explain this model, and if not, why."""
    if loaded.task not in (Task.CLASSIFICATION, Task.SEGMENTATION):
        return {k: (False, "Explanations cover 2D classification and segmentation.")
                for k in METHODS}
    if loaded.backend == Backend.ULTRALYTICS and loaded.task != Task.CLASSIFICATION:
        return {k: (False, "For Ultralytics models, explanations cover classification; its "
                           "segmentation models predict instances, not a score per pixel.")
                for k in METHODS}
    out = {}
    for key, package in MISSING.items():
        try:
            __import__({"gradcam": "pytorch_grad_cam"}.get(key, key))
            out[key] = (True, "")
        except ImportError:
            out[key] = (False, f"`pip install {package}` is needed for {METHODS[key]['label']}.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def explain(loaded, image: np.ndarray, method: str, target: int,
            budget: str = "balanced", seed: int = 0) -> Explanation:
    ok, why = available(loaded).get(method, (False, f"Unknown method `{method}`."))
    if not ok:
        raise ExplainError(why)
    m = _Model(loaded)
    work = _work_image(image, m.size)
    name = loaded.classes[target] if target < len(loaded.classes) else str(target)
    t0 = time.time()
    if method == "gradcam":
        heat = _gradcam(m, work, target)
    elif method == "lime":
        heat = _lime(m, work, target, BUDGETS[budget][0], seed)
    else:
        heat = _shap(m, work, target, BUDGETS[budget][1], seed)
    elapsed = time.time() - t0

    # LIME's map is piecewise constant, one value per superpixel; keep it so
    heat = _to_input_size(heat, image.shape[:2], nearest=method == "lime")
    overlay = _render(method, _rgb(image), heat)
    note = (f"Explains the mean probability of “{name}” over the region predicted for it."
            if loaded.task == Task.SEGMENTATION else f"Explains the probability of “{name}”.")
    return Explanation(method, overlay, heat.astype(np.float32), elapsed, note)


def _to_input_size(heat: np.ndarray, shape: tuple[int, int], nearest: bool = False) -> np.ndarray:
    import cv2

    return cv2.resize(heat.astype(np.float32), (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

_SKIP = (nn.ReLU, nn.ReLU6, nn.SiLU, nn.GELU, nn.Hardswish, nn.Sigmoid, nn.Identity,
         nn.Dropout, nn.Dropout2d, nn.LeakyReLU, nn.Softmax)


def _square(n: int) -> bool:
    return n >= 4 and math.isqrt(n) ** 2 == n


def _prefix(model: nn.Module, n_tokens: int) -> int:
    """Class / register tokens in front of the patch tokens."""
    k = getattr(model, "num_prefix_tokens", None)
    if isinstance(k, int) and _square(n_tokens - k):
        return k
    side = math.isqrt(n_tokens)
    return n_tokens - side * side


def _spatial(out, model: nn.Module) -> bool:
    """Has this output a spatial layout Grad-CAM can map back onto the image?"""
    if not isinstance(out, torch.Tensor):
        return False
    if out.ndim == 4:
        return (out.shape[2] == out.shape[3] and out.shape[2] > 1) or \
               (out.shape[1] == out.shape[2] and out.shape[1] > 1)
    if out.ndim == 3:
        return _square(out.shape[1] - _prefix(model, out.shape[1]))
    return False


def reshape_transform(model: nn.Module):
    """Activations → (B, C, h, w): NCHW kept, NHWC permuted (Swin), tokens folded (ViT)."""
    def fn(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 4:
            if t.shape[2] == t.shape[3]:
                return t
            return t.permute(0, 3, 1, 2)
        if t.ndim == 3:
            t = t[:, _prefix(model, t.shape[1]):, :]
            side = math.isqrt(t.shape[1])
            return t.reshape(t.shape[0], side, side, t.shape[2]).permute(0, 3, 1, 2)
        return t
    return fn


def _search_scope(m: _Model) -> nn.Module:
    """Where the feature maps live: the encoder of a segmentation model (smp,
    torchvision, Mask2Former; for SegFormer the model without its decode head)."""
    raw, loaded = m.raw, m.loaded
    if loaded.task == Task.SEGMENTATION:
        for path in ("encoder", "net.backbone", "model.pixel_level_module.encoder",
                     "backbone", "base_model"):
            mod = raw
            for part in path.split("."):
                mod = getattr(mod, part, None)
                if mod is None:
                    break
            if mod is not None:
                return mod
    return raw


def _target_layer(m: _Model, work: np.ndarray) -> nn.Module:
    """The layer Grad-CAM watches.

    Token models whose class token is the output: the first norm of the last
    block, because past it the patch tokens no longer reach the score (timm ViT,
    EVA and DINOv3; torchvision ViT). Everything else: the module that finishes
    last, in a forward pass, with an output that still has a spatial layout —
    found by running one, so no list of attribute names has to follow the
    catalogue. Containers count, so a ResNet gives `layer4` (after the residual
    sum and its ReLU) rather than the last BatchNorm inside it.
    """
    raw = m.raw
    blocks = getattr(raw, "blocks", None)
    if blocks is not None and getattr(raw, "num_prefix_tokens", 0) and hasattr(blocks[-1], "norm1"):
        return blocks[-1].norm1
    enc = getattr(raw, "encoder", None)
    layers = getattr(enc, "layers", None) if enc is not None else None
    if layers is not None and hasattr(layers[-1], "ln_1") and hasattr(raw, "class_token"):
        return layers[-1].ln_1

    scope = _search_scope(m)
    seen: list[nn.Module] = []
    hooks = []

    def hook(mod, _inp, out):
        if _spatial(out, raw):
            seen.append(mod)

    for mod in scope.modules():
        if mod is not scope and not isinstance(mod, _SKIP):
            hooks.append(mod.register_forward_hook(hook))
    try:
        with torch.no_grad():
            m.net(m.prep(work[None]))
    finally:
        for h in hooks:
            h.remove()
    if not seen:
        raise ExplainError("No layer with a spatial layout was found for Grad-CAM in this model.")
    return seen[-1]


def _gradcam(m: _Model, work: np.ndarray, target: int) -> np.ndarray:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import (
        ClassifierOutputTarget,
        SemanticSegmentationTarget,
    )

    layer = _target_layer(m, work)
    x = m.prep(work[None])
    if m.loaded.task == Task.SEGMENTATION:
        with torch.no_grad():
            pred = m.net(x).argmax(1)[0].cpu().numpy()
        mask = (pred == target).astype(np.float32)
        if not mask.any():
            mask[:] = 1.0
        targets = [SemanticSegmentationTarget(target, mask)]
    else:
        targets = [ClassifierOutputTarget(target)]
    # Ultralytics loads its weights frozen; the activations still need gradients
    params = [p for p in m.net.parameters() if not p.requires_grad]
    for p in params:
        p.requires_grad_(True)
    try:
        with GradCAM(model=m.net, target_layers=[layer],
                     reshape_transform=reshape_transform(m.raw)) as cam:
            heat = cam(input_tensor=x, targets=targets)[0]
    finally:
        for p in params:
            p.requires_grad_(False)
    return np.nan_to_num(heat).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────────────────────────────────────


def _lime(m: _Model, work: np.ndarray, target: int, samples: int, seed: int) -> np.ndarray:
    from lime import lime_image
    from skimage.segmentation import slic

    score = ScoreFn(m, work, target)
    segments = slic(work, n_segments=50, compactness=10, start_label=0)
    explainer = lime_image.LimeImageExplainer(random_state=seed)
    exp = explainer.explain_instance(
        work, score, labels=(0,), top_labels=None, hide_color=None,
        num_samples=samples, batch_size=32, segmentation_fn=lambda _img: segments,
        random_seed=seed)
    weights = dict(exp.local_exp[0])
    heat = np.zeros(segments.shape, dtype=np.float32)
    for seg, w in weights.items():
        heat[segments == seg] = w
    peak = float(np.abs(heat).max())
    return heat / peak if peak > 0 else heat


# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────


def _shap(m: _Model, work: np.ndarray, target: int, max_evals: int, seed: int) -> np.ndarray:
    import shap

    score = ScoreFn(m, work, target)
    blur = max(3, m.size // 14) | 1
    masker = shap.maskers.Image(f"blur({blur},{blur})", work.shape)
    explainer = shap.Explainer(score, masker, algorithm="partition")
    # The Partition explainer breaks ties between regions with np.random; seed it,
    # without disturbing anyone else's use of the global generator
    saved = np.random.get_state()
    np.random.seed(seed)
    try:
        values = explainer(work[None].astype(np.float64), max_evals=max_evals, batch_size=32,
                           silent=True).values
    finally:
        np.random.set_state(saved)
    heat = np.asarray(values)[0].reshape(work.shape[0], work.shape[1], -1).sum(-1)
    peak = float(np.abs(heat).max())
    return (heat / peak if peak > 0 else heat).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────


def _render(method: str, base: np.ndarray, heat: np.ndarray) -> np.ndarray:
    rgb = base.astype(np.float32) / 255.0
    if method == "gradcam":
        from pytorch_grad_cam.utils.image import show_cam_on_image

        return show_cam_on_image(rgb, np.clip(heat, 0, 1), use_rgb=True)
    if method == "lime":
        return _render_lime(rgb, heat)
    # SHAP: red adds to the score, blue takes from it
    pos, neg = np.clip(heat, 0, 1), np.clip(-heat, 0, 1)
    colour = np.zeros_like(rgb)
    colour[..., 0], colour[..., 2] = pos, neg
    alpha = (np.maximum(pos, neg) * 0.75)[..., None]
    grey = rgb.mean(-1, keepdims=True).repeat(3, -1) * 0.8 + 0.1
    return (np.clip(grey * (1 - alpha) + colour * alpha, 0, 1) * 255).astype(np.uint8)


def _render_lime(rgb: np.ndarray, heat: np.ndarray, top: int = 5) -> np.ndarray:
    """The strongest supporting superpixels tinted green, opposing ones red."""
    from skimage.segmentation import find_boundaries

    out = rgb * 0.55 + 0.2
    levels = np.unique(heat)
    pos = sorted((v for v in levels if v > 0), reverse=True)[:top]
    neg = sorted(v for v in levels if v < 0)[:top]
    for values, colour in ((pos, (0.1, 0.85, 0.3)), (neg, (0.95, 0.2, 0.2))):
        for v in values:
            region = heat == v
            out[region] = rgb[region] * 0.45 + np.array(colour) * 0.55
            out[find_boundaries(region, mode="inner")] = colour
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)
