"""Loading a trained checkpoint and running it on a single image.

A checkpoint carries its own `config.json` inside it, so inference asks the user
for nothing extra — the architecture, classes, input size and normalisation
settings all travel with the weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from core.schemas import Backend, Layout, RunConfig, Task
from data.readers import read_image
from trainers.base import resolve_device


@dataclass
class LoadedModel:
    model: object
    cfg: RunConfig
    device: torch.device
    classes: list[str]
    epoch: int | None = None
    best_value: float | None = None

    @property
    def task(self) -> Task:
        return self.cfg.dataset.task

    @property
    def backend(self) -> Backend:
        return self.cfg.model.backend


@dataclass
class Prediction:
    """Which fields are populated depends on the task."""
    task: Task
    probs: np.ndarray | None = None          # classification: (K,)
    label: str = ""
    confidence: float = 0.0
    mask: np.ndarray | None = None           # segmentation: (H,W) class indices
    overlay: np.ndarray | None = None        # the visualised result
    class_areas: dict[str, float] = field(default_factory=dict)
    inference_ms: float = 0.0


def load_checkpoint(run_dir: str | Path, which: str = "best") -> LoadedModel:
    """Get the checkpoint in a run directory ready for inference."""
    run_dir = Path(run_dir)
    ckpt_path = run_dir / (Layout.BEST if which == "best" else Layout.LAST)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"There is no checkpoint: {ckpt_path}")

    cfg = RunConfig.load(run_dir / Layout.CONFIG)
    device = resolve_device(cfg.device)

    if cfg.model.backend == Backend.ULTRALYTICS:
        from ultralytics import YOLO

        model = YOLO(str(ckpt_path))
        return LoadedModel(model=model, cfg=cfg, device=device,
                           classes=cfg.dataset.classes)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = _rebuild(cfg)
    model.load_state_dict(state["model"])
    model.eval().to(device)
    return LoadedModel(
        model=model, cfg=cfg, device=device,
        classes=state.get("classes") or cfg.dataset.classes,
        epoch=state.get("epoch"), best_value=state.get("best_value"),
    )


def _rebuild(cfg: RunConfig):
    """Rebuild the same architecture before loading the weights."""
    task, backend = cfg.dataset.task, cfg.model.backend
    n = max(2, cfg.dataset.num_classes)
    channels = cfg.dataset.channels if cfg.dataset.channels in (1, 3) else 3

    if backend == Backend.TIMM:
        import timm

        return timm.create_model(cfg.model.arch, pretrained=False,
                                 num_classes=n, in_chans=channels)
    if backend == Backend.TORCHVISION:
        # Built through the same helpers the trainer used, so the state-dict keys
        # line up — including the `net.` prefix the segmentation wrapper adds.
        from trainers.torchvision_common import build_classifier, build_segmenter

        builder = build_classifier if task == Task.CLASSIFICATION else build_segmenter
        model, _ = builder(cfg.model.arch, n_classes=n, channels=channels,
                           pretrained=False)
        return model
    if backend == Backend.SMP:
        import segmentation_models_pytorch as smp

        return smp.create_model(arch=cfg.model.arch,
                                encoder_name=cfg.model.encoder or "resnet50",
                                encoder_weights=None, in_channels=channels, classes=n)
    if backend == Backend.HF:
        from transformers import AutoConfig, AutoModelForSemanticSegmentation

        conf = AutoConfig.from_pretrained(cfg.model.arch, num_labels=n)
        return AutoModelForSemanticSegmentation.from_config(conf)
    if backend == Backend.MONAI:
        from core.events import EventWriter
        from trainers.seg3d_monai import MonaiSegmentation3DTrainer

        # Rather than duplicating the trainer's architecture-building logic, we
        # create an instance and call only build_model; no events are written.
        import tempfile

        writer = EventWriter(tempfile.mkdtemp())
        trainer = MonaiSegmentation3DTrainer(cfg, writer)
        model = trainer.build_model()
        writer.close()
        return model
    raise ValueError(f"Unsupported backend: {backend}")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────


def preprocess(loaded: LoadedModel, image: np.ndarray) -> torch.Tensor:
    """Apply exactly the validation transform used during training."""
    from data.datasets_2d import build_transforms

    cfg = loaded.cfg
    channels = cfg.dataset.channels if cfg.dataset.channels in (1, 3) else 3
    tf = build_transforms(cfg.aug, cfg.hp.img_size, train=False,
                          task=cfg.dataset.task, modality=cfg.dataset.modality,
                          channels=channels)
    return tf(image=image)["image"].unsqueeze(0)


def read_input(path: str | Path, loaded: LoadedModel) -> np.ndarray:
    return read_image(path, loaded.cfg.dataset.modality, loaded.cfg.dataset.window,
                      loaded.cfg.dataset.channels if
                      loaded.cfg.dataset.channels in (1, 3) else 3)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def predict(loaded: LoadedModel, image: np.ndarray) -> Prediction:
    import time

    from trainers.preview import overlay_mask

    t0 = time.time()

    if loaded.backend == Backend.ULTRALYTICS:
        return _predict_ultralytics(loaded, image, t0)

    x = preprocess(loaded, image).to(loaded.device)
    out = loaded.model(x)
    logits = out.logits if hasattr(out, "logits") else out

    if loaded.task == Task.CLASSIFICATION:
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return Prediction(
            task=loaded.task, probs=probs, label=loaded.classes[idx],
            confidence=float(probs[idx]),
            inference_ms=(time.time() - t0) * 1000,
        )

    import torch.nn.functional as F

    if logits.shape[-2:] != image.shape[:2]:
        logits = F.interpolate(logits.float(), size=image.shape[:2],
                               mode="bilinear", align_corners=False)
    mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
    total = mask.size
    areas = {loaded.classes[c]: float((mask == c).sum() / total)
             for c in np.unique(mask) if c < len(loaded.classes)}
    return Prediction(
        task=loaded.task, mask=mask, overlay=overlay_mask(image, mask),
        class_areas=areas, inference_ms=(time.time() - t0) * 1000,
    )


def _predict_ultralytics(loaded: LoadedModel, image: np.ndarray, t0: float) -> Prediction:
    import time

    from trainers.preview import overlay_mask

    results = loaded.model.predict(image, imgsz=loaded.cfg.hp.img_size, verbose=False)
    r = results[0]
    ms = (time.time() - t0) * 1000

    if loaded.task == Task.CLASSIFICATION and getattr(r, "probs", None) is not None:
        probs = r.probs.data.cpu().numpy()
        idx = int(probs.argmax())
        names = [r.names[i] for i in sorted(r.names)] if getattr(r, "names", None) \
            else loaded.classes
        return Prediction(task=loaded.task, probs=probs,
                          label=names[idx] if idx < len(names) else str(idx),
                          confidence=float(probs[idx]), inference_ms=ms)

    mask = np.zeros(image.shape[:2], dtype=np.int32)
    if getattr(r, "masks", None) is not None and r.masks is not None:
        import cv2

        classes = r.boxes.cls.cpu().numpy().astype(int) if getattr(r, "boxes", None) is not None \
            else np.zeros(len(r.masks.data), dtype=int)
        for m, c in zip(r.masks.data.cpu().numpy(), classes):
            resized = cv2.resize(m, (image.shape[1], image.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            # YOLO class indices exclude the background; +1 moves them into our scheme
            mask[resized > 0.5] = int(c) + 1

    total = mask.size
    areas = {loaded.classes[c] if c < len(loaded.classes) else str(c):
             float((mask == c).sum() / total) for c in np.unique(mask)}
    return Prediction(task=loaded.task, mask=mask,
                      overlay=overlay_mask(image, mask),
                      class_areas=areas, inference_ms=ms)


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM — shows where a classification model is looking
# ─────────────────────────────────────────────────────────────────────────────


def gradcam(loaded: LoadedModel, image: np.ndarray, class_idx: int | None = None
            ) -> np.ndarray | None:
    """Return the image with a heatmap overlaid; None if it cannot be produced."""
    if loaded.task != Task.CLASSIFICATION or \
            loaded.backend not in (Backend.TIMM, Backend.TORCHVISION):
        return None
    try:
        import cv2
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        return None

    model = loaded.model
    # Target the last convolution/block. The attribute differs per family, and
    # for a plain ViT none of these match — Grad-CAM then returns None rather
    # than a heatmap built from the wrong tensor.
    target_layer = None
    for attr in ("layer4", "stages", "blocks", "features", "norm"):
        mod = getattr(model, attr, None)
        if mod is not None:
            try:
                target_layer = mod[-1] if hasattr(mod, "__getitem__") else mod
            except Exception:
                target_layer = mod
            break
    if target_layer is None:
        return None

    try:
        x = preprocess(loaded, image).to(loaded.device)
        targets = None
        if class_idx is not None:
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            targets = [ClassifierOutputTarget(class_idx)]
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            grayscale = cam(input_tensor=x, targets=targets)[0]
        base = cv2.resize(image, (grayscale.shape[1], grayscale.shape[0]))
        rgb = np.float32(base) / 255.0
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        return show_cam_on_image(rgb, grayscale, use_rgb=True)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Export (triggered from the UI)
# ─────────────────────────────────────────────────────────────────────────────


def export_to(loaded: LoadedModel, fmt: str) -> Path:
    """`fmt`: "onnx" | "torchscript" — returns the path of the file produced."""
    cfg = loaded.cfg
    out_dir = cfg.run_dir / Layout.EXPORTS
    out_dir.mkdir(parents=True, exist_ok=True)

    if loaded.backend == Backend.ULTRALYTICS:
        import shutil

        produced = loaded.model.export(format=fmt, imgsz=cfg.hp.img_size)
        target = out_dir / Path(produced).name
        if Path(produced).resolve() != target.resolve():
            shutil.move(str(produced), target)
        return target

    channels = cfg.dataset.channels if cfg.dataset.channels in (1, 3) else 3
    dummy = torch.randn(1, channels, cfg.hp.img_size, cfg.hp.img_size)
    model = loaded.model.to("cpu").eval()
    try:
        if fmt == "onnx":
            path = out_dir / "model.onnx"
            torch.onnx.export(model, dummy, str(path),
                              input_names=["input"], output_names=["output"],
                              dynamic_axes={"input": {0: "batch"},
                                            "output": {0: "batch"}},
                              opset_version=17)
        else:
            path = out_dir / "model.torchscript"
            torch.jit.save(torch.jit.trace(model, dummy), str(path))
        return path
    finally:
        model.to(loaded.device)
