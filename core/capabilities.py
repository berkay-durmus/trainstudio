"""Which hyperparameters each backend can actually apply, and how.

Libraries disagree on regularisation: the same "dropout" is `drop_rate` in timm,
`decoder_dropout` in smp's FPN, `dropout_prob` in MONAI's SegResNet — and some
architectures take a keyword only to swallow it (torchvision's ConvNeXt accepts
`dropout` and ignores it). Every entry here was checked against the installed
libraries by building the model and finding the value in its layers.

Each knob also carries the library's own default for that architecture. The
recommendation starts from it, so a run nobody touched trains exactly as it did
before these fields were exposed: passing 0.0 where FPN expects 0.2 would
quietly change the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.schemas import Backend, Task


@dataclass(frozen=True)
class Knob:
    param: str          # the keyword the library takes
    default: float      # the library's own default for this architecture


# torchvision classifiers in the catalogue, by architecture prefix:
# (dropout, stochastic depth). ResNet and RegNet take neither; ConvNeXt takes
# `dropout` and MobileNetV3 `stochastic_depth_prob` only to ignore them; Swin
# and ViT reject `stochastic_depth_prob`. Anything else — a model picked from
# outside the catalogue — gets no knob rather than a guess.
_TV_CLS: dict[str, tuple[Knob | None, Knob | None]] = {
    "efficientnet_v2_s":  (Knob("dropout", 0.2), Knob("stochastic_depth_prob", 0.2)),
    "mobilenet_v3_large": (Knob("dropout", 0.2), None),
    "vit_b_16":           (Knob("dropout", 0.0), None),
    "swin_v2_":           (Knob("dropout", 0.0), None),
    "convnext_tiny":      (None, Knob("stochastic_depth_prob", 0.1)),
    "convnext_base":      (None, Knob("stochastic_depth_prob", 0.5)),
}

# smp decoders with a dropout of their own; none of them has stochastic depth
_SMP_DROPOUT = {
    "fpn": Knob("decoder_dropout", 0.2),
    "pspnet": Knob("psp_dropout", 0.2),
    "deeplabv3": Knob("decoder_aspp_dropout", 0.5),
    "deeplabv3plus": Knob("decoder_aspp_dropout", 0.5),
}

# MONAI 3D networks: (dropout, stochastic depth)
_MONAI: dict[str, tuple[Knob | None, Knob | None]] = {
    "SwinUNETR": (Knob("drop_rate", 0.0), Knob("dropout_path_rate", 0.0)),
    "UNETR": (Knob("dropout_rate", 0.0), None),
    "SegResNet": (Knob("dropout_prob", 0.0), None),
    "UNet": (Knob("dropout", 0.0), None),
    "AttentionUnet": (Knob("dropout", 0.0), None),
    "DynUNet": (Knob("dropout", 0.0), None),
}

# timm models whose constructor rejects drop_path_rate outright
_TIMM_NO_DROP_PATH = ("densenet",)


def knobs(backend: Backend, arch: str, task: Task) -> dict[str, Knob]:
    """{"drop_rate" | "drop_path_rate": Knob} for the knobs this model honours."""
    low = arch.lower()
    out: dict[str, Knob] = {}

    def put(field: str, knob: Knob | None) -> None:
        if knob is not None:
            out[field] = knob

    if backend == Backend.TIMM:
        put("drop_rate", Knob("drop_rate", 0.0))
        if not low.startswith(_TIMM_NO_DROP_PATH):
            put("drop_path_rate", Knob("drop_path_rate", 0.0))
    elif backend == Backend.TORCHVISION and task == Task.CLASSIFICATION:
        for prefix, (drop, path) in _TV_CLS.items():
            if low.startswith(prefix):
                put("drop_rate", drop)
                put("drop_path_rate", path)
                break
    elif backend == Backend.SMP:
        put("drop_rate", _SMP_DROPOUT.get(low))
    elif backend == Backend.HF:
        if "mask2former" in low:
            put("drop_rate", Knob("dropout", 0.0))
        elif "segformer" in low:
            put("drop_rate", Knob("classifier_dropout_prob", 0.1))
            put("drop_path_rate", Knob("drop_path_rate", 0.1))
    elif backend == Backend.MONAI:
        drop, path = _MONAI.get(arch, (None, None))
        put("drop_rate", drop)
        put("drop_path_rate", path)
    elif backend == Backend.ULTRALYTICS and task == Task.CLASSIFICATION:
        put("drop_rate", Knob("dropout", 0.0))
    return out


def spec_knobs(spec) -> dict[str, Knob]:
    return knobs(spec.backend, spec.arch, spec.task)


def model_kwargs(cfg) -> dict[str, float]:
    """The library keywords for a run's dropout settings.

    Only values that differ from the library's own default are passed, so an
    untouched run builds exactly the model it built before. A version-1
    config.json predates this: only timm applied these fields then, so for any
    other backend nothing is passed and the old run is reproduced as it was.
    """
    if cfg.version < 2 and cfg.model.backend != Backend.TIMM:
        return {}
    out = {}
    for field, k in knobs(cfg.model.backend, cfg.model.arch, cfg.dataset.task).items():
        value = float(getattr(cfg.hp, field))
        if abs(value - k.default) > 1e-9:
            out[k.param] = value
    return out


def ignored(cfg) -> list[str]:
    """Dropout fields set to something this model cannot apply — for the run log."""
    have = knobs(cfg.model.backend, cfg.model.arch, cfg.dataset.task)
    return [f for f in ("drop_rate", "drop_path_rate")
            if f not in have and getattr(cfg.hp, f)]


def log_applied(cfg, writer, applied: dict[str, float]) -> None:
    """One log line per run saying which regularisation reached the model."""
    if applied:
        writer.log("Regularisation: " + ", ".join(f"{k}={v:g}" for k, v in applied.items()))
    for f in ignored(cfg):
        writer.log(f"{f}={getattr(cfg.hp, f):g} is not applied: "
                   f"{cfg.model.arch} has no such setting.", "warning")


_ULTRA = "Ultralytics runs its own training loop"


def supports(spec, field: str) -> tuple[bool, str]:
    """(can this backend apply `field`, why not when it cannot)."""
    if field in ("drop_rate", "drop_path_rate"):
        if field in spec_knobs(spec):
            return True, ""
        what = "dropout" if field == "drop_rate" else "stochastic depth"
        return False, f"{spec.display_name} has no {what} setting"
    if spec.backend == Backend.ULTRALYTICS:
        if field in ("ema", "ema_decay"):
            return False, f"{_ULTRA} and keeps its own EMA of the weights"
        if field in ("grad_clip", "layer_decay", "freeze_backbone_epochs", "label_smoothing",
                     "nesterov", "step_size", "step_gamma", "compile_model"):
            return False, f"{_ULTRA}; this setting is not passed to it"
    if field == "layer_decay" and spec.backend != Backend.TIMM:
        return False, "Layer-wise decay uses timm's parameter grouping (timm models only)"
    return True, ""
