"""The contract between the UI and the training process.

RunConfig is written to disk as `<run_dir>/config.json` and read by runner.py as
its single input. Everything in this file must be JSON-serialisable: the same
training run must be reproducible without the UI (`python runner.py --config ...`).
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────


class Task(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    SEGMENTATION3D = "segmentation3d"

    @property
    def label(self) -> str:
        return {
            "classification": "Classification",
            "segmentation": "Segmentation (2D)",
            "segmentation3d": "Segmentation (3D)",
        }[self.value]


class Modality(str, Enum):
    RGB = "rgb"
    GRAYSCALE = "grayscale"
    CT = "ct"
    MR = "mr"

    @property
    def label(self) -> str:
        return {"rgb": "RGB image", "grayscale": "Grayscale",
                "ct": "CT", "mr": "MR"}[self.value]

    @property
    def is_medical(self) -> bool:
        return self in (Modality.CT, Modality.MR)


class Backend(str, Enum):
    TIMM = "timm"
    TORCHVISION = "torchvision"
    SMP = "smp"
    HF = "hf"
    ULTRALYTICS = "ultralytics"
    MONAI = "monai"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

    @property
    def label(self) -> str:
        return {"queued": "Queued", "running": "Running", "completed": "Completed",
                "failed": "Failed", "stopped": "Stopped"}[self.value]

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.STOPPED)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DICOM_EXTS = {".dcm", ".dicom", ".ima", ""}          # DICOM files may have no extension
VOLUME_EXTS = {".nii", ".nii.gz", ".mha", ".mhd", ".nrrd"}

# Metric direction: True → higher is better
METRIC_HIGHER_IS_BETTER = {
    "loss": False, "val_loss": False, "train_loss": False,
    "accuracy": True, "balanced_accuracy": True, "f1_macro": True,
    "f1_weighted": True, "precision_macro": True, "recall_macro": True,
    "auroc": True, "auprc": True, "kappa": True, "ece": False,
    "dice": True, "iou": True, "dice_macro": True, "iou_macro": True,
    "boundary_f1": True, "hd95": False, "assd": False,
    "map50": True, "map50_95": True,
}

DEFAULT_MONITOR = {
    Task.CLASSIFICATION: "balanced_accuracy",
    Task.SEGMENTATION: "dice_macro",
    Task.SEGMENTATION3D: "dice_macro",
}


def metric_mode(name: str) -> Literal["max", "min"]:
    return "max" if METRIC_HIGHER_IS_BETTER.get(name, True) else "min"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class WindowSpec(BaseModel):
    """CT/DICOM windowing (Hounsfield units)."""
    model_config = ConfigDict(extra="forbid")
    center: float = 40.0
    width: float = 400.0

    @property
    def bounds(self) -> tuple[float, float]:
        return self.center - self.width / 2, self.center + self.width / 2


class DatasetConfig(BaseModel):
    """Machine-specific description of a validated dataset."""
    model_config = ConfigDict(extra="forbid")

    root: str
    task: Task
    modality: Modality = Modality.RGB
    classes: list[str] = Field(default_factory=list)
    ignore_index: int = 255
    window: WindowSpec | None = None

    # Counts filled in by scan.py — the recommendation engine consumes these
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    class_counts: dict[str, int] = Field(default_factory=dict)
    median_image_size: tuple[int, int] | None = None
    mask_foreground_ratio: float | None = None   # sparsity measure for segmentation
    channels: int = 3

    # Produced by splitter.py when there is no split beyond train/
    splits_file: str | None = None

    @field_validator("root")
    @classmethod
    def _abs_root(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def imbalance_ratio(self) -> float:
        """Largest / smallest class ratio. Drives loss selection in classification."""
        counts = [c for c in self.class_counts.values() if c > 0]
        if len(counts) < 2:
            return 1.0
        return max(counts) / min(counts)

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_val + self.n_test


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────


class AugConfig(BaseModel):
    """Modality-aware augmentation. Probability fields are in [0, 1]."""
    model_config = ConfigDict(extra="forbid")

    preset: Literal["none", "light", "medium", "heavy", "custom"] = "medium"

    hflip: float = 0.5
    vflip: float = 0.0
    rot90: float = 0.0
    rotate_limit: int = 15          # degrees
    scale_limit: float = 0.10
    shift_limit: float = 0.0625
    affine_p: float = 0.5

    brightness_contrast: float = 0.2   # magnitude
    brightness_p: float = 0.5
    gamma_p: float = 0.0
    blur_p: float = 0.0
    noise_p: float = 0.0
    sharpen_p: float = 0.0

    elastic_p: float = 0.0
    grid_distortion_p: float = 0.0
    coarse_dropout_p: float = 0.0

    # Classification only
    mixup: float = 0.0
    cutmix: float = 0.0
    randaugment: int = 0            # 0 = off, otherwise magnitude

    # CT/MR only — intensity jitter instead of colour jitter
    hu_shift: float = 0.0           # ± in HU
    hu_scale: float = 0.0           # ± as a fraction

    normalize: Literal["imagenet", "dataset", "minmax", "none"] = "imagenet"


AUG_PRESETS: dict[str, dict[str, Any]] = {
    "none": dict(preset="none", hflip=0, vflip=0, rot90=0, affine_p=0,
                 brightness_p=0, brightness_contrast=0),
    "light": dict(preset="light", hflip=0.5, affine_p=0.3, rotate_limit=10,
                  scale_limit=0.05, shift_limit=0.03, brightness_p=0.3,
                  brightness_contrast=0.1),
    "medium": dict(preset="medium", hflip=0.5, affine_p=0.5, rotate_limit=15,
                   scale_limit=0.10, shift_limit=0.0625, brightness_p=0.5,
                   brightness_contrast=0.2, blur_p=0.1, noise_p=0.1),
    "heavy": dict(preset="heavy", hflip=0.5, vflip=0.2, rot90=0.2, affine_p=0.7,
                  rotate_limit=30, scale_limit=0.2, shift_limit=0.1,
                  brightness_p=0.7, brightness_contrast=0.3, gamma_p=0.3,
                  blur_p=0.2, noise_p=0.2, elastic_p=0.2, grid_distortion_p=0.2,
                  coarse_dropout_p=0.2),
}


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

Prob = Annotated[float, Field(ge=0.0, le=1.0)]


class Hyperparams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ── Basics ───────────────────────────────────────────────────────────
    epochs: int = Field(50, ge=1, le=10_000)
    batch_size: int = Field(16, ge=1, le=4096)
    img_size: int = Field(224, ge=32, le=4096)
    accumulate_grad_batches: int = Field(1, ge=1, le=256)

    # ── Optimisation ─────────────────────────────────────────────────────
    optimizer: Literal["adamw", "adam", "sgd", "rmsprop"] = "adamw"
    lr: float = Field(3e-4, gt=0, le=10)
    weight_decay: float = Field(1e-4, ge=0, le=1)
    momentum: float = Field(0.9, ge=0, le=1)          # sgd/rmsprop only
    nesterov: bool = True
    beta1: float = Field(0.9, ge=0, lt=1)
    beta2: float = Field(0.999, ge=0, lt=1)
    layer_decay: float | None = Field(None, gt=0, le=1)   # ViT full fine-tune

    scheduler: Literal["cosine", "step", "plateau", "onecycle", "poly", "none"] = "cosine"
    warmup_epochs: float = Field(3.0, ge=0)
    min_lr: float = Field(1e-6, ge=0)
    step_size: int = 30           # scheduler="step"
    step_gamma: float = 0.1

    # ── Loss ─────────────────────────────────────────────────────────────
    loss: str = "ce"
    class_weights: Literal["none", "balanced"] = "none"
    label_smoothing: float = Field(0.0, ge=0, lt=1)
    focal_gamma: float = 2.0
    tversky_alpha: float = 0.5
    tversky_beta: float = 0.5
    dice_weight: float = 0.5      # Dice share in combined losses

    # ── Model ────────────────────────────────────────────────────────────
    pretrained: bool = True
    freeze_backbone_epochs: int = Field(0, ge=0)
    drop_rate: float = Field(0.0, ge=0, lt=1)
    drop_path_rate: float = Field(0.0, ge=0, lt=1)
    encoder: str | None = None    # timm encoder name for smp architectures

    # ── Regularisation / stability ───────────────────────────────────────
    grad_clip: float | None = Field(None, gt=0)
    ema: bool = False
    ema_decay: float = Field(0.9998, gt=0, lt=1)

    # ── Runtime ──────────────────────────────────────────────────────────
    amp: bool = False
    channels_last: bool = False
    num_workers: int = Field(4, ge=0, le=64)
    seed: int = 42
    deterministic: bool = False
    compile_model: bool = False

    # ── Monitoring / stopping ────────────────────────────────────────────
    monitor_metric: str = "auto"
    early_stopping: bool = True
    patience: int = Field(15, ge=1)
    val_interval: int = Field(1, ge=1)
    save_last: bool = True
    log_every_n_steps: int = Field(10, ge=1)
    preview_every_n_epochs: int = Field(1, ge=0)   # 0 = no previews

    @property
    def monitor_mode(self) -> Literal["max", "min"]:
        return metric_mode(self.monitor_metric)


# ─────────────────────────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────────────────────────


class ModelSelection(BaseModel):
    """A model picked from the registry plus the user's settings."""
    model_config = ConfigDict(extra="forbid")

    spec_id: str                    # MODEL_REGISTRY key
    backend: Backend
    arch: str                       # the real architecture name passed to the backend
    display_name: str = ""
    weights: str | None = None      # timm pretrained tag / path to a .pt file
    encoder: str | None = None      # smp
    encoder_weights: str | None = "imagenet"


# ─────────────────────────────────────────────────────────────────────────────
# Run configuration — the root object written to disk
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_VERSION = 1


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = CONFIG_VERSION
    run_name: str
    output_dir: str                 # root directory chosen by the user
    notes: str = ""
    created_at: str = ""

    dataset: DatasetConfig
    model: ModelSelection
    hp: Hyperparams
    aug: AugConfig = Field(default_factory=AugConfig)

    device: str = "auto"            # "auto" | "cuda:0" | "mps" | "cpu"
    export_onnx: bool = False
    export_torchscript: bool = False

    @field_validator("output_dir")
    @classmethod
    def _abs_out(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())

    @field_validator("run_name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("run_name cannot be empty")
        bad = set('/\\:*?"<>|')
        if any(ch in bad for ch in v):
            raise ValueError(f"run_name may not contain these characters: {''.join(sorted(bad))}")
        return v

    @model_validator(mode="after")
    def _resolve_monitor(self) -> "RunConfig":
        if self.hp.monitor_metric == "auto":
            self.hp.monitor_metric = DEFAULT_MONITOR[self.dataset.task]
        return self

    # ── Directory layout (single source of truth) ────────────────────────
    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    # ── Serialisation ────────────────────────────────────────────────────
    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.path("config.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


# Fixed file names inside a run directory — read by both the UI and the runner.
class Layout:
    CONFIG = "config.json"
    ENV = "env.json"
    STATE = "state.json"
    EVENTS = "events.jsonl"
    LOG = "train.log"
    SPLITS = "splits.json"
    STOP = "STOP"
    CHECKPOINTS = "checkpoints"
    BEST = "checkpoints/best.pt"
    LAST = "checkpoints/last.pt"
    METRICS_DIR = "metrics"
    METRICS_CSV = "metrics/metrics.csv"
    METRICS_XLSX = "metrics/metrics.xlsx"
    SUMMARY = "metrics/summary.json"
    PER_CLASS = "metrics/per_class.csv"
    PLOTS = "plots"
    PREVIEWS = "previews"
    PREDICTIONS = "predictions"
    EXPORTS = "exports"
    REPORT = "report.html"
