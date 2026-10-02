"""The hyperparameter recommendation engine, with reasons.

Rule based and deterministic: the same (model, dataset, hardware) triple always
produces the same recommendation. Every field also gets a one-sentence rationale —
the UI shows it as a caption under the field, so the user can see where the value
came from and deviate deliberately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.capabilities import spec_knobs
from core.hardware import DeviceInfo, detect, recommended_workers, usable_vram_gb
from core.registry import DEFAULT_ENCODER, ModelSpec
from core.schemas import (
    AUG_PRESETS,
    DEFAULT_MONITOR,
    AugConfig,
    Backend,
    DatasetConfig,
    Hyperparams,
    Modality,
    Task,
)


@dataclass
class Recommendation:
    hp: Hyperparams
    aug: AugConfig
    why: dict[str, str]                    # field name → rationale
    warnings: list[str]                    # warnings to surface to the user

    def reason(self, field: str) -> str:
        return self.why.get(field, "")


# ─────────────────────────────────────────────────────────────────────────────
# Memory estimation
# ─────────────────────────────────────────────────────────────────────────────


def estimate_batch_size(
    spec: ModelSpec, img_size: int, budget_gb: float, amp: bool = False
) -> int:
    """Estimate the batch size that fits in the given memory budget.

    The per-sample activation cost is a linear model calibrated against measured
    reference points (fp32, no gradient checkpointing):

        ResNet-50   @224 ≈ 0.12 GB/sample     ViT-B/16    @224 ≈ 0.29 GB/sample
        U-Net(R50)  @512 ≈ 1.2  GB/sample     SwinUNETR   @96³ ≈ 5.8  GB/sample

    The goal is not precision but a safe starting point that does not OOM; the
    user can always change the value by hand.
    """
    # AdamW: weights + gradients + 2 moments ≈ 16 bytes/parameter
    static_gb = spec.params_m * 1e6 * 16 / 1024**3
    free_gb = max(0.4, budget_gb - static_gb) * 0.90        # 10% safety margin

    if spec.task == Task.SEGMENTATION3D:
        per_sample = (img_size / 96) ** 3 * (1.5 + spec.params_m * 0.07)
    elif spec.task == Task.SEGMENTATION:
        # Full-resolution decoder activations dominate
        per_sample = (img_size / 224) ** 2 * (0.10 + spec.params_m * 0.0035)
    else:
        per_sample = (img_size / 224) ** 2 * (0.045 + spec.params_m * 0.0028)

    if amp:
        per_sample *= 0.6      # mixed precision cuts activations by roughly 40%

    raw = free_gb / max(per_sample, 1e-4)
    # Round down to a power of two
    bs = 2 ** int(math.floor(math.log2(max(1.0, raw))))
    lo = 1 if spec.task == Task.SEGMENTATION3D else 2
    hi = 32 if spec.task == Task.SEGMENTATION3D else 64
    return int(max(lo, min(bs, hi)))


# ─────────────────────────────────────────────────────────────────────────────
# The main recommendation
# ─────────────────────────────────────────────────────────────────────────────


def recommend(
    spec: ModelSpec,
    ds: DatasetConfig,
    device: DeviceInfo | None = None,
) -> Recommendation:
    device = device or detect()
    why: dict[str, str] = {}
    warns: list[str] = []

    # Start from the model's own recommendations
    hp = Hyperparams(**{k: v for k, v in spec.rec.items() if k in Hyperparams.model_fields})

    # ── Image size ───────────────────────────────────────────────────────
    img = int(spec.rec.get("img_size", spec.default_img_size))
    if ds.median_image_size and ds.task != Task.SEGMENTATION3D:
        h, w = ds.median_image_size
        native = max(h, w)
        if native < img * 0.6:
            # Upscaling small images adds cost without adding information
            img = max(64, 2 ** int(round(math.log2(native))))
            why["img_size"] = (
                f"The dataset's median size is {h}×{w}; using {img}px instead of upscaling "
                f"to the model default ({spec.default_img_size}px) cuts cost without losing information."
            )
    why.setdefault(
        "img_size",
        f"{spec.display_name} was pretrained at this resolution; the weights fit best here.",
    )
    hp.img_size = img

    # ── Batch size ───────────────────────────────────────────────────────
    # AMP is decided first because it affects the batch-size estimate.
    hp.amp = device.supports_amp
    budget = usable_vram_gb(device)
    bs = estimate_batch_size(spec, img, budget, amp=hp.amp)
    bs = min(bs, max(2, ds.n_train)) if ds.n_train else bs
    hp.batch_size = bs
    dev_label = {"cuda": "VRAM", "mps": "unified memory", "cpu": "RAM"}[device.kind]
    why["batch_size"] = (
        f"Estimated upper bound for {budget:.0f} GB of usable {dev_label} at {img}px input. "
        f"If you hit an OOM, halve it and raise gradient accumulation."
    )
    if spec.min_vram_gb > budget:
        warns.append(
            f"⚠️ {spec.display_name} recommends around {spec.min_vram_gb:.0f} GB of memory; "
            f"{budget:.0f} GB is available on this machine. Training may be very slow or OOM."
        )

    # Very small batches break BatchNorm statistics → suggest accumulation
    if bs < 8 and spec.task != Task.SEGMENTATION3D:
        hp.accumulate_grad_batches = max(1, 16 // max(bs, 1))
        why["accumulate_grad_batches"] = (
            f"The effective batch becomes {bs}×{hp.accumulate_grad_batches}={bs * hp.accumulate_grad_batches}; "
            "this stabilises normalisation statistics at small batch sizes."
        )

    # ── Optimizer / learning rate ────────────────────────────────────────
    why["optimizer"] = {
        "adamw": "Transformers and modern convolutional networks converge more stably with AdamW, which decouples weight decay.",
        "sgd": "This classic ResNet/DenseNet family behaves closest to the published results with momentum SGD.",
        "adam": "Adam is sufficient for this architecture.",
        "rmsprop": "RMSProp matches the original training recipe.",
    }[hp.optimizer]

    # If the batch size deviates from the default, scale the lr (linear scaling rule)
    ref_bs = int(spec.rec.get("batch_size", 16))
    if ref_bs and bs != ref_bs:
        scale = bs / ref_bs
        # Square-root scaling is safer for adaptive optimizers
        factor = scale if hp.optimizer == "sgd" else math.sqrt(scale)
        # The scaling rule breaks down at the extremes; cap it at 4×.
        factor = min(4.0, max(0.25, factor))
        hp.lr = float(f"{hp.lr * factor:.2e}")
        why["lr"] = (
            f"The base value for {spec.display_name} is {spec.rec.get('lr', hp.lr):.1e} (batch {ref_bs}); "
            f"scaled {'linearly' if hp.optimizer == 'sgd' else 'by square root'} for batch {bs}."
        )
    else:
        why["lr"] = f"The starting value commonly used in the literature for the {spec.display_name} family."

    if hp.layer_decay:
        why["layer_decay"] = (
            "Layer-wise learning-rate decay: lower layers update more slowly and the pretrained "
            "representation is preserved. The key to fine-tuning large ViTs."
        )

    # ── Number of epochs ─────────────────────────────────────────────────
    n = ds.n_train or 1000
    if n < 500:
        epochs = 120
        note = "a very small dataset — it needs more passes"
    elif n < 1_000:
        epochs = 100
        note = "a small dataset"
    elif n < 10_000:
        epochs = 50
        note = "a mid-sized dataset"
    elif n < 50_000:
        epochs = 30
        note = "a large dataset"
    else:
        epochs = 20
        note = "a very large dataset — many steps per epoch"
    hp.epochs = epochs
    why["epochs"] = (f"{n:,} training samples ({note}). Early stopping is on, so no epochs are wasted.")

    hp.warmup_epochs = float(max(2, round(epochs * 0.05)))
    why["warmup_epochs"] = (
        "About 5% of the total epochs. Warmup keeps the pretrained weights from being "
        "damaged during the first steps."
    )
    why["scheduler"] = "Cosine decay works well almost every time and needs no extra tuning."
    hp.min_lr = max(1e-7, hp.lr / 100)

    # ── Loss function ────────────────────────────────────────────────────
    if ds.task == Task.CLASSIFICATION:
        imb = ds.imbalance_ratio
        if imb >= 10:
            hp.loss, hp.class_weights = "focal", "balanced"
            why["loss"] = (
                f"Class imbalance is {imb:.0f}× — the focal loss suppresses the contribution of easy "
                "samples and, together with class weights, rescues the minority class."
            )
        elif imb >= 3:
            hp.loss, hp.class_weights = "ce", "balanced"
            why["loss"] = (
                f"Class imbalance is {imb:.1f}× — cross entropy is balanced with class weights."
            )
        else:
            hp.loss, hp.class_weights = "ce", "none"
            why["loss"] = f"The classes are balanced ({imb:.1f}×); plain cross entropy is enough."
        if ds.num_classes == 2:
            why["loss"] += " On a two-class problem, remember to watch AUROC as well."
    else:
        fg = ds.mask_foreground_ratio
        if fg is not None and fg < 0.02:
            hp.loss = "dice_focal"
            why["loss"] = (
                f"Foreground pixels are only {fg * 100:.2f}% of the total — very sparse masks. "
                "Dice combined with focal breaks the dominance of the background."
            )
        elif fg is not None and fg < 0.10:
            hp.loss = "dice_ce"
            hp.dice_weight = 0.7
            why["loss"] = (
                f"The foreground ratio is {fg * 100:.1f}% — sparse. The Dice weight was raised to 0.7."
            )
        else:
            hp.loss = "dice_ce"
            why["loss"] = "Dice + cross entropy is the most dependable default combination for segmentation."

    if spec.backend in (Backend.HF, Backend.ULTRALYTICS):
        hp.loss = "native"
        why["loss"] = (
            f"{spec.family} uses its own internal loss function; "
            "loss selection is disabled for this backend."
        )

    # ── Label smoothing / gradient clipping ──────────────────────────────
    if ds.task == Task.CLASSIFICATION and spec.params_m >= 50:
        hp.label_smoothing = 0.1
        why["label_smoothing"] = (
            "Large models tend to become overconfident about the training labels; "
            "0.1 smoothing improves calibration."
        )
    if "vit" in spec.arch.lower() or spec.family in ("DINOv3", "EVA-02", "Swin", "MaxViT", "Hiera"):
        hp.grad_clip = 1.0
        why["grad_clip"] = "The standard safeguard against exploding gradients when training transformers."

    # ── Freezing ─────────────────────────────────────────────────────────
    if n < 2_000 and spec.params_m > 50 and hp.pretrained:
        hp.freeze_backbone_epochs = max(2, int(epochs * 0.05))
        why["freeze_backbone_epochs"] = (
            f"{n:,} samples is little for {spec.params_label} parameters. For the first "
            f"{hp.freeze_backbone_epochs} epochs only the head is trained, so the randomly "
            "initialised head cannot damage the backbone."
        )

    # ── Regularisation ───────────────────────────────────────────────────
    # Start every knob from the library's own default for this architecture, so
    # an untouched run trains the model it always did; the catalogue's
    # stochastic-depth presets only ever reached timm, and still do.
    knobs = spec_knobs(spec)
    for field in ("drop_rate", "drop_path_rate"):
        if field not in knobs:
            setattr(hp, field, 0.0)
        elif spec.backend != Backend.TIMM or field not in spec.rec:
            setattr(hp, field, knobs[field].default)
    small = n < 5_000
    if "drop_rate" in knobs:
        why["drop_rate"] = (
            f"{spec.display_name}'s own default ({knobs['drop_rate'].default:g}). "
            + ("With few samples, 0.2–0.3 before the classifier is a cheap guard "
               "against overfitting." if small else
               "Raise it if the validation loss climbs while the training loss keeps falling.")
        )
    if "drop_path_rate" in knobs:
        why["drop_path_rate"] = (
            ("The catalogue's value for this family: " if spec.backend == Backend.TIMM
             and "drop_path_rate" in spec.rec else f"{spec.display_name}'s own default: ")
            + f"{hp.drop_path_rate:g}. Stochastic depth randomly skips residual blocks; "
            "deeper and larger models usually take 0.1–0.3."
        )
    total_steps = max(1, n // max(1, hp.batch_size)) * epochs
    hp.ema = False
    why["ema"] = ("An exponential moving average of the weights is evaluated instead of the "
                  "raw ones. It usually adds a little accuracy and smooths noisy validation "
                  "curves, at the cost of a second copy of the model in memory.")
    why["ema_decay"] = (
        "How slowly the average follows the weights. Keep 1 / (1 − decay) well below the "
        f"total number of steps (about {total_steps:,} here): 0.999 for short "
        "runs, 0.9998–0.9999 for long ones."
    )

    # ── Runtime ──────────────────────────────────────────────────────────
    why["amp"] = (
        "On CUDA, mixed precision halves memory use and speeds training up."
        if device.supports_amp
        else f"Mixed precision is not reliable on {device.kind.upper()}; it was left off."
    )
    hp.channels_last = device.supports_channels_last and spec.backend in (
        Backend.TIMM, Backend.TORCHVISION, Backend.SMP)
    hp.num_workers = recommended_workers()
    why["num_workers"] = f"A sensible amount of data-loading parallelism for {device.name}."

    # ── Early stopping ───────────────────────────────────────────────────
    hp.patience = max(10, int(epochs * 0.20))
    why["patience"] = (
        f"20% of the total epochs. Training stops if the monitored metric does not improve "
        f"for {hp.patience} epochs."
    )
    hp.early_stopping = True
    # Name the metric outright: the Settings page offers concrete metrics only,
    # and "auto" there fell through to the first in the list — accuracy, even on
    # an imbalanced dataset, where it is the metric to distrust.
    hp.monitor_metric = DEFAULT_MONITOR[ds.task]
    why["monitor_metric"] = (
        "Balanced accuracy weighs every class equally, so a model cannot score well by "
        "favouring the largest class; it picks the best checkpoint and drives early stopping."
        if ds.task == Task.CLASSIFICATION else
        "Mean Dice over the foreground classes: it tracks overlap with the masks rather than "
        "pixel accuracy, which an all-background prediction already scores well on."
    )

    # ── Encoder (smp) ────────────────────────────────────────────────────
    if spec.needs_encoder:
        hp.encoder = spec.rec.get("encoder") or DEFAULT_ENCODER
        why["encoder"] = (
            "ResNet-50 is a safe default. Try ConvNeXt V2 for higher accuracy, "
            "or ResNet-34 for less memory."
        )

    # ── Augmentation ─────────────────────────────────────────────────────
    aug, aug_why = _recommend_aug(spec, ds, n)
    why.update(aug_why)

    # ── General warnings ─────────────────────────────────────────────────
    if ds.n_val == 0:
        warns.append("⚠️ There is no validation set. Early stopping and best-model selection will not work.")
    if ds.task == Task.CLASSIFICATION and ds.imbalance_ratio >= 10:
        warns.append(
            f"⚠️ Class imbalance is {ds.imbalance_ratio:.0f}×. Accuracy will be misleading — "
            "watch balanced accuracy and macro F1 instead."
        )
    if device.kind == "cpu":
        warns.append("⚠️ No GPU was found. Training on CPU will be very slow; pick a small model.")
    elif device.kind == "mps":
        warns.append(
            "ℹ️ Apple GPU (MPS) is in use. Some operations may fall back to the CPU and mixed "
            "precision is disabled; it is normal for this to be slower than CUDA."
        )

    return Recommendation(hp=hp, aug=aug, why=why, warnings=warns)


def _recommend_aug(spec: ModelSpec, ds: DatasetConfig, n: int) -> tuple[AugConfig, dict[str, str]]:
    """Augmentation that is sensitive to modality and dataset size."""
    why: dict[str, str] = {}

    if n < 1_000:
        preset, note = "heavy", "the dataset is small, so strong augmentation delays overfitting"
    elif n < 20_000:
        preset, note = "medium", "a balanced setting for a mid-sized dataset"
    else:
        preset, note = "light", "the dataset is already varied, and heavy augmentation slows convergence"

    aug = AugConfig(**AUG_PRESETS[preset])
    why["aug_preset"] = f"{n:,} training samples — {note}."

    if ds.modality.is_medical:
        # In medical imaging, anatomical orientation is meaningful and pixel values are physical units
        aug.vflip = 0.0
        aug.rot90 = 0.0
        aug.brightness_p = 0.0
        aug.brightness_contrast = 0.0
        aug.gamma_p = 0.0
        aug.rotate_limit = min(aug.rotate_limit, 15)
        aug.hu_shift = 20.0 if ds.modality == Modality.CT else 0.0
        aug.hu_scale = 0.05
        aug.normalize = "dataset"
        why["aug_modality"] = (
            f"{ds.modality.label}: vertical flips and colour/brightness jitter were turned off "
            "(anatomical orientation is meaningful and pixel values are physical units). "
            "Intensity (HU) jitter and dataset-derived normalisation are used instead."
        )
    else:
        why["aug_modality"] = "Natural images: horizontal flips and photometric jitter are safe."

    if ds.task == Task.CLASSIFICATION and n >= 5_000 and spec.params_m >= 50:
        aug.mixup = 0.2
        aug.cutmix = 0.5 if not ds.modality.is_medical else 0.0
        why["mixup"] = (
            "Large model plus enough data: MixUp/CutMix provide regularisation."
            + (" CutMix is off because it breaks anatomy in medical images." if ds.modality.is_medical else "")
        )

    if ds.task in (Task.SEGMENTATION, Task.SEGMENTATION3D):
        aug.mixup = aug.cutmix = 0.0
        aug.coarse_dropout_p = 0.0   # risks becoming inconsistent with the mask
        if not ds.modality.is_medical:
            aug.elastic_p = min(aug.elastic_p, 0.2)
        else:
            aug.elastic_p = 0.2
            aug.grid_distortion_p = 0.1
            why["aug_elastic"] = (
                "Elastic deformation is the most effective augmentation in medical segmentation "
                "because it mimics anatomical variation (as recommended by the original U-Net paper)."
            )

    aug.preset = "custom" if ds.modality.is_medical else preset
    return aug, why
