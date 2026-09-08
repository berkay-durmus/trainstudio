"""Loss functions.

In segmentation, Dice-based losses are robust to class imbalance but produce a
noisy gradient on their own; in practice they are combined with cross entropy.
The recommendation engine picks the combination based on mask sparsity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """Suppresses the contribution of easy samples by (1-p)^γ; effective under heavy imbalance."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, target, weight=w,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────────────────────────────────────


def _one_hot(target: torch.Tensor, n_classes: int, ignore_index: int
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,H,W[,D]) → (B,K,H,W[,D]) one-hot plus a validity mask."""
    valid = target != ignore_index
    safe = torch.where(valid, target, torch.zeros_like(target))
    oh = F.one_hot(safe.long(), n_classes)
    # (B,H,W,K) → (B,K,H,W) — moves the last axis to the front in 3D as well
    dims = [0, oh.ndim - 1] + list(range(1, oh.ndim - 1))
    return oh.permute(*dims).float(), valid.unsqueeze(1).float()


class DiceLoss(nn.Module):
    """Soft Dice. `include_background=False` removes the background from the loss —
    which keeps the gradient focused on the foreground when the background dominates."""

    def __init__(self, n_classes: int, ignore_index: int = 255,
                 include_background: bool = False, smooth: float = 1.0):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.include_background = include_background or n_classes < 2
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        oh, valid = _one_hot(target, self.n_classes, self.ignore_index)
        probs, oh = probs * valid, oh * valid

        reduce_dims = tuple(range(2, probs.ndim))          # H,W[,D]
        inter = (probs * oh).sum(dim=reduce_dims)
        denom = probs.sum(dim=reduce_dims) + oh.sum(dim=reduce_dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)

        if not self.include_background:
            dice = dice[:, 1:]
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    """A generalisation of Dice: β > α penalises false negatives (missed lesions) more heavily."""

    def __init__(self, n_classes: int, alpha: float = 0.5, beta: float = 0.5,
                 ignore_index: int = 255, include_background: bool = False,
                 smooth: float = 1.0):
        super().__init__()
        self.n_classes = n_classes
        self.alpha, self.beta = alpha, beta
        self.ignore_index = ignore_index
        self.include_background = include_background or n_classes < 2
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        oh, valid = _one_hot(target, self.n_classes, self.ignore_index)
        probs, oh = probs * valid, oh * valid

        dims = tuple(range(2, probs.ndim))
        tp = (probs * oh).sum(dim=dims)
        fp = (probs * (1 - oh)).sum(dim=dims)
        fn = ((1 - probs) * oh).sum(dim=dims)
        tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if not self.include_background:
            tv = tv[:, 1:]
        return 1.0 - tv.mean()


class SegFocalLoss(nn.Module):
    """The focal loss applied per pixel."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 ignore_index: int = 255):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, target.long(), weight=w,
                             ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        valid = target != self.ignore_index
        return loss[valid].mean() if valid.any() else loss.mean()


class CombinedLoss(nn.Module):
    """w·region + (1−w)·pixel — the most dependable default in segmentation."""

    def __init__(self, region: nn.Module, pixel: nn.Module, region_weight: float = 0.5):
        super().__init__()
        self.region = region
        self.pixel = pixel
        self.w = region_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.w * self.region(logits, target) + (1 - self.w) * self.pixel(logits, target)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def build_loss(task, hp, n_classes: int, ignore_index: int = 255,
               weight: torch.Tensor | None = None) -> nn.Module:
    """Turn the `hp.loss` string from RunConfig into a real loss object."""
    from core.schemas import Task

    name = (hp.loss or "").lower()

    if task == Task.CLASSIFICATION:
        if name == "focal":
            return FocalLoss(hp.focal_gamma, weight, hp.label_smoothing)
        if name == "bce":
            return nn.BCEWithLogitsLoss(pos_weight=weight)
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=hp.label_smoothing)

    # ── segmentation ─────────────────────────────────────────────────────
    ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                             label_smoothing=hp.label_smoothing)
    focal = SegFocalLoss(hp.focal_gamma, weight, ignore_index)
    dice = DiceLoss(n_classes, ignore_index)
    tversky = TverskyLoss(n_classes, hp.tversky_alpha, hp.tversky_beta, ignore_index)

    if name == "dice":
        return dice
    if name == "ce":
        return ce
    if name == "focal":
        return focal
    if name == "tversky":
        return tversky
    if name == "dice_focal":
        return CombinedLoss(dice, focal, hp.dice_weight)
    return CombinedLoss(dice, ce, hp.dice_weight)          # default: dice_ce


LOSS_LABELS = {
    "ce": "Cross entropy",
    "focal": "Focal",
    "bce": "Binary cross entropy",
    "dice": "Dice",
    "dice_ce": "Dice + cross entropy",
    "dice_focal": "Dice + focal",
    "tversky": "Tversky",
    "native": "The model's own loss",
}
