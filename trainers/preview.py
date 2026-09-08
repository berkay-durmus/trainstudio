"""Validation previews produced during training.

Numeric metrics tell you *how* good a model is; previews show you *where* it goes
wrong. Mask overlap in segmentation, the most confidently misclassified samples in
classification — both are the fastest signal for cutting a run short and changing
a setting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OVERLAY_COLORS = np.array([
    [0, 0, 0], [45, 212, 191], [96, 165, 250], [244, 114, 182], [251, 191, 36],
    [167, 139, 250], [52, 211, 153], [251, 146, 60], [248, 113, 113], [34, 211, 238],
], dtype=np.float32)


def denormalize(tensor, mean=None, std=None) -> np.ndarray:
    """Turn a normalised (C,H,W) tensor into a displayable uint8 HWC array."""
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if mean is not None and std is not None:
        m = np.asarray(mean, dtype=np.float32)[: arr.shape[-1]]
        s = np.asarray(std, dtype=np.float32)[: arr.shape[-1]]
        arr = arr * s + m
    else:
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo + 1e-8)
    arr = np.clip(arr, 0, 1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr * 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Convert a class-index mask to RGB."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in np.unique(mask):
        if c == ignore_index or c == 0:
            continue
        out[mask == c] = OVERLAY_COLORS[int(c) % len(OVERLAY_COLORS)].astype(np.uint8)
    return out


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45,
                 ignore_index: int = 255) -> np.ndarray:
    out = image.astype(np.float32).copy()
    for c in np.unique(mask):
        if c == 0 or c == ignore_index:
            continue
        m = mask == c
        color = OVERLAY_COLORS[int(c) % len(OVERLAY_COLORS)]
        out[m] = (1 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def contour_of(mask: np.ndarray) -> np.ndarray:
    """Return the mask's boundary pixels as a boolean array."""
    try:
        from scipy import ndimage

        b = mask > 0
        return b ^ ndimage.binary_erosion(b, iterations=1, border_value=0)
    except Exception:
        return np.zeros_like(mask, dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# Grid builders
# ─────────────────────────────────────────────────────────────────────────────


def segmentation_grid(
    images: list[np.ndarray],
    truths: list[np.ndarray],
    preds: list[np.ndarray],
    dices: list[float],
    out_path: Path,
    title: str = "",
    ignore_index: int = 255,
) -> Path:
    """One sample per row: image · ground truth · prediction · overlay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    if n == 0:
        raise ValueError("There are no samples to preview")

    fig, axes = plt.subplots(n, 4, figsize=(11, 2.8 * n), facecolor="#0E1117")
    if n == 1:
        axes = axes[None, :]

    headers = ["Image", "Ground truth", "Prediction", "Comparison"]
    for r in range(n):
        img, gt, pr = images[r], truths[r], preds[r]
        # 4th column: the ground-truth boundary in green, the predicted one in teal
        comp = img.astype(np.float32).copy()
        comp[contour_of(gt)] = [34, 197, 94]
        comp[contour_of(pr)] = [45, 212, 191]
        panels = [img, overlay_mask(img, gt, 0.55, ignore_index),
                  overlay_mask(img, pr, 0.55, ignore_index),
                  np.clip(comp, 0, 255).astype(np.uint8)]
        for c in range(4):
            ax = axes[r, c]
            ax.imshow(panels[c])
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#2A333F")
            if r == 0:
                ax.set_title(headers[c], color="#8B98A5", fontsize=10, pad=6)
        axes[r, 0].set_ylabel(f"Dice {dices[r]:.3f}", color="#E6EDF3", fontsize=9)

    if title:
        fig.suptitle(title, color="#E6EDF3", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor="#0E1117", bbox_inches="tight")
    plt.close(fig)
    return out_path


def classification_grid(
    images: list[np.ndarray],
    truths: list[str],
    preds: list[str],
    confidences: list[float],
    out_path: Path,
    title: str = "",
    cols: int = 6,
) -> Path:
    """Misclassified samples — a red frame, with truth→prediction underneath."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    if n == 0:
        raise ValueError("There are no samples to preview")
    cols = min(cols, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.55 * rows),
                             facecolor="#0E1117", squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.set_xticks([]), ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(images[i])
        correct = truths[i] == preds[i]
        color = "#22C55E" if correct else "#EF4444"
        for sp in ax.spines.values():
            sp.set_color(color)
            sp.set_linewidth(2.2)
        ax.set_xlabel(f"{truths[i]} → {preds[i]}\n{confidences[i]*100:.0f}% confidence",
                      color=color, fontsize=8)

    if title:
        fig.suptitle(title, color="#E6EDF3", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor="#0E1117", bbox_inches="tight")
    plt.close(fig)
    return out_path
