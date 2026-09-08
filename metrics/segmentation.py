"""Segmentation metrics (shared by 2D and 3D).

Per-epoch metrics are computed from a K×K confusion matrix: the matrix is
accumulated on the device, the cost per batch is constant, and there is no need
to keep every pixel/voxel prediction in memory. Dice, IoU, precision and recall
are derived exactly from that matrix.

Distance-based metrics (HD95, ASSD) need surface points and are therefore
expensive; they are computed per sample only in the final evaluation.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-7


class SegmentationMetrics:
    """A confusion-matrix accumulator.

    `include_background=False` (the default) leaves class 0 out of the macro
    averages: the background dominates most images, so including it inflates the
    metric artificially.
    """

    def __init__(self, n_classes: int, class_names: list[str] | None = None,
                 ignore_index: int = 255, include_background: bool = False):
        self.n_classes = n_classes
        self.class_names = class_names or [str(i) for i in range(n_classes)]
        self.ignore_index = ignore_index
        self.include_background = include_background or n_classes == 1
        self.reset()

    def reset(self) -> None:
        self.cm = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        self._per_sample_dice: list[float] = []

    # ── accumulation ─────────────────────────────────────────────────────
    def update(self, logits, targets) -> None:
        """logits: (B,K,H,W[,D]) or (B,H,W); targets: (B,H,W[,D])."""
        import torch

        with torch.no_grad():
            if isinstance(logits, torch.Tensor):
                pred = logits.argmax(dim=1) if logits.ndim == targets.ndim + 1 else logits
                pred = pred.detach().reshape(-1)
                tgt = targets.detach().reshape(-1)
                valid = tgt != self.ignore_index
                pred, tgt = pred[valid], tgt[valid]
                # A K×K matrix via bincount — a single kernel call on the GPU
                k = self.n_classes
                idx = (tgt.long() * k + pred.long()).clamp_(0, k * k - 1)
                binc = torch.bincount(idx, minlength=k * k).reshape(k, k)
                self.cm += binc.cpu().numpy().astype(np.int64)
            else:
                self._update_numpy(np.asarray(logits), np.asarray(targets))

    def _update_numpy(self, pred: np.ndarray, tgt: np.ndarray) -> None:
        if pred.ndim == tgt.ndim + 1:
            pred = pred.argmax(axis=1)
        pred, tgt = pred.ravel(), tgt.ravel()
        valid = tgt != self.ignore_index
        pred, tgt = pred[valid], tgt[valid]
        k = self.n_classes
        idx = np.clip(tgt.astype(np.int64) * k + pred.astype(np.int64), 0, k * k - 1)
        self.cm += np.bincount(idx, minlength=k * k).reshape(k, k).astype(np.int64)

    # ── derived quantities ───────────────────────────────────────────────
    @property
    def tp(self) -> np.ndarray:
        return np.diag(self.cm).astype(np.float64)

    @property
    def fp(self) -> np.ndarray:
        return self.cm.sum(axis=0).astype(np.float64) - self.tp

    @property
    def fn(self) -> np.ndarray:
        return self.cm.sum(axis=1).astype(np.float64) - self.tp

    @property
    def support(self) -> np.ndarray:
        return self.cm.sum(axis=1).astype(np.int64)

    @property
    def empty(self) -> bool:
        return self.cm.sum() == 0

    def _macro_indices(self) -> list[int]:
        """The classes that enter the macro average — classes never seen are excluded."""
        start = 0 if self.include_background else 1
        return [i for i in range(start, self.n_classes) if self.support[i] > 0]

    # ── per class ────────────────────────────────────────────────────────
    def dice_per_class(self) -> np.ndarray:
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn + EPS)

    def iou_per_class(self) -> np.ndarray:
        return self.tp / (self.tp + self.fp + self.fn + EPS)

    def precision_per_class(self) -> np.ndarray:
        return self.tp / (self.tp + self.fp + EPS)

    def recall_per_class(self) -> np.ndarray:
        return self.tp / (self.tp + self.fn + EPS)

    # ── summary ──────────────────────────────────────────────────────────
    def compute(self, full: bool = True) -> dict[str, float]:
        if self.empty:
            return {}
        idx = self._macro_indices()
        dice, iou = self.dice_per_class(), self.iou_per_class()
        prec, rec = self.precision_per_class(), self.recall_per_class()

        out = {
            "dice_macro": float(np.mean(dice[idx])) if idx else 0.0,
            "iou_macro": float(np.mean(iou[idx])) if idx else 0.0,
            "precision_macro": float(np.mean(prec[idx])) if idx else 0.0,
            "recall_macro": float(np.mean(rec[idx])) if idx else 0.0,
            "pixel_accuracy": float(self.tp.sum() / (self.cm.sum() + EPS)),
        }
        # On a two-class problem, a single "dice" value is easier to read
        if self.n_classes == 2:
            out["dice"] = float(dice[1])
            out["iou"] = float(iou[1])
        if full and self._per_sample_dice:
            out["dice_per_sample_mean"] = float(np.mean(self._per_sample_dice))
            out["dice_per_sample_std"] = float(np.std(self._per_sample_dice))
        return out

    def per_class(self) -> list[dict]:
        dice, iou = self.dice_per_class(), self.iou_per_class()
        prec, rec = self.precision_per_class(), self.recall_per_class()
        rows = []
        for i, name in enumerate(self.class_names):
            rows.append({
                "class": name,
                "dice": float(dice[i]), "iou": float(iou[i]),
                "precision": float(prec[i]), "recall": float(rec[i]),
                "pixels": int(self.support[i]),
            })
        return rows

    def confusion_matrix(self) -> np.ndarray:
        return self.cm.copy()

    # ── per-sample Dice (for the distribution plot) ──────────────────────
    def add_sample_dice(self, pred, target) -> float:
        """Record the foreground Dice of a single sample (image/volume)."""
        import torch

        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        p, t = pred > 0, target > 0
        inter = np.logical_and(p, t).sum()
        denom = p.sum() + t.sum()
        d = 1.0 if denom == 0 else float(2 * inter / denom)
        self._per_sample_dice.append(d)
        return d

    @property
    def sample_dices(self) -> list[float]:
        return list(self._per_sample_dice)


# ─────────────────────────────────────────────────────────────────────────────
# Distance-based metrics — final evaluation only
# ─────────────────────────────────────────────────────────────────────────────


def surface_distances(pred: np.ndarray, target: np.ndarray,
                      spacing: tuple[float, ...] | None = None) -> np.ndarray | None:
    """Distances from the predicted surface to the true surface (and back), combined."""
    try:
        from scipy import ndimage
    except ImportError:
        return None

    p, t = pred.astype(bool), target.astype(bool)
    if not p.any() or not t.any():
        return None

    def border(mask: np.ndarray) -> np.ndarray:
        eroded = ndimage.binary_erosion(mask, iterations=1, border_value=0)
        return mask ^ eroded

    sp = spacing or (1.0,) * pred.ndim
    dt_t = ndimage.distance_transform_edt(~t, sampling=sp)
    dt_p = ndimage.distance_transform_edt(~p, sampling=sp)
    d_pt = dt_t[border(p)]
    d_tp = dt_p[border(t)]
    if d_pt.size == 0 or d_tp.size == 0:
        return None
    return np.concatenate([d_pt, d_tp])


def hd95(pred: np.ndarray, target: np.ndarray,
         spacing: tuple[float, ...] | None = None) -> float | None:
    """The 95th-percentile Hausdorff distance — a boundary error robust to outliers."""
    d = surface_distances(pred, target, spacing)
    return float(np.percentile(d, 95)) if d is not None else None


def assd(pred: np.ndarray, target: np.ndarray,
         spacing: tuple[float, ...] | None = None) -> float | None:
    """The average symmetric surface distance."""
    d = surface_distances(pred, target, spacing)
    return float(np.mean(d)) if d is not None else None


def boundary_f1(pred: np.ndarray, target: np.ndarray, tolerance: float = 2.0,
                spacing: tuple[float, ...] | None = None) -> float | None:
    """Boundary F1 — boundary agreement within `tolerance` pixels/mm."""
    try:
        from scipy import ndimage
    except ImportError:
        return None

    p, t = pred.astype(bool), target.astype(bool)
    if not p.any() and not t.any():
        return 1.0
    if not p.any() or not t.any():
        return 0.0

    def border(mask):
        return mask ^ ndimage.binary_erosion(mask, iterations=1, border_value=0)

    sp = spacing or (1.0,) * pred.ndim
    bp, bt = border(p), border(t)
    dt_t = ndimage.distance_transform_edt(~bt, sampling=sp)
    dt_p = ndimage.distance_transform_edt(~bp, sampling=sp)
    prec = (dt_t[bp] <= tolerance).mean() if bp.any() else 0.0
    rec = (dt_p[bt] <= tolerance).mean() if bt.any() else 0.0
    return float(2 * prec * rec / (prec + rec + EPS))


def volume_similarity(pred: np.ndarray, target: np.ndarray) -> float:
    """1 − |Vp − Vt| / (Vp + Vt) — how accurate the volume estimate is."""
    vp, vt = float((pred > 0).sum()), float((target > 0).sum())
    if vp + vt == 0:
        return 1.0
    return float(1.0 - abs(vp - vt) / (vp + vt))


def distance_metrics_for_case(pred: np.ndarray, target: np.ndarray, n_classes: int,
                              spacing: tuple[float, ...] | None = None) -> dict[str, float]:
    """Class-wise HD95/ASSD/boundary-F1 averages for a single sample."""
    hd_vals, assd_vals, bf_vals = [], [], []
    for c in range(1, max(2, n_classes)):
        p, t = pred == c, target == c
        if not t.any() and not p.any():
            continue
        h, a = hd95(p, t, spacing), assd(p, t, spacing)
        b = boundary_f1(p, t, 2.0, spacing)
        if h is not None:
            hd_vals.append(h)
        if a is not None:
            assd_vals.append(a)
        if b is not None:
            bf_vals.append(b)
    out: dict[str, float] = {}
    if hd_vals:
        out["hd95"] = float(np.mean(hd_vals))
    if assd_vals:
        out["assd"] = float(np.mean(assd_vals))
    if bf_vals:
        out["boundary_f1"] = float(np.mean(bf_vals))
    out["volume_similarity"] = volume_similarity(pred, target)
    return out
