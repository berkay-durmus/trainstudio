"""Classification metrics.

The accumulator (`ClassificationMetrics`) is fed batch by batch and computes
everything once at the end of an epoch. Probabilities are retained because
AUROC/AUPRC and calibration are threshold independent; the memory cost is
N×K float32 (100k samples × 10 classes ≈ 4 MB).
"""

from __future__ import annotations

import numpy as np


class ClassificationMetrics:
    def __init__(self, n_classes: int, class_names: list[str] | None = None):
        self.n_classes = n_classes
        self.class_names = class_names or [str(i) for i in range(n_classes)]
        self.reset()

    def reset(self) -> None:
        self._probs: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def update(self, logits, targets) -> None:
        """logits: a (B,K) tensor/array, targets: (B,) integers."""
        import torch

        if isinstance(logits, torch.Tensor):
            probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
        else:
            probs = _softmax(np.asarray(logits, dtype=np.float64))
        t = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) \
            else np.asarray(targets)
        self._probs.append(probs.astype(np.float32))
        self._targets.append(t.astype(np.int64).ravel())

    # ── derived arrays ───────────────────────────────────────────────────
    @property
    def probs(self) -> np.ndarray:
        return np.concatenate(self._probs) if self._probs else np.zeros((0, self.n_classes))

    @property
    def targets(self) -> np.ndarray:
        return np.concatenate(self._targets) if self._targets else np.zeros(0, dtype=np.int64)

    @property
    def preds(self) -> np.ndarray:
        p = self.probs
        return p.argmax(axis=1) if len(p) else np.zeros(0, dtype=np.int64)

    @property
    def empty(self) -> bool:
        return not self._targets

    # ── the main computation ─────────────────────────────────────────────
    def compute(self, full: bool = True) -> dict[str, float]:
        """`full=False` → the cheap subset used during training (no AUROC)."""
        if self.empty:
            return {}
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, cohen_kappa_score,
            f1_score, precision_score, recall_score,
        )

        y, p, probs = self.targets, self.preds, self.probs
        out: dict[str, float] = {
            "accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0)),
            "precision_macro": float(precision_score(y, p, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y, p, average="macro", zero_division=0)),
        }
        if not full:
            return out

        out["kappa"] = float(cohen_kappa_score(y, p)) if len(np.unique(y)) > 1 else 0.0
        out["ece"] = float(expected_calibration_error(probs, y))
        if self.n_classes >= 2 and len(np.unique(y)) > 1:
            auroc, auprc = _auc_scores(probs, y, self.n_classes)
            if auroc is not None:
                out["auroc"] = auroc
            if auprc is not None:
                out["auprc"] = auprc
        if self.n_classes > 2:
            k = min(5, self.n_classes)
            out[f"top{k}_accuracy"] = float(top_k_accuracy(probs, y, k))
        return out

    # ── detailed outputs (end of epoch / final only) ─────────────────────
    def confusion_matrix(self) -> np.ndarray:
        from sklearn.metrics import confusion_matrix

        if self.empty:
            return np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        return confusion_matrix(self.targets, self.preds,
                                labels=list(range(self.n_classes)))

    def per_class(self) -> list[dict]:
        from sklearn.metrics import precision_recall_fscore_support

        if self.empty:
            return []
        pr, rc, f1, sup = precision_recall_fscore_support(
            self.targets, self.preds, labels=list(range(self.n_classes)), zero_division=0
        )
        rows = []
        for i, name in enumerate(self.class_names):
            rows.append({
                "class": name, "support": int(sup[i]),
                "precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
            })
        return rows

    def roc_data(self) -> dict[str, tuple[list, list, float]]:
        """{class: (fpr, tpr, auc)} — one versus rest."""
        from sklearn.metrics import auc, roc_curve

        out = {}
        y, probs = self.targets, self.probs
        for i, name in enumerate(self.class_names):
            binary = (y == i).astype(int)
            if binary.sum() == 0 or binary.sum() == len(binary):
                continue
            fpr, tpr, _ = roc_curve(binary, probs[:, i])
            out[name] = (fpr.tolist(), tpr.tolist(), float(auc(fpr, tpr)))
        return out

    def pr_data(self) -> dict[str, tuple[list, list, float]]:
        from sklearn.metrics import average_precision_score, precision_recall_curve

        out = {}
        y, probs = self.targets, self.probs
        for i, name in enumerate(self.class_names):
            binary = (y == i).astype(int)
            if binary.sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(binary, probs[:, i])
            out[name] = (rec.tolist(), prec.tolist(),
                         float(average_precision_score(binary, probs[:, i])))
        return out

    def calibration_data(self, n_bins: int = 12):
        return calibration_bins(self.probs, self.targets, n_bins)

    def predictions_table(self, paths: list[str] | None = None) -> list[dict]:
        """Per-sample predictions — written out as predictions/*.csv."""
        y, p, probs = self.targets, self.preds, self.probs
        rows = []
        for i in range(len(y)):
            row = {
                "truth": self.class_names[y[i]],
                "prediction": self.class_names[p[i]],
                "correct": bool(y[i] == p[i]),
                "confidence": float(probs[i, p[i]]),
            }
            if paths is not None and i < len(paths):
                row = {"file": paths[i], **row}
            for k, name in enumerate(self.class_names):
                row[f"p_{name}"] = float(probs[i, k])
            rows.append(row)
        return rows

    def misclassified_indices(self, limit: int = 12) -> list[int]:
        """The most confident mistakes — the most instructive ones for a preview."""
        y, p, probs = self.targets, self.preds, self.probs
        wrong = np.where(y != p)[0]
        if len(wrong) == 0:
            return []
        conf = probs[wrong, p[wrong]]
        return wrong[np.argsort(-conf)][:limit].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _auc_scores(probs: np.ndarray, y: np.ndarray, n_classes: int):
    from sklearn.metrics import average_precision_score, roc_auc_score

    auroc = auprc = None
    try:
        if n_classes == 2:
            auroc = float(roc_auc_score(y, probs[:, 1]))
            auprc = float(average_precision_score(y, probs[:, 1]))
        else:
            present = np.unique(y)
            if len(present) == n_classes:
                auroc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
                onehot = np.eye(n_classes)[y]
                auprc = float(average_precision_score(onehot, probs, average="macro"))
            else:
                # If a class is missing, average only over the ones that are present
                aucs = []
                for i in present:
                    b = (y == i).astype(int)
                    if 0 < b.sum() < len(b):
                        aucs.append(roc_auc_score(b, probs[:, i]))
                auroc = float(np.mean(aucs)) if aucs else None
    except Exception:
        pass
    return auroc, auprc


def top_k_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    if len(y) == 0:
        return 0.0
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y[i] in topk[i] for i in range(len(y))]))


def calibration_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 12):
    """(bin centres, observed accuracy, sample count)"""
    if len(y) == 0:
        return [], [], []
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        centers.append(float(conf[m].mean()))
        accs.append(float(correct[m].mean()))
        counts.append(int(m.sum()))
    return centers, accs, counts


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """The weighted mean gap between confidence and actual accuracy."""
    if len(y) == 0:
        return 0.0
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def bootstrap_ci(probs: np.ndarray, y: np.ndarray, metric_fn, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """A 95% confidence interval on the test set — for publication-facing reports.

    Returns: (point estimate, lower bound, upper bound)
    """
    rng = np.random.default_rng(seed)
    point = float(metric_fn(probs, y))
    n = len(y)
    if n < 10:
        return point, point, point
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(float(metric_fn(probs[idx], y[idx])))
        except Exception:
            continue
    if not vals:
        return point, point, point
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
