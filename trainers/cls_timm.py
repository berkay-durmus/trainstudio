"""The timm-based classification trainer.

timm is the common interface to roughly a thousand pretrained image models; when
`create_model` is given the number of classes and input channels, the head is
rebuilt automatically to match our dataset.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.capabilities import log_applied, model_kwargs
from core.schemas import Layout, Task
from data.datasets_2d import build_datasets
from metrics.classification import ClassificationMetrics
from trainers.base import BaseTrainer
from trainers.preview import classification_grid, denormalize


class TimmClassificationTrainer(BaseTrainer):
    task = Task.CLASSIFICATION

    # ── model ────────────────────────────────────────────────────────────
    def build_model(self) -> nn.Module:
        import timm

        kwargs = dict(
            pretrained=self.hp.pretrained,
            num_classes=self.n_classes(),
            in_chans=self.ds.channels if self.ds.channels in (1, 3) else 3,
        )
        reg = model_kwargs(self.cfg)
        kwargs.update(reg)
        log_applied(self.cfg, self.w, reg)

        try:
            model = timm.create_model(self.cfg.model.arch, **kwargs)
        except RuntimeError as exc:
            if not self.hp.pretrained:
                raise
            # If the weights could not be downloaded (offline, etc.), warn instead of failing
            self.w.log(f"Could not load pretrained weights ({exc}); starting from scratch.",
                       "warning")
            kwargs["pretrained"] = False
            model = timm.create_model(self.cfg.model.arch, **kwargs)

        cfg = getattr(model, "pretrained_cfg", {}) or {}
        if cfg.get("input_size"):
            native = cfg["input_size"][-1]
            if native != self.hp.img_size:
                self.w.log(
                    f"The model was pretrained for {native}px but is training at "
                    f"{self.hp.img_size}px. Architectures that require a fixed size may fail."
                )
        return model

    def backbone_modules(self) -> list[nn.Module]:
        """Everything except the classification head is the backbone."""
        model = self.model
        classifier = None
        for getter in ("get_classifier",):
            if hasattr(model, getter):
                try:
                    classifier = getattr(model, getter)()
                except Exception:
                    classifier = None
        head_params = set()
        if classifier is not None:
            head_params = {id(p) for p in classifier.parameters()}

        class _Body(nn.Module):
            def __init__(self, params):
                super().__init__()
                self._params = params

            def parameters(self, recurse: bool = True):
                return iter(self._params)

        body = [p for p in model.parameters() if id(p) not in head_params]
        return [_Body(body)] if body else []

    # ── data ─────────────────────────────────────────────────────────────
    def build_data(self) -> None:
        channels = self.ds.channels if self.ds.channels in (1, 3) else 3
        self.datasets = build_datasets(self.ds, self.cfg.aug, self.hp.img_size, channels)

        pin = self.device.type == "cuda"
        for split, dataset in self.datasets.items():
            self.loaders[split] = DataLoader(
                **self.loader_source(split, dataset),
                batch_size=self.hp.batch_size,
                num_workers=self.hp.num_workers,
                pin_memory=pin,
                drop_last=(split == "train" and len(dataset) > self.hp.batch_size),
                persistent_workers=self.hp.num_workers > 0,
            )

    def new_metrics(self) -> ClassificationMetrics:
        return ClassificationMetrics(self.n_classes(), self.ds.classes)

    # ── MixUp / CutMix ───────────────────────────────────────────────────
    def forward_batch(self, batch):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        if self.hp.channels_last and x.ndim == 4:
            x = x.to(memory_format=torch.channels_last)

        self._mix = None
        if self.model.training:
            x, self._mix = self._maybe_mix(x, y)
        return self.model(x), y

    def _maybe_mix(self, x: torch.Tensor, y: torch.Tensor):
        """Returns (perm, lam) if MixUp/CutMix was applied; the loss blends both targets."""
        aug = self.cfg.aug
        use_mixup = aug.mixup > 0
        use_cutmix = aug.cutmix > 0
        if not (use_mixup or use_cutmix) or x.shape[0] < 2:
            return x, None

        rng = np.random
        if use_mixup and use_cutmix:
            do_cutmix = rng.rand() < 0.5
        else:
            do_cutmix = use_cutmix
        alpha = aug.cutmix if do_cutmix else aug.mixup
        lam = float(rng.beta(alpha, alpha))
        perm = torch.randperm(x.shape[0], device=x.device)

        if do_cutmix:
            h, w = x.shape[-2:]
            r = np.sqrt(1 - lam)
            ch, cw = int(h * r), int(w * r)
            cy, cx = rng.randint(h), rng.randint(w)
            y1, y2 = np.clip([cy - ch // 2, cy + ch // 2], 0, h)
            x1, x2 = np.clip([cx - cw // 2, cx + cw // 2], 0, w)
            x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
            lam = 1 - ((y2 - y1) * (x2 - x1) / (h * w))
        else:
            x = lam * x + (1 - lam) * x[perm]
        return x, (perm, lam)

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mix = getattr(self, "_mix", None)
        if mix is None:
            return self.criterion(logits, target)
        perm, lam = mix
        return lam * self.criterion(logits, target) + \
            (1 - lam) * self.criterion(logits, target[perm])

    # ── preview ──────────────────────────────────────────────────────────
    def save_preview(self, epoch: int, metrics) -> str | None:
        """The validation samples that were misclassified most confidently."""
        if metrics is None or not isinstance(metrics, ClassificationMetrics):
            return None
        idxs = metrics.misclassified_indices(limit=12)
        val_ds = self.datasets.get("val")
        if val_ds is None:
            return None

        probs, targets, preds = metrics.probs, metrics.targets, metrics.preds
        if not idxs:
            # If everything is correct, show the least confident correct predictions
            conf = probs[np.arange(len(preds)), preds] if len(preds) else np.array([])
            if conf.size == 0:
                return None
            idxs = np.argsort(conf)[:6].tolist()
            title = f"epoch {epoch} — least confident correct predictions"
        else:
            title = f"epoch {epoch} — most confident mistakes"

        mean, std = self._norm_stats()
        images, truths, predicted, confs = [], [], [], []
        for i in idxs[:12]:
            if i >= len(val_ds):
                continue
            x, _ = val_ds[i]
            images.append(denormalize(x, mean, std))
            truths.append(self.ds.classes[targets[i]])
            predicted.append(self.ds.classes[preds[i]])
            confs.append(float(probs[i, preds[i]]))
        if not images:
            return None

        rel = f"{Layout.PREVIEWS}/ep_{epoch:04d}.png"
        classification_grid(images, truths, predicted, confs,
                            self.cfg.path(rel), title)
        return rel

    def _norm_stats(self):
        from data.datasets_2d import IMAGENET_MEAN, IMAGENET_STD

        if self.cfg.aug.normalize == "imagenet":
            return IMAGENET_MEAN, IMAGENET_STD
        return None, None

    # ── finalisation ─────────────────────────────────────────────────────
    def finalize(self, status) -> None:
        super().finalize(status)
        from metrics.report import finalize_classification

        finalize_classification(self, status)
