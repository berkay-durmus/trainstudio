"""The 2D semantic segmentation trainer built on segmentation_models_pytorch.

What makes smp valuable is that it separates the architecture from the backbone:
U-Net, DeepLabV3+ or UPerNet can all be built on the same timm backbone. The user
chooses those two axes independently and we feed the same data pipeline to both.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.capabilities import log_applied, model_kwargs
from core.schemas import Layout, Task
from data.datasets_2d import build_datasets
from metrics.segmentation import SegmentationMetrics
from trainers.base import BaseTrainer
from trainers.preview import denormalize, segmentation_grid


class SmpSegmentationTrainer(BaseTrainer):
    task = Task.SEGMENTATION

    def build_model(self) -> nn.Module:
        import segmentation_models_pytorch as smp

        encoder = self.cfg.model.encoder or self.hp.encoder or "resnet50"
        weights = self.cfg.model.encoder_weights if self.hp.pretrained else None
        channels = self.ds.channels if self.ds.channels in (1, 3) else 3

        kwargs = dict(
            arch=self.cfg.model.arch,
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=channels,
            classes=self.n_classes(),
        )
        reg = model_kwargs(self.cfg)
        kwargs.update(reg)
        log_applied(self.cfg, self.w, reg)
        try:
            model = smp.create_model(**kwargs)
        except Exception as exc:
            if weights is None:
                raise
            self.w.log(f"Could not load the encoder weights ({exc}); starting from scratch.",
                       "warning")
            kwargs["encoder_weights"] = None
            model = smp.create_model(**kwargs)

        self.w.log(f"Architecture {self.cfg.model.arch} · encoder {encoder}"
                   + (f" ({weights} weights)" if weights else " (from scratch)"))
        return model

    def backbone_modules(self) -> list[nn.Module]:
        enc = getattr(self.model, "encoder", None)
        return [enc] if enc is not None else []

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

    def new_metrics(self) -> SegmentationMetrics:
        return SegmentationMetrics(self.n_classes(), self.ds.classes,
                                   ignore_index=self.ds.ignore_index)

    # ── preview ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def save_preview(self, epoch: int, metrics) -> str | None:
        val_loader = self.loaders.get("val")
        if val_loader is None:
            return None
        self.model.eval()

        batch = next(iter(val_loader), None)
        if batch is None:
            return None
        x, y = batch
        x = x.to(self.device)
        logits = self.model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        truth = y.numpy()

        mean, std = self._norm_stats()
        n = min(3, x.shape[0])
        images, gts, prs, dices = [], [], [], []
        for i in range(n):
            images.append(denormalize(x[i], mean, std))
            gts.append(truth[i])
            prs.append(pred[i])
            p, t = pred[i] > 0, truth[i] > 0
            denom = p.sum() + t.sum()
            dices.append(1.0 if denom == 0 else float(2 * np.logical_and(p, t).sum() / denom))

        rel = f"{Layout.PREVIEWS}/ep_{epoch:04d}.png"
        segmentation_grid(images, gts, prs, dices, self.cfg.path(rel),
                          f"epoch {epoch} — validation samples",
                          ignore_index=self.ds.ignore_index)
        self.model.train()
        return rel

    def _norm_stats(self):
        from data.datasets_2d import IMAGENET_MEAN, IMAGENET_STD

        if self.cfg.aug.normalize == "imagenet":
            return IMAGENET_MEAN, IMAGENET_STD
        return None, None

    def finalize(self, status) -> None:
        super().finalize(status)
        from metrics.report import finalize_segmentation

        finalize_segmentation(self, status)
