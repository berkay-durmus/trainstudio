"""2D semantic segmentation built on HuggingFace transformers.

Two different model families are handled by a single trainer:

* **Pixel-logit models** (SegFormer, UperNet, DPT) return a low-resolution logit
  map; we upsample it to the target size and apply our own loss function — so the
  Dice/Focal combinations the user picked apply here too.
* **Mask-classification models** (Mask2Former) want a completely different target
  format: per-instance binary masks plus class ids. For that family we must use
  the model's own (Hungarian-matched) loss; for metric computation the output is
  reduced back to a semantic map.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.capabilities import log_applied, model_kwargs
from core.schemas import Layout, Task
from data.datasets_2d import build_datasets
from metrics.segmentation import SegmentationMetrics
from trainers.base import BaseTrainer
from trainers.preview import denormalize, segmentation_grid

MASK_CLASSIFICATION_FAMILIES = ("mask2former", "maskformer", "oneformer")


class HfSegmentationTrainer(BaseTrainer):
    task = Task.SEGMENTATION

    @property
    def is_mask_classification(self) -> bool:
        return any(f in self.cfg.model.arch.lower() for f in MASK_CLASSIFICATION_FAMILIES)

    # ── model ────────────────────────────────────────────────────────────
    def build_model(self) -> nn.Module:
        from transformers import AutoModelForSemanticSegmentation

        name = self.cfg.model.arch
        n = self.n_classes()
        labels = {str(i): c for i, c in enumerate(self.ds.classes)}
        common = dict(
            num_labels=n,
            id2label=labels,
            label2id={v: int(k) for k, v in labels.items()},
            # The pretrained head has a different number of classes; it must be rebuilt
            ignore_mismatched_sizes=True,
        )
        # Dropout settings are config attributes: from_pretrained overrides them
        reg = model_kwargs(self.cfg)
        common.update(reg)
        log_applied(self.cfg, self.w, reg)

        if self.is_mask_classification:
            from transformers import Mask2FormerForUniversalSegmentation

            model = Mask2FormerForUniversalSegmentation.from_pretrained(name, **common) \
                if self.hp.pretrained else \
                Mask2FormerForUniversalSegmentation.from_config(
                    Mask2FormerForUniversalSegmentation.config_class.from_pretrained(
                        name, **common))
            self.w.log("Mask2Former: the model's own Hungarian-matched loss is used; "
                       "the loss function chosen in Settings does not apply to this family.")
            return model

        model = AutoModelForSemanticSegmentation.from_pretrained(name, **common)
        self.w.log(f"{name} loaded · the head was rebuilt for {n} classes")
        return model

    def backbone_modules(self) -> list[nn.Module]:
        for attr in ("segformer", "backbone", "model", "base_model"):
            mod = getattr(self.model, attr, None)
            if isinstance(mod, nn.Module):
                return [mod]
        return []

    # ── data ─────────────────────────────────────────────────────────────
    def build_data(self) -> None:
        channels = 3            # HF image models expect 3 channels
        self.datasets = build_datasets(self.ds, self.cfg.aug, self.hp.img_size, channels)
        pin = self.device.type == "cuda"
        for split, dataset in self.datasets.items():
            self.loaders[split] = DataLoader(
                dataset,
                batch_size=self.hp.batch_size,
                shuffle=(split == "train"),
                num_workers=self.hp.num_workers,
                pin_memory=pin,
                drop_last=(split == "train" and len(dataset) > self.hp.batch_size),
                persistent_workers=self.hp.num_workers > 0,
            )

    def new_metrics(self) -> SegmentationMetrics:
        return SegmentationMetrics(self.n_classes(), self.ds.classes,
                                   ignore_index=self.ds.ignore_index)

    def build_criterion(self) -> nn.Module:
        if self.is_mask_classification:
            return nn.Identity()          # the loss lives inside the model
        return super().build_criterion()

    # ── forward pass ─────────────────────────────────────────────────────
    def forward_batch(self, batch):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        if self.is_mask_classification:
            return self._forward_mask2former(x, y)

        out = self.model(pixel_values=x)
        logits = out.logits
        # SegFormer returns logits at 1/4 resolution
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:],
                                   mode="bilinear", align_corners=False)
        return logits, y

    def _forward_mask2former(self, x: torch.Tensor, y: torch.Tensor):
        """Convert the semantic mask into instance masks plus class ids."""
        mask_labels, class_labels = [], []
        for i in range(y.shape[0]):
            target = y[i]
            present = [int(v) for v in torch.unique(target)
                       if int(v) != self.ds.ignore_index]
            if not present:
                # A completely empty target, so the model can learn the "no object" case
                mask_labels.append(torch.zeros((0, *target.shape), device=self.device))
                class_labels.append(torch.zeros((0,), dtype=torch.long, device=self.device))
                continue
            masks = torch.stack([(target == v).float() for v in present])
            mask_labels.append(masks)
            class_labels.append(torch.tensor(present, dtype=torch.long, device=self.device))

        out = self.model(pixel_values=x, mask_labels=mask_labels, class_labels=class_labels)
        self._native_loss = out.loss
        logits = self._semantic_from_mask_output(out, y.shape[-2:])
        return logits, y

    def _semantic_from_mask_output(self, out, size) -> torch.Tensor:
        """Derive a per-pixel class score from the mask queries and class logits.

        This is Mask2Former's standard semantic inference rule:
        score(c, pixel) = Σ_q P(class=c | q) · mask_q(pixel)
        """
        class_logits = out.class_queries_logits          # (B, Q, K+1)
        mask_logits = out.masks_queries_logits           # (B, Q, h, w)
        mask_logits = F.interpolate(mask_logits, size=size, mode="bilinear",
                                    align_corners=False)
        class_probs = class_logits.softmax(dim=-1)[..., :-1]   # drop the "no object" class
        mask_probs = mask_logits.sigmoid()
        return torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.is_mask_classification:
            return self._native_loss
        return self.criterion(logits, target)

    # ── preview ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def save_preview(self, epoch: int, metrics) -> str | None:
        loader = self.loaders.get("val")
        if loader is None:
            return None
        self.model.eval()
        batch = next(iter(loader), None)
        if batch is None:
            return None

        logits, target = self.forward_batch(batch)
        pred = logits.argmax(dim=1).cpu().numpy()
        truth = target.cpu().numpy()
        x = batch[0]

        from data.datasets_2d import IMAGENET_MEAN, IMAGENET_STD

        stats = (IMAGENET_MEAN, IMAGENET_STD) if self.cfg.aug.normalize == "imagenet" \
            else (None, None)

        images, gts, prs, dices = [], [], [], []
        for i in range(min(3, x.shape[0])):
            images.append(denormalize(x[i], *stats))
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

    def finalize(self, status) -> None:
        super().finalize(status)
        from metrics.report import finalize_segmentation

        finalize_segmentation(self, status)
