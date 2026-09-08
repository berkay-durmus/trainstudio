"""2D semantic segmentation with torchvision's DeepLabV3 / FCN / LRASPP.

The data pipeline, previews, metrics and reporting are shared with the smp
trainer, so only the model differs. `build_segmenter` wraps the model so it
returns a plain logit tensor at input resolution, which is what the inherited
code already expects — no forward-pass override is needed.
"""

from __future__ import annotations

import torch.nn as nn

from trainers.seg_smp import SmpSegmentationTrainer
from trainers.torchvision_common import build_segmenter


class TorchvisionSegmentationTrainer(SmpSegmentationTrainer):

    def build_model(self) -> nn.Module:
        channels = self.ds.channels if self.ds.channels in (1, 3) else 3
        model, notes = build_segmenter(
            self.cfg.model.arch,
            n_classes=self.n_classes(),
            channels=channels,
            pretrained=self.hp.pretrained,
        )
        for note in notes:
            self.w.log(note)
        return model

    def backbone_modules(self) -> list[nn.Module]:
        """The feature extractor, so `freeze_backbone_epochs` trains only the head."""
        net = getattr(self.model, "net", self.model)
        backbone = getattr(net, "backbone", None)
        return [backbone] if isinstance(backbone, nn.Module) else []
