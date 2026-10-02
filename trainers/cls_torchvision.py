"""Classification with torchvision's reference models.

Only model construction and backbone identification differ from the timm
trainer; the data pipeline, MixUp/CutMix, previews and reporting are identical,
so this subclasses it rather than restating any of that.
"""

from __future__ import annotations

import torch.nn as nn

from core.capabilities import log_applied, model_kwargs
from trainers.cls_timm import TimmClassificationTrainer
from trainers.torchvision_common import build_classifier


class TorchvisionClassificationTrainer(TimmClassificationTrainer):

    def build_model(self) -> nn.Module:
        channels = self.ds.channels if self.ds.channels in (1, 3) else 3
        reg = model_kwargs(self.cfg)
        log_applied(self.cfg, self.w, reg)
        model, notes = build_classifier(
            self.cfg.model.arch,
            n_classes=self.n_classes(),
            channels=channels,
            pretrained=self.hp.pretrained,
            weights=self.cfg.model.weights,
            **reg,
        )
        for note in notes:
            self.w.log(note)
        return model

    def backbone_modules(self) -> list[nn.Module]:
        """Everything except the final Linear.

        The timm implementation asks the model for `get_classifier()`, which
        torchvision models do not have, so the head is located the same way it
        was replaced: the last Linear in definition order.
        """
        from trainers.torchvision_common import _last_linear

        _, head = _last_linear(self.model)
        head_params = {id(p) for p in head.parameters()} if head is not None else set()

        class _Body(nn.Module):
            def __init__(self, params):
                super().__init__()
                self._params = params

            def parameters(self, recurse: bool = True):
                return iter(self._params)

        body = [p for p in self.model.parameters() if id(p) not in head_params]
        return [_Body(body)] if body else []
