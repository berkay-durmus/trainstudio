"""Model construction for the torchvision backend.

torchvision ships the reference implementations of the architectures everything
else is measured against, and — unlike timm — a set of *segmentation* models
(DeepLabV3, FCN, LRASPP) with COCO-trained heads. Two things have to be
reconciled with the rest of the project:

* **The head has the wrong width.** A pretrained classifier predicts its
  original 1000 ImageNet classes, so the final `Linear` is replaced with one
  sized for this dataset. torchvision has no `num_classes` that coexists with
  pretrained weights the way timm's does, and the attribute holding the head
  differs per family (`fc`, `head`, `heads.head`, `classifier.2`, …), so it is
  located structurally: the last `Linear` in definition order. The segmentation
  builders *do* take `num_classes` alongside `weights_backbone`, so those need
  no surgery.
* **The output shape.** Segmentation models return an `OrderedDict`; the
  trainers, the metrics, Grad-CAM and the ONNX export all expect a plain logit
  tensor, so the model is wrapped rather than special-cased in each of them.

Kept free of trainer state so `export/inference.py` can rebuild a checkpoint's
architecture from here directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UnwrapDict(nn.Module):
    """Return the `out` tensor of a torchvision segmentation model.

    Wrapping is deliberate: it keeps one shape convention for every backend, so
    the preview, metric, export and Grad-CAM paths need no branch. It does add a
    `net.` prefix to the state-dict keys, which is why inference rebuilds the
    architecture through this same function rather than assembling it itself.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if isinstance(out, dict):
            return out["out"]
        return out


def _last_linear(model: nn.Module) -> tuple[str, nn.Linear] | tuple[None, None]:
    name = mod = None
    for candidate_name, candidate in model.named_modules():
        if isinstance(candidate, nn.Linear):
            name, mod = candidate_name, candidate
    return name, mod


def replace_head(model: nn.Module, n_classes: int) -> str:
    """Swap the final `Linear` for one predicting `n_classes`. Returns its name.

    `getattr`/`setattr` reach into an `nn.Sequential` as well: its children are
    registered under the string keys "0", "1", … so `classifier.2` resolves the
    same way `heads.head` does.
    """
    name, old = _last_linear(model)
    if name is None or old is None:
        raise RuntimeError(
            "No Linear layer found in this torchvision model, so its "
            "classification head cannot be resized."
        )

    parent = model
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], nn.Linear(old.in_features, n_classes))
    return name


def adapt_in_channels(model: nn.Module, channels: int) -> bool:
    """Retune the first convolution for a non-RGB input. True if it changed.

    torchvision models are all 3-channel. For a grayscale or CT dataset the RGB
    filters are summed instead of being thrown away: a replicated grey image
    through the summed filter gives the same response as through the original,
    so the pretrained features survive the change.
    """
    if channels == 3:
        return False

    name = old = None
    for candidate_name, candidate in model.named_modules():
        if isinstance(candidate, nn.Conv2d):
            name, old = candidate_name, candidate
            break
    if name is None or old is None:
        return False

    new = nn.Conv2d(
        channels, old.out_channels, old.kernel_size, old.stride,
        old.padding, old.dilation, old.groups, old.bias is not None,
    )
    with torch.no_grad():
        weight = old.weight.data
        if channels == 1:
            new.weight.data = weight.sum(dim=1, keepdim=True)
        else:
            # Cycle the RGB filters to cover however many channels there are,
            # then rescale so the summed response keeps its original magnitude.
            reps = (channels + weight.shape[1] - 1) // weight.shape[1]
            tiled = weight.repeat(1, reps, 1, 1)[:, :channels]
            new.weight.data = tiled * (weight.shape[1] / channels)
        if old.bias is not None:
            new.bias.data = old.bias.data

    parent = model
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new)
    return True


def build_classifier(arch: str, n_classes: int, channels: int = 3,
                     pretrained: bool = True, weights: str | None = None,
                     **model_kwargs):
    """A torchvision classification model with a head sized for this dataset.

    `model_kwargs` reach the architecture's constructor (dropout, stochastic
    depth — see core/capabilities.py for which architectures honour which).
    Returns `(model, notes)`; `notes` are lines worth showing in the run log.
    """
    from torchvision.models import get_model

    notes: list[str] = []
    tag = weights or "DEFAULT"
    model = None
    if pretrained:
        try:
            model = get_model(arch, weights=tag, **model_kwargs)
        except Exception as exc:                     # offline, or an unknown tag
            notes.append(f"Could not load the pretrained weights ({exc}); "
                         "starting from scratch.")
    if model is None:
        model = get_model(arch, weights=None, **model_kwargs)

    head = replace_head(model, n_classes)
    notes.append(f"{arch} · head `{head}` rebuilt for {n_classes} classes")
    if adapt_in_channels(model, channels):
        notes.append(f"The first convolution was adapted to {channels} channel(s).")
    return model, notes


def build_segmenter(arch: str, n_classes: int, channels: int = 3,
                    pretrained: bool = True):
    """A torchvision segmentation model, wrapped to return a logit tensor.

    `weights_backbone` is the point of this call: it keeps the ImageNet-trained
    encoder while `num_classes` builds a fresh head, which is what transfer
    learning onto a new label set needs. Asking for the full COCO `weights`
    instead would fix the output at its original 21 classes.
    """
    # The top-level `get_model` resolves names across every sub-module,
    # segmentation included; `torchvision.models.segmentation` does not export
    # one of its own.
    from torchvision.models import get_model

    notes: list[str] = []
    net = None
    if pretrained:
        try:
            net = get_model(arch, weights=None, weights_backbone="DEFAULT",
                            num_classes=n_classes)
            notes.append(f"{arch} · ImageNet-pretrained encoder, "
                         f"fresh head for {n_classes} classes")
        except Exception as exc:
            notes.append(f"Could not load the encoder weights ({exc}); "
                         "starting from scratch.")
    if net is None:
        net = get_model(arch, weights=None, weights_backbone=None,
                        num_classes=n_classes)
        notes.append(f"{arch} · trained from scratch for {n_classes} classes")

    if adapt_in_channels(net, channels):
        notes.append(f"The first convolution was adapted to {channels} channel(s).")
    return UnwrapDict(net), notes
