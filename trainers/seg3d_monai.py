"""The MONAI-based 3D segmentation trainer.

Training runs on patches; validation runs over the whole volume with a sliding
window. That distinction matters: the Dice measured on a patch is optimistic,
because patches are drawn around the foreground. What is clinically meaningful is
the value measured over the whole volume — which is why the monitored metric comes
from the sliding-window result.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from core.schemas import Layout, Task
from data.datasets_3d import build_datasets_3d, median_spacing
from metrics.segmentation import SegmentationMetrics
from trainers.base import BaseTrainer
from trainers.preview import segmentation_grid

SLIDING_OVERLAP = 0.5


class MonaiSegmentation3DTrainer(BaseTrainer):
    task = Task.SEGMENTATION3D

    def __init__(self, cfg, writer):
        super().__init__(cfg, writer)
        self.patch = int(cfg.hp.img_size)
        self.spacing: tuple[float, float, float] | None = None

    # ── model ────────────────────────────────────────────────────────────
    def build_model(self) -> nn.Module:
        import monai.networks.nets as nets

        arch = self.cfg.model.arch
        n = self.n_classes()
        size = (self.patch,) * 3
        common = dict(spatial_dims=3, in_channels=1, out_channels=n)

        builders = {
            "SwinUNETR": lambda: nets.SwinUNETR(in_channels=1, out_channels=n,
                                                feature_size=48, spatial_dims=3),
            "UNETR": lambda: nets.UNETR(in_channels=1, out_channels=n, img_size=size,
                                        feature_size=16, hidden_size=768,
                                        mlp_dim=3072, num_heads=12),
            "SegResNet": lambda: nets.SegResNet(spatial_dims=3, in_channels=1,
                                                out_channels=n, init_filters=16,
                                                blocks_down=(1, 2, 2, 4),
                                                blocks_up=(1, 1, 1)),
            "UNet": lambda: nets.UNet(**common, channels=(16, 32, 64, 128, 256),
                                      strides=(2, 2, 2, 2), num_res_units=2),
            "AttentionUnet": lambda: nets.AttentionUnet(
                **common, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)),
            "VNet": lambda: nets.VNet(spatial_dims=3, in_channels=1, out_channels=n),
            "DynUNet": lambda: nets.DynUNet(
                **common,
                kernel_size=[3, 3, 3, 3, 3], strides=[1, 2, 2, 2, 2],
                upsample_kernel_size=[2, 2, 2, 2],
                filters=[32, 64, 128, 256, 320],
                norm_name="instance", deep_supervision=False, res_block=True),
        }
        if arch not in builders:
            raise ValueError(f"Unsupported MONAI architecture: {arch}")

        model = builders[arch]()
        if self.hp.pretrained and arch in ("SwinUNETR", "UNETR"):
            self.w.log(f"No off-the-shelf 3D weights are used for {arch}; "
                       "training from scratch is the norm in 3D segmentation.")
        self.w.log(f"{arch} built · {n} output classes · patch {self.patch}³")
        return model

    # ── data ─────────────────────────────────────────────────────────────
    def build_data(self) -> None:
        from monai.data import DataLoader, list_data_collate

        self.spacing = median_spacing(self.ds)
        self.w.log(f"Resampling the volumes to a spacing of {self.spacing} mm")

        self.datasets = build_datasets_3d(self.ds, self.cfg.aug, self.patch,
                                          self.spacing)
        for split, dataset in self.datasets.items():
            is_train = split == "train"
            self.loaders[split] = DataLoader(
                dataset,
                # Full volumes have different sizes at validation time; batch 1 is required
                batch_size=self.hp.batch_size if is_train else 1,
                shuffle=is_train,
                num_workers=self.hp.num_workers,
                collate_fn=list_data_collate,
                pin_memory=self.device.type == "cuda",
                persistent_workers=self.hp.num_workers > 0,
            )

    def new_metrics(self) -> SegmentationMetrics:
        return SegmentationMetrics(self.n_classes(), self.ds.classes,
                                   ignore_index=self.ds.ignore_index)

    # ── forward pass ─────────────────────────────────────────────────────
    def forward_batch(self, batch):
        x = batch["image"].to(self.device, non_blocking=True)
        y = batch["label"].to(self.device, non_blocking=True)
        y = y.squeeze(1).long()                      # (B,1,D,H,W) → (B,D,H,W)

        if self.model.training:
            return self.model(x), y

        # Validation: the whole volume rather than patches, via a sliding window
        from monai.inferers import sliding_window_inference

        logits = sliding_window_inference(
            x, (self.patch,) * 3, sw_batch_size=1, predictor=self.model,
            overlap=SLIDING_OVERLAP, mode="gaussian",
        )
        return logits, y

    # ── preview ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def save_preview(self, epoch: int, metrics) -> str | None:
        """The axial slices where the foreground is densest."""
        loader = self.loaders.get("val")
        if loader is None:
            return None
        self.model.eval()
        batch = next(iter(loader), None)
        if batch is None:
            return None

        logits, target = self.forward_batch(batch)
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        truth = target[0].cpu().numpy()
        volume = batch["image"][0, 0].cpu().numpy()

        fg_per_slice = (truth > 0).sum(axis=(1, 2))
        if fg_per_slice.max() > 0:
            centre = int(np.argmax(fg_per_slice))
            offsets = [o for o in (-6, 0, 6) if 0 <= centre + o < truth.shape[0]]
            slices = [centre + o for o in offsets][:3]
        else:
            slices = [truth.shape[0] // 2]

        images, gts, prs, dices = [], [], [], []
        for z in slices:
            frame = volume[z]
            lo, hi = float(frame.min()), float(frame.max())
            norm = np.zeros_like(frame) if hi <= lo else (frame - lo) / (hi - lo)
            images.append((np.stack([norm] * 3, axis=-1) * 255).astype(np.uint8))
            gts.append(truth[z].astype(np.int32))
            prs.append(pred[z].astype(np.int32))
            p, t = pred[z] > 0, truth[z] > 0
            denom = p.sum() + t.sum()
            dices.append(1.0 if denom == 0 else float(2 * np.logical_and(p, t).sum() / denom))

        rel = f"{Layout.PREVIEWS}/ep_{epoch:04d}.png"
        segmentation_grid(images, gts, prs, dices, self.cfg.path(rel),
                          f"epoch {epoch} — axial slices (z={slices})",
                          ignore_index=self.ds.ignore_index)
        self.model.train()
        return rel

    def finalize(self, status) -> None:
        super().finalize(status)
        from metrics.report import finalize_segmentation

        finalize_segmentation(self, status)
