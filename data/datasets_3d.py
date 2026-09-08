"""The 3D volume dataset (MONAI transform pipeline).

3D segmentation differs from 2D in two fundamental ways:

1. **A full volume does not fit in memory.** Training happens on small patches
   cropped around the foreground; validation sweeps the whole volume with a
   sliding window. Sampling patch centres relative to the foreground keeps the
   model from seeing nothing but background on sparse structures.
2. **Voxel spacing varies from volume to volume.** For a physically meaningful
   segmentation, the volumes are resampled to a common spacing.
"""

from __future__ import annotations

from pathlib import Path

from core.schemas import AugConfig, DatasetConfig, Modality
from data import spec as dspec
from data.readers import is_volume, list_files
from data.splitter import resolve_split_files

DEFAULT_SPACING = (1.5, 1.5, 2.0)      # (x, y, z) mm — a common CT starting point


def _vol_stem(p: Path) -> str:
    name = p.name
    for ext in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return p.stem


def volume_items(ds: DatasetConfig, split: str) -> list[dict]:
    """The [{"image": path, "label": path}, …] shape MONAI's dictionary pipeline expects."""
    root = Path(ds.root)

    from_split = resolve_split_files(root, split) if ds.splits_file else None
    if from_split is not None:
        images = from_split
        ldir = dspec.mask_dir(root, "train")
    else:
        idir = dspec.images_dir(root, split)
        ldir = dspec.mask_dir(root, split)
        if not dspec.is_dir(idir) or ldir is None:
            return []
        images = list_files(idir, predicate=is_volume)

    if ldir is None:
        return []
    labels = {_vol_stem(p): p for p in list_files(ldir, predicate=is_volume)}
    return [{"image": str(p), "label": str(labels[_vol_stem(p)])}
            for p in images if _vol_stem(p) in labels]


def build_transforms_3d(ds: DatasetConfig, aug: AugConfig, patch: int, train: bool,
                        spacing: tuple[float, float, float] | None = None):
    """The MONAI transform chain (dictionary based)."""
    from monai import transforms as T

    spacing = spacing or DEFAULT_SPACING
    keys = ["image", "label"]

    ops: list = [
        T.LoadImaged(keys=keys, image_only=True),
        T.EnsureChannelFirstd(keys=keys),
        T.Orientationd(keys=keys, axcodes="RAS"),
        T.Spacingd(keys=keys, pixdim=spacing, mode=("bilinear", "nearest")),
    ]

    # Intensity normalisation depends on the modality: for CT, HU is a fixed
    # physical scale so windowing is right; for MR, a per-volume z-score is.
    if ds.modality == Modality.CT:
        win = ds.window
        lo, hi = win.bounds if win else (-175.0, 250.0)
        ops.append(T.ScaleIntensityRanged(keys=["image"], a_min=lo, a_max=hi,
                                          b_min=0.0, b_max=1.0, clip=True))
    else:
        ops.append(T.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True))

    ops.append(T.CropForegroundd(keys=keys, source_key="image", allow_smaller=True))

    if not train:
        return T.Compose(ops)

    size = (patch, patch, patch)
    ops += [
        T.SpatialPadd(keys=keys, spatial_size=size),
        # Half of the patches are centred on the foreground: on sparse targets this
        # stops the model from never seeing a positive voxel.
        T.RandCropByPosNegLabeld(
            keys=keys, label_key="label", spatial_size=size,
            pos=2, neg=1, num_samples=1, image_key="image", allow_smaller=True,
        ),
    ]
    if aug.hflip > 0:
        ops.append(T.RandFlipd(keys=keys, spatial_axis=0, prob=aug.hflip))
    if aug.vflip > 0:
        ops.append(T.RandFlipd(keys=keys, spatial_axis=1, prob=aug.vflip))
    if aug.rot90 > 0:
        ops.append(T.RandRotate90d(keys=keys, prob=aug.rot90, max_k=3))
    if aug.affine_p > 0:
        import math

        rad = math.radians(min(aug.rotate_limit, 20))
        ops.append(T.RandAffined(
            keys=keys, prob=aug.affine_p,
            rotate_range=(rad, rad, rad),
            scale_range=(aug.scale_limit,) * 3,
            translate_range=tuple(aug.shift_limit * patch for _ in range(3)),
            mode=("bilinear", "nearest"), padding_mode="zeros",
        ))
    if aug.hu_scale > 0:
        ops.append(T.RandScaleIntensityd(keys=["image"], factors=aug.hu_scale, prob=0.5))
    if aug.hu_shift > 0:
        # The HU shift is converted to the [0,1] scale produced by ScaleIntensityRanged
        win = ds.window
        span = (win.width if win else 425.0) or 425.0
        ops.append(T.RandShiftIntensityd(keys=["image"],
                                         offsets=float(aug.hu_shift / span), prob=0.5))
    if aug.noise_p > 0:
        ops.append(T.RandGaussianNoised(keys=["image"], prob=aug.noise_p, std=0.03))
    if aug.blur_p > 0:
        ops.append(T.RandGaussianSmoothd(keys=["image"], prob=aug.blur_p))
    if aug.elastic_p > 0:
        ops.append(T.Rand3DElasticd(
            keys=keys, prob=aug.elastic_p, sigma_range=(5, 8), magnitude_range=(50, 110),
            mode=("bilinear", "nearest"), padding_mode="zeros",
        ))

    return T.Compose(ops)


def build_datasets_3d(ds: DatasetConfig, aug: AugConfig, patch: int,
                      spacing: tuple[float, float, float] | None = None,
                      cache_rate: float = 0.0) -> dict:
    """MONAI Dataset objects for train/val/test."""
    from monai.data import CacheDataset, Dataset

    out: dict = {}
    for split in ("train", "val", "test"):
        items = volume_items(ds, split)
        if not items:
            continue
        tf = build_transforms_3d(ds, aug, patch, split == "train", spacing)
        if cache_rate > 0:
            out[split] = CacheDataset(data=items, transform=tf,
                                      cache_rate=cache_rate, num_workers=0)
        else:
            out[split] = Dataset(data=items, transform=tf)
    return out


def median_spacing(ds: DatasetConfig, limit: int = 12
                   ) -> tuple[float, float, float]:
    """The dataset's median voxel spacing — the resampling target.

    Using the dataset's own spacing rather than a fixed default avoids needless
    resampling and the interpolation loss that comes with it.
    """
    import statistics

    from data.readers import volume_header

    items = volume_items(ds, "train")[:limit]
    zs, ys, xs = [], [], []
    for it in items:
        try:
            h = volume_header(it["image"])
            z, y, x = h["spacing"]
            zs.append(z), ys.append(y), xs.append(x)
        except Exception:
            continue
    if not zs:
        return DEFAULT_SPACING
    # volume_header returns (D,H,W) order; MONAI's pixdim wants (x,y,z)
    return (round(statistics.median(xs), 3),
            round(statistics.median(ys), 3),
            round(statistics.median(zs), 3))
