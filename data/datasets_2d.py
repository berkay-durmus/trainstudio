"""2D datasets and the augmentation pipeline (albumentations 2.x).

Classification and segmentation share the same reading layer (data/readers.py);
the only difference is whether a sample's label is an integer or a mask.

If a `splits.json` exists, the file lists are read from it — so the automatic
split works without touching the original folder at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from core.schemas import AugConfig, DatasetConfig, Modality, Task, WindowSpec
from data import spec as dspec
from data.readers import list_files, read_image, read_mask
from data.splitter import resolve_split_files

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────


def build_transforms(
    aug: AugConfig,
    img_size: int,
    train: bool,
    task: Task,
    modality: Modality,
    channels: int = 3,
    dataset_stats: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
):
    """The albumentations pipeline. Masks are transformed in sync via `mask=`."""
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    ops: list = []

    if train and task == Task.CLASSIFICATION and aug.preset in ("medium", "heavy"):
        # Scale cropping is the single most effective augmentation in classification
        scale_lo = 0.6 if aug.preset == "heavy" else 0.75
        ops.append(A.RandomResizedCrop(size=(img_size, img_size),
                                       scale=(scale_lo, 1.0), ratio=(0.85, 1.18), p=1.0))
    else:
        ops.append(A.Resize(img_size, img_size,
                            interpolation=cv2.INTER_LINEAR,
                            mask_interpolation=cv2.INTER_NEAREST))

    if train:
        if aug.hflip > 0:
            ops.append(A.HorizontalFlip(p=aug.hflip))
        if aug.vflip > 0:
            ops.append(A.VerticalFlip(p=aug.vflip))
        if aug.rot90 > 0:
            ops.append(A.RandomRotate90(p=aug.rot90))
        if aug.affine_p > 0:
            ops.append(A.Affine(
                scale=(1 - aug.scale_limit, 1 + aug.scale_limit),
                translate_percent=(-aug.shift_limit, aug.shift_limit),
                rotate=(-aug.rotate_limit, aug.rotate_limit),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0,
                p=aug.affine_p,
            ))
        if aug.brightness_p > 0 and aug.brightness_contrast > 0:
            ops.append(A.RandomBrightnessContrast(
                brightness_limit=aug.brightness_contrast,
                contrast_limit=aug.brightness_contrast, p=aug.brightness_p))
        if aug.gamma_p > 0:
            ops.append(A.RandomGamma(gamma_limit=(80, 120), p=aug.gamma_p))
        if aug.blur_p > 0:
            ops.append(A.GaussianBlur(blur_limit=(3, 7), p=aug.blur_p))
        if aug.noise_p > 0:
            ops.append(A.GaussNoise(std_range=(0.03, 0.12), p=aug.noise_p))
        if aug.sharpen_p > 0:
            ops.append(A.Sharpen(p=aug.sharpen_p))
        if aug.elastic_p > 0:
            ops.append(A.ElasticTransform(
                alpha=1.0, sigma=50, interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT, p=aug.elastic_p))
        if aug.grid_distortion_p > 0:
            ops.append(A.GridDistortion(
                num_steps=5, distort_limit=0.2, interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT, p=aug.grid_distortion_p))
        if aug.coarse_dropout_p > 0 and task == Task.CLASSIFICATION:
            ops.append(A.CoarseDropout(
                num_holes_range=(1, 6), hole_height_range=(0.05, 0.18),
                hole_width_range=(0.05, 0.18), p=aug.coarse_dropout_p))

    # ── Normalisation ────────────────────────────────────────────────────
    if aug.normalize == "imagenet" and channels == 3:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif aug.normalize == "dataset" and dataset_stats is not None:
        mean, std = dataset_stats
    elif aug.normalize == "none":
        mean, std = (0.0,) * channels, (1.0,) * channels
    else:                                     # minmax, or no statistics available
        mean, std = (0.5,) * channels, (0.5,) * channels

    ops.append(A.Normalize(mean=mean[:channels], std=std[:channels], max_pixel_value=255.0))
    ops.append(ToTensorV2())
    return A.Compose(ops)


def compute_dataset_stats(paths: list[Path], modality: Modality,
                          window: WindowSpec | None, channels: int,
                          max_files: int = 200, size: int = 128
                          ) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-channel mean/std of the dataset (on the [0,1] scale)."""
    import cv2

    rng = np.random.default_rng(0)
    sel = paths if len(paths) <= max_files else [
        paths[i] for i in rng.choice(len(paths), max_files, replace=False)
    ]
    sums = np.zeros(channels, dtype=np.float64)
    sqs = np.zeros(channels, dtype=np.float64)
    n = 0
    for p in sel:
        try:
            img = read_image(p, modality, window, channels).astype(np.float32) / 255.0
        except Exception:
            continue
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = img[:, :, None]
        sums += img.reshape(-1, channels).sum(axis=0)
        sqs += (img.reshape(-1, channels) ** 2).sum(axis=0)
        n += img.shape[0] * img.shape[1]
    if n == 0:
        return (0.5,) * channels, (0.5,) * channels
    mean = sums / n
    std = np.sqrt(np.maximum(sqs / n - mean**2, 1e-8))
    return tuple(float(x) for x in mean), tuple(float(max(x, 1e-3)) for x in std)


# ─────────────────────────────────────────────────────────────────────────────
# File listing
# ─────────────────────────────────────────────────────────────────────────────


def classification_items(ds: DatasetConfig, split: str) -> list[tuple[Path, int]]:
    """(file, class index) pairs. Read from splits.json when it exists."""
    root = Path(ds.root)
    class_to_idx = {c: i for i, c in enumerate(ds.classes)}

    from_split = resolve_split_files(root, split) if ds.splits_file else None
    if from_split is not None:
        return [(p, class_to_idx[p.parent.name])
                for p in from_split if p.parent.name in class_to_idx]

    sdir = dspec.split_dir(root, split)
    if not dspec.is_dir(sdir):
        return []
    items: list[tuple[Path, int]] = []
    for cname, idx in class_to_idx.items():
        cdir = sdir / cname
        if dspec.is_dir(cdir):
            items.extend((p, idx) for p in list_files(cdir))
    return items


def segmentation_items(ds: DatasetConfig, split: str) -> list[tuple[Path, Path]]:
    """(image, mask) pairs — matched by identical file stem."""
    root = Path(ds.root)

    from_split = resolve_split_files(root, split) if ds.splits_file else None
    if from_split is not None:
        images = from_split
        mdir = dspec.mask_dir(root, "train")
    else:
        idir = dspec.images_dir(root, split)
        mdir = dspec.mask_dir(root, split)
        if not dspec.is_dir(idir) or mdir is None:
            return []
        images = list_files(idir)

    if mdir is None:
        return []
    masks = {p.stem: p for p in list_files(mdir)}
    return [(p, masks[p.stem]) for p in images if p.stem in masks]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset classes
# ─────────────────────────────────────────────────────────────────────────────


class ClassificationDataset(Dataset):
    def __init__(self, ds: DatasetConfig, split: str, transform, channels: int = 3):
        self.ds = ds
        self.split = split
        self.items = classification_items(ds, split)
        self.transform = transform
        self.channels = channels
        self.window = ds.window

    def __len__(self) -> int:
        return len(self.items)

    @property
    def targets(self) -> list[int]:
        """The label list, used for class weights and stratified sampling."""
        return [t for _, t in self.items]

    def __getitem__(self, i: int):
        path, target = self.items[i]
        try:
            img = read_image(path, self.ds.modality, self.window, self.channels)
        except Exception:
            # A single corrupt file must not bring training down
            img = np.zeros((64, 64, self.channels), dtype=np.uint8)
        out = self.transform(image=img)
        return out["image"], torch.tensor(target, dtype=torch.long)


class SegmentationDataset(Dataset):
    def __init__(self, ds: DatasetConfig, split: str, transform, channels: int = 3,
                 binary_255: bool = False):
        self.ds = ds
        self.split = split
        self.items = segmentation_items(ds, split)
        self.transform = transform
        self.channels = channels
        self.window = ds.window
        # Maps 255 → 1 for binary masks encoded as 0/255
        self.binary_255 = binary_255
        self.ignore_index = ds.ignore_index

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        img_p, mask_p = self.items[i]
        try:
            img = read_image(img_p, self.ds.modality, self.window, self.channels)
            mask = read_mask(mask_p)
        except Exception:
            img = np.zeros((64, 64, self.channels), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.int32)

        if self.binary_255:
            mask = (mask > 127).astype(np.int32)
        else:
            # Pull unexpected values other than ignore_index back to the background
            n = len(self.ds.classes)
            mask = np.where(
                (mask >= n) & (mask != self.ignore_index), 0, mask
            ).astype(np.int32)

        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].long()


# ─────────────────────────────────────────────────────────────────────────────
# Factory — the single function the trainer calls
# ─────────────────────────────────────────────────────────────────────────────


def detect_binary_255(ds: DatasetConfig) -> bool:
    """Are the masks encoded as 0/255 — determined from a handful of samples."""
    items = segmentation_items(ds, "train")[:6]
    if not items:
        return False
    vals: set[int] = set()
    for _, m in items:
        try:
            vals.update(int(v) for v in np.unique(read_mask(m))[:8])
        except Exception:
            continue
    non_ignore = vals - {ds.ignore_index}
    return bool(non_ignore) and max(non_ignore) > 200 and len(non_ignore) <= 2


def build_datasets(
    ds: DatasetConfig, aug: AugConfig, img_size: int, channels: int = 3,
) -> dict[str, Dataset]:
    """Ready-made Dataset objects for train/val/test (empty splits are skipped)."""
    stats = None
    if aug.normalize == "dataset":
        if ds.task == Task.CLASSIFICATION:
            paths = [p for p, _ in classification_items(ds, "train")]
        else:
            paths = [p for p, _ in segmentation_items(ds, "train")]
        stats = compute_dataset_stats(paths, ds.modality, ds.window, channels)

    binary = detect_binary_255(ds) if ds.task == Task.SEGMENTATION else False

    out: dict[str, Dataset] = {}
    for split in ("train", "val", "test"):
        tf = build_transforms(aug, img_size, split == "train", ds.task,
                              ds.modality, channels, stats)
        if ds.task == Task.CLASSIFICATION:
            d = ClassificationDataset(ds, split, tf, channels)
        else:
            d = SegmentationDataset(ds, split, tf, channels, binary_255=binary)
        if len(d) > 0:
            out[split] = d
    return out


def class_weights(dataset: ClassificationDataset, n_classes: int) -> torch.Tensor:
    """Balanced class weights: n / (k · n_c)."""
    counts = np.bincount(dataset.targets, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)
