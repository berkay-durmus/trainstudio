"""Converts the canonical dataset into the layout Ultralytics expects.

The user provides one single format; this module performs the necessary
adaptation in a cache directory, **without ever touching the source dataset**:

    classification  → the layout is already identical (train/<class>/*), symlinks suffice
    semantic seg.   → YOLO26 reads masks directly; directory links are enough to
                      transpose images/train ↔ train/images
    instance seg.   → masks are converted to polygons (the only real conversion)

The cache is keyed by the dataset path plus a digest of its contents, so a second
training run on the same dataset does not repeat the conversion.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import yaml

from core.schemas import DatasetConfig, Task
from data import spec as dspec
from data.datasets_2d import classification_items, segmentation_items
from data.readers import read_mask

CACHE_ROOT = Path(os.environ.get("TRAINSTUDIO_HOME", Path.home() / ".trainstudio")) / "yolo_cache"
MIN_POLYGON_AREA = 12          # pixels — smaller components are treated as noise
CONTOUR_EPSILON = 0.0035       # fraction of the perimeter; polygon simplification strength


def _fingerprint(ds: DatasetConfig, mode: str) -> str:
    """The dataset's identity — path, classes, split sizes and the split file."""
    parts = [str(ds.root), mode, ds.task.value, "|".join(ds.classes),
             str(ds.n_train), str(ds.n_val), str(ds.n_test)]
    splits = Path(ds.root) / "splits.json"
    if splits.is_file():
        parts.append(str(splits.stat().st_mtime))
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:16]


def cache_dir(ds: DatasetConfig, mode: str) -> Path:
    return CACHE_ROOT / f"{Path(ds.root).name}_{mode}_{_fingerprint(ds, mode)}"


def _link_dir(src: Path, dst: Path) -> None:
    """Point at a directory with a symlink; copy it if symlinks are unsupported."""
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except (OSError, NotImplementedError):
        shutil.copytree(src, dst)


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    return path


def _names(ds: DatasetConfig, drop_background: bool) -> dict[int, str]:
    """The YOLO `names` map. In instance segmentation the background is not a class."""
    classes = ds.classes[1:] if drop_background and len(ds.classes) > 1 else ds.classes
    return {i: c for i, c in enumerate(classes)}


# ─────────────────────────────────────────────────────────────────────────────
# Classification — no conversion needed
# ─────────────────────────────────────────────────────────────────────────────


def prepare_classification(ds: DatasetConfig, log=print) -> Path:
    """Ultralytics classification reads our layout as it is.

    A real reorganisation is only needed for datasets split via `splits.json`;
    otherwise the source root is used directly.
    """
    root = Path(ds.root)
    # Ultralytics looks for literally `train/` and `val/`. Our scanner accepts
    # `Train/`, `validation/` and friends, so a dataset that only *reads* as
    # canonical still has to be normalised into the cache before handing it over.
    if not ds.splits_file and not dspec.renamed_splits(root):
        return root

    out = cache_dir(ds, "cls")
    marker = out / ".done"
    if marker.is_file():
        return out

    log(f"Preparing the classification layout from the split file: {out}")
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        items = classification_items(ds, split)
        for path, class_idx in items:
            target = out / split / ds.classes[class_idx] / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(path, target)
            except (OSError, NotImplementedError):
                shutil.copy2(path, target)
    marker.write_text("ok", encoding="utf-8")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Semantic segmentation — directory transposition only
# ─────────────────────────────────────────────────────────────────────────────


def prepare_semantic(ds: DatasetConfig, log=print) -> Path:
    """Produce a data.yaml for YOLO26 semantic segmentation.

    YOLO26 reads masks directly as PNGs; the only difference is the directory
    order (`images/train` and `masks/train`), so no polygon conversion is needed.
    """
    root = Path(ds.root)
    out = cache_dir(ds, "sem")
    yaml_path = out / "dataset.yaml"
    if yaml_path.is_file():
        return yaml_path

    log(f"Preparing the semantic segmentation layout: {out}")
    out.mkdir(parents=True, exist_ok=True)

    splits_present = []
    for split in ("train", "val", "test"):
        idir = dspec.images_dir(root, split)
        mdir = dspec.mask_dir(root, split)
        if not dspec.is_dir(idir) or mdir is None:
            continue
        _link_dir(idir, out / "images" / split)
        _link_dir(mdir, out / "masks" / split)
        splits_present.append(split)

    if "train" not in splits_present:
        raise RuntimeError("Semantic segmentation requires `train/images` and `train/masks`.")

    data = {
        "path": str(out),
        "train": "images/train",
        "val": f"images/{'val' if 'val' in splits_present else 'train'}",
        "masks_dir": "masks",
        "nc": len(ds.classes),
        "names": _names(ds, drop_background=False),
    }
    if "test" in splits_present:
        data["test"] = "images/test"
    return _write_yaml(yaml_path, data)


# ─────────────────────────────────────────────────────────────────────────────
# Instance segmentation — mask → polygon
# ─────────────────────────────────────────────────────────────────────────────


def mask_to_polygons(mask: np.ndarray, class_id: int,
                     min_area: int = MIN_POLYGON_AREA) -> list[list[float]]:
    """Convert the connected components of one class into normalised polygons.

    Regions with holes are represented by their outer contour: YOLO's polygon
    format cannot describe holes. This is a small loss in segmentation quality
    and the user is told about it when instance segmentation is selected.
    """
    import cv2

    h, w = mask.shape[:2]
    binary = (mask == class_id).astype(np.uint8)
    if binary.sum() < min_area:
        return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        eps = CONTOUR_EPSILON * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        norm = approx.astype(np.float64)
        norm[:, 0] = np.clip(norm[:, 0] / w, 0.0, 1.0)
        norm[:, 1] = np.clip(norm[:, 1] / h, 0.0, 1.0)
        polygons.append(norm.reshape(-1).tolist())
    return polygons


def prepare_instance(ds: DatasetConfig, log=print) -> Path:
    """Convert the masks into YOLO polygon labels and produce a data.yaml."""
    out = cache_dir(ds, "seg")
    yaml_path = out / "dataset.yaml"
    if yaml_path.is_file():
        log(f"Using the cached YOLO conversion: {out}")
        return yaml_path

    log(f"Converting masks to polygons → {out}")
    out.mkdir(parents=True, exist_ok=True)

    # In instance segmentation the background is not a class; indices shift by one
    n_fg = max(1, len(ds.classes) - 1)
    stats = {"images": 0, "polygons": 0, "empty": 0}
    splits_present: list[str] = []

    for split in ("train", "val", "test"):
        pairs = segmentation_items(ds, split)
        if not pairs:
            continue
        splits_present.append(split)
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, mask_path in pairs:
            link = img_dir / img_path.name
            if not link.exists():
                try:
                    os.symlink(img_path, link)
                except (OSError, NotImplementedError):
                    shutil.copy2(img_path, link)

            try:
                mask = read_mask(mask_path)
            except Exception:
                stats["empty"] += 1
                (lbl_dir / f"{img_path.stem}.txt").write_text("", encoding="utf-8")
                continue

            values = [int(v) for v in np.unique(mask)
                      if v not in (0, ds.ignore_index)]
            lines: list[str] = []
            for v in values:
                yolo_class = v - 1 if len(ds.classes) > 1 else 0
                if not (0 <= yolo_class < n_fg):
                    continue
                for poly in mask_to_polygons(mask, v):
                    coords = " ".join(f"{c:.6f}" for c in poly)
                    lines.append(f"{yolo_class} {coords}")

            (lbl_dir / f"{img_path.stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8")
            stats["images"] += 1
            stats["polygons"] += len(lines)
            if not lines:
                stats["empty"] += 1

    if "train" not in splits_present:
        raise RuntimeError("No matching pairs were found in the training split for the YOLO conversion.")

    log(f"Conversion complete: {stats['images']} images, {stats['polygons']} polygons"
        + (f", {stats['empty']} images with no objects" if stats["empty"] else ""))

    data = {
        "path": str(out),
        "train": "images/train",
        "val": f"images/{'val' if 'val' in splits_present else 'train'}",
        "nc": n_fg,
        "names": _names(ds, drop_background=True),
    }
    if "test" in splits_present:
        data["test"] = "images/test"
    _write_yaml(yaml_path, data)
    (out / "conversion_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────


def prepare(ds: DatasetConfig, arch: str, log=print) -> Path:
    """Pick the right preparation based on the architecture name."""
    name = arch.lower()
    if ds.task == Task.CLASSIFICATION or "-cls" in name:
        return prepare_classification(ds, log)
    if "-sem" in name:
        return prepare_semantic(ds, log)
    return prepare_instance(ds, log)


def clear_cache() -> int:
    """Delete the entire conversion cache; returns how many directories were removed."""
    if not CACHE_ROOT.is_dir():
        return 0
    n = 0
    for d in CACHE_ROOT.iterdir():
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n
