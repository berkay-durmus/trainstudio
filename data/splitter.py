"""Stratified splitting for datasets that only contain `train/`.

Files are **never copied or moved**: a `splits.json` is created at the root and
the file lists are read from it during training. That way several splits can be
tried on the same raw data and the original folder never changes.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from core.schemas import Task
from data import spec as dspec
from data.readers import is_readable_image, is_volume, list_files

SPLITS_FILE = "splits.json"


def splits_path(root: str | Path) -> Path:
    return Path(root) / SPLITS_FILE


def read_splits(root: str | Path) -> dict | None:
    p = splits_path(root)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def make_splits(
    scan_result,
    val_frac: float = 0.2,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict:
    """Split the samples under `train/` in a stratified way and write splits.json.

    In classification, stratification is done per class; in segmentation the
    samples are sorted by file name and shuffled deterministically.
    """
    root = Path(scan_result.root)
    task: Task = scan_result.task
    rng = random.Random(seed)

    if task == Task.CLASSIFICATION:
        groups = _classification_groups(root)
    else:
        groups = {"__all__": _segmentation_items(root, task)}

    if not any(groups.values()):
        raise ValueError("No samples to split were found under `train/`.")

    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    per_class: dict[str, dict[str, int]] = defaultdict(dict)

    for label, items in groups.items():
        items = sorted(items)
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        # Keep at least one sample in every split (so no class disappears entirely)
        if n >= 3:
            n_val = max(1, n_val) if val_frac > 0 else 0
            n_test = max(1, n_test) if test_frac > 0 else 0
            n_val = min(n_val, n - 1 - n_test)

        test = items[:n_test]
        val = items[n_test:n_test + n_val]
        train = items[n_test + n_val:]

        for split, part in (("train", train), ("val", val), ("test", test)):
            result[split].extend(part)
            per_class[label][split] = len(part)

    info = {
        "version": 1,
        "task": task.value,
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "stratified": task == Task.CLASSIFICATION,
        "counts": {k: len(v) for k, v in result.items()},
        "per_class": {k: dict(v) for k, v in per_class.items()},
        "splits": {k: v for k, v in result.items() if v},
    }
    splits_path(root).write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return info


def _classification_groups(root: Path) -> dict[str, list[str]]:
    """class name → file paths relative to the dataset root."""
    train = dspec.split_dir(root, "train")
    groups: dict[str, list[str]] = {}
    for cd in dspec.subdirs(train):
        groups[cd.name] = [str(p.relative_to(root)) for p in list_files(cd)]
    return groups


def _segmentation_items(root: Path, task: Task) -> list[str]:
    idir = dspec.images_dir(root, "train")
    pred = is_volume if task == Task.SEGMENTATION3D else is_readable_image
    return [str(p.relative_to(root)) for p in list_files(idir, predicate=pred)]


def split_summary(info: dict) -> str:
    c = info["counts"]
    base = (f"Split written — {c['train']:,} train · {c['val']:,} validation · "
            f"{c['test']:,} test (seed {info['seed']})")
    return base + (", stratified by class." if info.get("stratified") else ".")


def resolve_split_files(root: str | Path, split: str) -> list[Path] | None:
    """Return that split's file paths if splits.json exists, otherwise None."""
    info = read_splits(root)
    if not info or split not in info.get("splits", {}):
        return None
    return [Path(root) / rel for rel in info["splits"][split]]
