"""Dataset scanning, validation and statistics.

The Dataset page shows the single output of this module (`ScanResult`): is the
structure correct, how many samples are there, what does the class distribution
look like, do the masks line up with the images.

For speed, statistics are computed by sampling: the file *list* is complete, but
pixels are only read for `sample_size` files. That keeps a scan of a 100k-image
dataset down to a few seconds.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from core.schemas import DatasetConfig, Modality, Task, WindowSpec
from data import spec as dspec
from data.readers import (
    ReadError,
    image_shape,
    is_dicom,
    is_volume,
    list_files,
    read_mask,
    volume_header,
)

Level = Literal["error", "warning", "info"]


@dataclass
class Issue:
    level: Level
    title: str
    detail: str = ""
    items: list[str] = field(default_factory=list)     # example file names

    @property
    def icon(self) -> str:
        return {"error": "🛑", "warning": "⚠️", "info": "ℹ️"}[self.level]


@dataclass
class ScanResult:
    root: str
    task: Task | None = None
    modality: Modality = Modality.RGB
    classes: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    # counts
    split_counts: dict[str, int] = field(default_factory=dict)          # split → samples
    class_counts: dict[str, dict[str, int]] = field(default_factory=dict)  # split → class → n

    # statistics derived from the sample
    median_size: tuple[int, int] | None = None
    size_range: tuple[tuple[int, int], tuple[int, int]] | None = None
    channels: int = 3
    mask_fg_ratio: float | None = None
    mask_values: list[int] = field(default_factory=list)
    volume_shape: tuple[int, int, int] | None = None
    volume_spacing: tuple[float, float, float] | None = None
    intensity_range: tuple[float, float] | None = None

    preview: list[dict] = field(default_factory=list)   # {"image": path, "mask": path, "label": str}
    candidates: list[str] = field(default_factory=list)  # dataset roots found one level down
    from_yaml: bool = False
    sampled: int = 0
    elapsed: float = 0.0

    # ── derived ──────────────────────────────────────────────────────────
    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def infos(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "info"]

    @property
    def ok(self) -> bool:
        return self.task is not None and not self.errors

    @property
    def n_train(self) -> int:
        return self.split_counts.get("train", 0)

    @property
    def n_val(self) -> int:
        return self.split_counts.get("val", 0)

    @property
    def n_test(self) -> int:
        return self.split_counts.get("test", 0)

    @property
    def total(self) -> int:
        return sum(self.split_counts.values())

    @property
    def needs_split(self) -> bool:
        """If only train/ exists, the UI offers an automatic split."""
        return self.n_train > 0 and self.n_val == 0

    def totals_per_class(self) -> dict[str, int]:
        out: dict[str, int] = {c: 0 for c in self.classes}
        for counts in self.class_counts.values():
            for c, n in counts.items():
                out[c] = out.get(c, 0) + n
        return out

    @property
    def imbalance_ratio(self) -> float:
        counts = [n for n in self.totals_per_class().values() if n > 0]
        return max(counts) / min(counts) if len(counts) > 1 else 1.0

    # ── bridge to RunConfig ──────────────────────────────────────────────
    def to_dataset_config(self, window: WindowSpec | None = None) -> DatasetConfig:
        if self.task is None:
            raise ValueError("The task could not be determined; the dataset cannot be used unvalidated.")
        return DatasetConfig(
            root=self.root,
            task=self.task,
            modality=self.modality,
            classes=list(self.classes),
            window=window,
            n_train=self.n_train,
            n_val=self.n_val,
            n_test=self.n_test,
            class_counts=self.totals_per_class(),
            median_image_size=self.median_size,
            mask_foreground_ratio=self.mask_fg_ratio,
            channels=self.channels,
        )

    def add(self, level: Level, title: str, detail: str = "", items: list[str] | None = None) -> None:
        self.issues.append(Issue(level, title, detail, items or []))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def scan_dataset(
    root: str | Path,
    task: Task | None = None,
    modality: Modality | None = None,
    sample_size: int = 48,
    seed: int = 0,
) -> ScanResult:
    """Validate a dataset and derive its statistics. Never raises.

    Every failure comes back as an `Issue` on the result, because the caller is
    a page that has to keep rendering: an exception here would replace the whole
    Dataset screen with a traceback.
    """
    t0 = time.time()
    try:
        return _scan_dataset(root, task, modality, sample_size, seed, t0)
    except Exception as exc:                                    # never propagate
        res = ScanResult(root=str(root))
        res.add("error", "The dataset could not be scanned",
                f"{type(exc).__name__}: {exc}")
        return _finish(res, t0)


def _scan_dataset(
    root: str | Path,
    task: Task | None,
    modality: Modality | None,
    sample_size: int,
    seed: int,
    t0: float,
) -> ScanResult:
    root = Path(root).expanduser()
    res = ScanResult(root=str(root))

    try:
        exists, is_folder = root.exists(), root.is_dir()
    except OSError as exc:
        res.add("error", "The folder cannot be read", f"{root} — {exc.strerror or exc}")
        return _finish(res, t0)

    if not exists:
        res.add("error", "Folder not found", str(root))
        return _finish(res, t0)
    if not is_folder:
        res.add("error", "The selected path is not a folder", str(root))
        return _finish(res, t0)
    if not dspec.listdir(root) and not dspec.is_dir(root / dspec.REQUIRED_SPLIT):
        res.add("error", "The folder is empty or cannot be listed",
                "Check that you have read permission for it and that the drive is mounted.")
        return _finish(res, t0)

    # ── dataset.yaml takes priority when present ─────────────────────────
    ds_yaml = dspec.read_dataset_yaml(root)
    if ds_yaml is not None:
        res.from_yaml = True
        task = task or ds_yaml.task
        modality = modality or ds_yaml.modality
        res.classes = list(ds_yaml.classes)
    elif dspec.yaml_path(root) is not None:
        res.add("warning", "dataset.yaml could not be read",
                "The file is corrupt or missing the expected fields; the layout is being inferred "
                "from the folders instead.")

    # ── task ─────────────────────────────────────────────────────────────
    task = task or dspec.infer_task(root)
    if task is None:
        res.candidates = [str(p) for p in dspec.candidate_roots(root)]
        if res.candidates:
            res.add(
                "error", "This folder contains datasets but is not one itself",
                f"{len(res.candidates)} dataset(s) were found one level down — pick one of "
                "them below, or select its folder in the picker.",
                items=[Path(c).name for c in res.candidates],
            )
        else:
            res.add(
                "error", "The dataset structure was not recognised",
                _diagnose(root),
                items=[p.name for p in dspec.listdir(root)[:12] if not p.name.startswith(".")],
            )
        return _finish(res, t0)
    res.task = task

    # ── splits ───────────────────────────────────────────────────────────
    splits = dspec.present_splits(root)
    if dspec.REQUIRED_SPLIT not in splits:
        res.add("error", "There is no `train/` folder", "The training split is required.")
        return _finish(res, t0)

    renamed = dspec.renamed_splits(root)
    if renamed:
        res.add("info", "Split folders matched by name",
                "These folders are not named the canonical way, so they were matched "
                "case-insensitively: "
                + ", ".join(f"`{actual}/` → **{canon}**" for canon, actual in renamed.items()))

    rng = random.Random(seed)
    if task == Task.CLASSIFICATION:
        _scan_classification(root, splits, res, rng, sample_size)
    elif task == Task.SEGMENTATION:
        _scan_segmentation2d(root, splits, res, rng, sample_size)
    else:
        _scan_segmentation3d(root, splits, res, rng, sample_size)

    # ── modality ─────────────────────────────────────────────────────────
    res.modality = modality or dspec.guess_modality(root, task)

    # ── shared checks ────────────────────────────────────────────────────
    if res.needs_split:
        res.add(
            "info", "There is no validation set",
            "Only `train/` was found. You can split automatically below; the original files "
            "are not copied, only a `splits.json` is written.",
        )
    if res.n_test == 0:
        res.add("info", "There is no test set",
                "A separate test split is recommended if you are working towards publication.")
    if 0 < res.n_train < 50:
        res.add("warning", f"The training set is very small ({res.n_train} samples)",
                "Results may not be reliable; consider strong augmentation and cross-validation.")

    return _finish(res, t0)


def _finish(res: ScanResult, t0: float) -> ScanResult:
    res.elapsed = time.time() - t0
    return res


def _diagnose(root: Path) -> str:
    """Explain which part of the layout is missing, rather than restating the rule.

    "Not recognised" is useless on its own — the folder is nearly always *almost*
    right, and naming the missing piece is the difference between a fix and a
    guess.
    """
    train = dspec.split_dir(root, dspec.REQUIRED_SPLIT)
    names = [d.name for d in dspec.subdirs(root)][:12]

    if not dspec.is_dir(train):
        accepted = ", ".join(f"`{a}/`" for a in dspec.split_aliases(dspec.REQUIRED_SPLIT))
        found = f" Folders here: {', '.join(f'`{n}/`' for n in names)}." if names else                 " There are no subfolders here."
        return (f"No training folder was found. One of {accepted} is required "
                f"(the name is matched case-insensitively).{found}")

    inner = [d.name for d in dspec.subdirs(train)][:12]
    if not inner:
        n_files = len(list_files(train))
        if n_files:
            return (f"`{train.name}/` holds {n_files} file(s) directly. Put them in one "
                    "subfolder per class for classification, or in `images/` next to a "
                    "`masks/` folder for segmentation.")
        return f"`{train.name}/` is empty."

    imgs = dspec.match_dir(train, dspec.IMAGE_DIR_ALIASES)
    masks = dspec.match_dir(train, dspec.MASK_DIR_ALIASES)
    if imgs is not None and masks is None:
        accepted = ", ".join(f"`{a}/`" for a in dspec.MASK_DIR_ALIASES[:6])
        return (f"`{train.name}/{imgs.name}/` was found but there is no mask folder "
                f"beside it. One of {accepted} … is required for segmentation.")
    if masks is not None and imgs is None:
        accepted = ", ".join(f"`{a}/`" for a in dspec.IMAGE_DIR_ALIASES[:5])
        return (f"`{train.name}/{masks.name}/` was found but there is no image folder "
                f"beside it. One of {accepted} … is required.")
    return (f"`{train.name}/` contains {', '.join(f'`{n}/`' for n in inner)} — that is "
            "neither one folder per class nor an image/mask pair.")


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────


def _scan_classification(root: Path, splits: list[str], res: ScanResult,
                         rng: random.Random, sample_size: int) -> None:
    per_split_classes: dict[str, list[str]] = {}
    all_files: list[Path] = []

    unreadable: list[str] = []

    for split in splits:
        sdir = dspec.split_dir(root, split)
        class_dirs = dspec.subdirs(sdir)
        per_split_classes[split] = [d.name for d in class_dirs]

        counts: dict[str, int] = {}
        for cd in class_dirs:
            files = list_files(cd)
            counts[cd.name] = len(files)
            all_files.extend(files)
            if not files:
                # An unreadable folder and an empty one both list as nothing;
                # only the permission case is actionable, so separate them.
                if not _can_list(cd):
                    unreadable.append(f"{split}/{cd.name}")
                elif split == "train":
                    res.add("warning", f"The `{split}/{cd.name}` class is empty",
                            "Empty class folders cause errors during training.")
        res.class_counts[split] = counts
        res.split_counts[split] = sum(counts.values())

        stray = [p.name for p in dspec.listdir(sdir)
                 if not p.name.startswith(".") and not dspec.is_dir(p)][:8]
        if stray:
            res.add("warning", f"There are files outside class folders in `{split}/`",
                    "These files will be ignored.", items=stray)

    if unreadable:
        res.add("error", f"{len(unreadable)} class folder(s) cannot be read",
                "Permission denied. Fix it with, for example: "
                f"`sudo chmod -R a+rX {root}`", items=unreadable[:8])

    # class list
    train_classes = per_split_classes.get("train", [])
    if not res.classes:
        res.classes = train_classes
    if len(train_classes) < 2:
        res.add("error", "At least two classes are required",
                f"{len(train_classes)} class folders were found under `train/`.",
                items=train_classes)

    # consistency across splits
    for split, classes in per_split_classes.items():
        if split == "train":
            continue
        missing = set(train_classes) - set(classes)
        extra = set(classes) - set(train_classes)
        if missing:
            res.add("warning", f"Classes missing from the `{split}/` split",
                    "No metrics can be computed for these classes.", items=sorted(missing))
        if extra:
            res.add("error", f"The `{split}/` split has classes that are not in `train/`",
                    "The class lists must match.", items=sorted(extra))

    if res.classes and res.imbalance_ratio >= 3:
        totals = res.totals_per_class()
        hi = max(totals, key=totals.get)
        lo = min((c for c in totals if totals[c] > 0), key=lambda c: totals[c])
        res.add("warning", f"Class imbalance of {res.imbalance_ratio:.1f}×",
                f"The largest class is `{hi}` ({totals[hi]}), the smallest `{lo}` ({totals[lo]}). "
                "The recommendation engine will tune the loss function accordingly.")

    _sample_image_stats(all_files, res, rng, sample_size)
    res.preview = [
        {"image": str(p), "label": p.parent.name}
        for p in _pick(all_files, 8, rng)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2D segmentation
# ─────────────────────────────────────────────────────────────────────────────


def _scan_segmentation2d(root: Path, splits: list[str], res: ScanResult,
                         rng: random.Random, sample_size: int) -> None:
    pairs: list[tuple[Path, Path]] = []

    for split in splits:
        idir = dspec.images_dir(root, split)
        mdir = dspec.mask_dir(root, split)

        if not dspec.is_dir(idir):
            res.add("error", f"`{split}/images/` is missing", "This folder is required for segmentation.")
            res.split_counts[split] = 0
            continue
        if mdir is None:
            res.add("error", f"`{split}/masks/` is missing",
                    "The mask folder must be named `masks/`, `labels/`, `gt/`, "
                    "`annotations/` or similar.")
            res.split_counts[split] = 0
            continue
        for d in (idir, mdir):
            if not _can_list(d):
                res.add("error", f"`{d}` cannot be read",
                        f"Permission denied. Fix it with, for example: `sudo chmod -R a+rX {root}`")

        images = list_files(idir)
        masks = {p.stem: p for p in list_files(mdir)}
        matched = [(p, masks[p.stem]) for p in images if p.stem in masks]

        missing = [p.name for p in images if p.stem not in masks]
        orphan = [n for n in masks if n not in {p.stem for p in images}]
        if missing:
            res.add("error", f"{len(missing)} images in the `{split}` split have no mask",
                    "A mask file must carry the **same file name** as its image (the extension may differ).",
                    items=missing[:8])
        if orphan:
            res.add("warning", f"{len(orphan)} unmatched masks in the `{split}` split",
                    "These masks will be ignored.", items=orphan[:8])

        res.split_counts[split] = len(matched)
        pairs.extend(matched)

    if not pairs:
        res.add("error", "No matching image/mask pairs were found")
        return

    _sample_image_stats([p for p, _ in pairs], res, rng, sample_size)
    _sample_mask_stats(pairs, res, rng, sample_size)

    res.preview = [
        {"image": str(i), "mask": str(m), "label": i.stem}
        for i, m in _pick(pairs, 6, rng)
    ]


def _sample_mask_stats(pairs: list[tuple[Path, Path]], res: ScanResult,
                       rng: random.Random, sample_size: int) -> None:
    sample = _pick(pairs, sample_size, rng)
    values: set[int] = set()
    fg_ratios: list[float] = []
    mismatched: list[str] = []
    unreadable: list[str] = []

    for img_p, mask_p in sample:
        try:
            m = read_mask(mask_p)
        except (ReadError, Exception):
            unreadable.append(mask_p.name)
            continue
        vals = np.unique(m)
        values.update(int(v) for v in vals[:64])
        fg_ratios.append(float((m > 0).mean()))

        shp = image_shape(str(img_p))
        if shp and (shp[0], shp[1]) != m.shape[:2]:
            mismatched.append(f"{img_p.name} {shp[0]}×{shp[1]} ≠ mask {m.shape[0]}×{m.shape[1]}")

    if unreadable:
        res.add("error", f"{len(unreadable)} masks could not be read", items=unreadable[:8])
    if mismatched:
        res.add("error", "Image and mask dimensions do not match",
                "In segmentation the mask must be pixel-for-pixel the same size as the image.",
                items=mismatched[:8])

    res.mask_values = sorted(values)
    if fg_ratios:
        res.mask_fg_ratio = float(statistics.fmean(fg_ratios))

    _check_mask_labels(res)


def _check_mask_labels(res: ScanResult) -> None:
    """Compare the mask indices against the class list."""
    vals = [v for v in res.mask_values if v != 255]
    if not vals:
        res.add("error", "No labels were found in the masks")
        return

    if max(vals) > 200 and set(vals) - {0} and len(vals) <= 3:
        res.add(
            "warning", "The masks look binary (0/255)",
            f"Values found: {res.mask_values}. During training 255 will be mapped to 1. "
            "If you meant a multi-class mask, re-encode the values as 0, 1, 2, …",
        )
        n_needed = 2
    else:
        n_needed = max(vals) + 1
        gaps = sorted(set(range(n_needed)) - set(vals))
        if gaps:
            res.add("warning", "Some class indices never appear",
                    f"Indices missing from the sampled masks: {gaps[:10]}. "
                    "This can be normal in a small sample.")

    if not res.classes:
        res.classes = dspec.default_classes(n_needed)
        res.add("info", f"Class names were generated ({n_needed} classes)",
                "You can create a `dataset.yaml` to give them real names.")
    elif len(res.classes) < n_needed:
        res.add("error", "The class list does not match the masks",
                f"The highest index in the masks is {max(vals)} → at least {n_needed} classes are "
                f"required, but `dataset.yaml` defines {len(res.classes)}.")

    if res.mask_fg_ratio is not None and res.mask_fg_ratio < 0.005:
        res.add("info", f"The masks are very sparse (foreground {res.mask_fg_ratio * 100:.2f}%)",
                "The recommendation engine will suggest a Dice + focal loss.")


# ─────────────────────────────────────────────────────────────────────────────
# 3D segmentation
# ─────────────────────────────────────────────────────────────────────────────


def _scan_segmentation3d(root: Path, splits: list[str], res: ScanResult,
                         rng: random.Random, sample_size: int) -> None:
    pairs: list[tuple[Path, Path]] = []

    for split in splits:
        idir = dspec.images_dir(root, split)
        mdir = dspec.mask_dir(root, split)
        if not dspec.is_dir(idir) or mdir is None:
            res.add("error", f"`{split}/images/` and `{split}/labels/` must both exist")
            res.split_counts[split] = 0
            continue

        images = list_files(idir, predicate=is_volume)
        labels = {_vol_stem(p): p for p in list_files(mdir, predicate=is_volume)}
        matched = [(p, labels[_vol_stem(p)]) for p in images if _vol_stem(p) in labels]

        missing = [p.name for p in images if _vol_stem(p) not in labels]
        if missing:
            res.add("error", f"{len(missing)} volumes in the `{split}` split have no label",
                    items=missing[:8])
        res.split_counts[split] = len(matched)
        pairs.extend(matched)

    if not pairs:
        res.add("error", "No matching volume/label pairs were found")
        return

    sample = _pick(pairs, min(sample_size, 12), rng)
    shapes, spacings, values, fg = [], [], set(), []

    for vol_p, lab_p in sample:
        try:
            h = volume_header(vol_p)
            shapes.append(h["shape"])
            spacings.append(h["spacing"])
        except Exception as exc:
            res.add("warning", f"Could not read the volume header: {vol_p.name}", str(exc)[:120])
            continue
        try:
            from data.readers import read_volume

            lab, _ = read_volume(lab_p)
            v = np.unique(lab)
            values.update(int(x) for x in v[:64])
            fg.append(float((lab > 0).mean()))
            img_h = volume_header(vol_p)["shape"]
            if img_h != lab.shape:
                res.add("error", "Volume and label dimensions do not match",
                        f"{vol_p.name}: {img_h} ≠ {lab.shape}")
        except Exception as exc:
            res.add("warning", f"Could not read the label: {lab_p.name}", str(exc)[:120])

    if shapes:
        res.volume_shape = tuple(int(statistics.median(s[i] for s in shapes)) for i in range(3))
        res.median_size = (res.volume_shape[1], res.volume_shape[2])
    if spacings:
        res.volume_spacing = tuple(round(statistics.median(s[i] for s in spacings), 3) for i in range(3))
        if len(set(round(s[0], 2) for s in spacings)) > 1:
            res.add("info", "Voxel spacing varies from volume to volume",
                    "The volumes will be resampled to a common spacing before training.")

    res.mask_values = sorted(values)
    res.channels = 1
    if fg:
        res.mask_fg_ratio = float(statistics.fmean(fg))
    _check_mask_labels(res)

    res.preview = [{"image": str(i), "mask": str(m), "label": i.name}
                   for i, m in _pick(pairs, 4, rng)]


def _vol_stem(p: Path) -> str:
    """`case_001.nii.gz` → `case_001`"""
    name = p.name
    for ext in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return p.stem


# ─────────────────────────────────────────────────────────────────────────────
# Shared sampling
# ─────────────────────────────────────────────────────────────────────────────


def _can_list(directory: Path) -> bool:
    """Whether the directory can be traversed at all — tells an empty folder
    apart from one we are not allowed to read."""
    try:
        next(iter(directory.iterdir()), None)
        return True
    except OSError:
        return False


def _pick(items: list, k: int, rng: random.Random) -> list:
    if len(items) <= k:
        return list(items)
    return rng.sample(items, k)


def _sample_image_stats(files: list[Path], res: ScanResult,
                        rng: random.Random, sample_size: int) -> None:
    """Median size, channel count, intensity range and unreadable files."""
    sample = _pick(files, sample_size, rng)
    res.sampled = len(sample)
    heights, widths, chans, bad = [], [], [], []
    lows, highs = [], []
    dicom_seen = False

    for p in sample:
        shp = image_shape(str(p))
        if shp is None:
            bad.append(p.name)
            continue
        heights.append(shp[0])
        widths.append(shp[1])
        chans.append(shp[2])
        if is_dicom(p):
            dicom_seen = True

    if bad:
        res.add("error", f"{len(bad)} files could not be read",
                "Corrupt or unsupported format.", items=bad[:8])

    if heights:
        res.median_size = (int(statistics.median(heights)), int(statistics.median(widths)))
        res.size_range = ((min(heights), min(widths)), (max(heights), max(widths)))
        res.channels = int(statistics.mode(chans)) if chans else 3
        if len(set(heights)) > 1 or len(set(widths)) > 1:
            res.add("info", "Image sizes vary",
                    f"They range from {res.size_range[0][0]}×{res.size_range[0][1]} to "
                    f"{res.size_range[1][0]}×{res.size_range[1][1]}. "
                    "During training they will all be resized to the input size you choose.")

    if dicom_seen:
        # For DICOM, the intensity range matters when suggesting a window
        for p in _pick([f for f in sample if is_dicom(f)], 6, rng):
            try:
                from data.readers import read_dicom_raw

                arr, _ = read_dicom_raw(p)
                lows.append(float(np.percentile(arr, 1)))
                highs.append(float(np.percentile(arr, 99)))
            except Exception:
                continue
        if lows:
            res.intensity_range = (round(statistics.fmean(lows), 1), round(statistics.fmean(highs), 1))
