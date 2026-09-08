"""The canonical dataset contract.

The user is **always** asked for the same folder layout. Conversions for backends
that expect a different format (YOLO polygon labels, for instance) are done inside
the application; the user only ever learns one layout.

    CLASSIFICATION               SEGMENTATION (2D)          SEGMENTATION (3D)
    ─────────────────────        ─────────────────────      ─────────────────────
    train/<class>/*.png          train/images/x.png         train/images/c1.nii.gz
    val/<class>/*.png            train/masks/x.png          train/labels/c1.nii.gz
    test/<class>/*.png  (opt.)   val/images/  val/masks/    val/images/  val/labels/
    dataset.yaml        (opt.)   test/... (opt.)            test/... (opt.)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from core.paths import is_dir, listdir, subdirs
from core.schemas import Modality, Task, WindowSpec
from data.readers import is_readable_image, is_volume, list_files

# ─────────────────────────────────────────────────────────────────────────────

SPLITS: tuple[str, ...] = ("train", "val", "test")
REQUIRED_SPLIT = "train"
IMAGES_DIR = "images"                    # the canonical name, used when reporting
YAML_NAME = "dataset.yaml"
YAML_ALTS = ("dataset.yaml", "dataset.yml", "data.yml", "data.yaml", "meta.yaml")

# ── Folder-name aliases ──────────────────────────────────────────────────────
# Real datasets are never named exactly the way a document says they should be:
# they arrive as `Train/`, `valid/`, `validation/`, `imgs/`, `gt/`. Matching is
# therefore done case-insensitively against a list of accepted names, so the
# canonical layout stays the *documented* one without rejecting the variants
# everybody actually has on disk. The first alias is the canonical name.
SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train", "training", "trainset", "train_set", "tr"),
    "val": ("val", "valid", "validation", "dev", "eval", "valset", "val_set"),
    "test": ("test", "testing", "testset", "test_set", "holdout", "eval_test"),
}
IMAGE_DIR_ALIASES: tuple[str, ...] = (
    "images", "image", "imgs", "img", "scans", "volumes", "vol",
)
MASK_DIR_ALIASES: tuple[str, ...] = (
    "masks", "mask", "labels", "label", "annotations", "annotation", "anns",
    "gt", "ground_truth", "groundtruth", "segmentations", "segmentation", "seg",
    "targets",
)

# The expected layout as shown in the UI
LAYOUT_HELP: dict[Task, str] = {
    Task.CLASSIFICATION: """<dataset>/
├── train/
│   ├── <class_name_1>/  *.png .jpg .tif .dcm
│   └── <class_name_2>/  ...
├── val/                 (same classes as train)
├── test/                (optional)
└── dataset.yaml         (optional — inferred from folders if absent)""",
    Task.SEGMENTATION: """<dataset>/
├── train/
│   ├── images/          patient_001.png
│   └── masks/           patient_001.png  ← same file name, uint8 index mask
├── val/
│   ├── images/
│   └── masks/
├── test/                (optional)
└── dataset.yaml         classes: [background, lesion]""",
    Task.SEGMENTATION3D: """<dataset>/
├── train/
│   ├── images/          case_001.nii.gz
│   └── labels/          case_001.nii.gz  ← same file name
├── val/
│   ├── images/
│   └── labels/
├── test/                (optional)
└── dataset.yaml         classes: [background, organ]""",
}

# Shown under the diagram: the layout above is the shape, not a spelling test.
ALIAS_HELP = (
    "Folder names are matched case-insensitively and by alias, so you do not have "
    "to rename anything: **train** also matches `Train/`, `training/`; **val** "
    "matches `valid/`, `validation/`, `dev/`; **test** matches `testing/`, "
    "`holdout/`; **images** matches `imgs/`, `scans/`, `volumes/`; **masks** "
    "matches `labels/`, `gt/`, `annotations/`, `segmentations/`. The transposed "
    "YOLO order (`images/train/` + `labels/train/`) is read as well."
)


# ─────────────────────────────────────────────────────────────────────────────
# dataset.yaml
# ─────────────────────────────────────────────────────────────────────────────


class DatasetYaml(BaseModel):
    """The optional description file that sits at the dataset root."""

    model_config = ConfigDict(extra="allow")

    task: Task
    modality: Modality = Modality.RGB
    classes: list[str] = Field(default_factory=list)
    ignore_index: int = 255
    window: WindowSpec | None = None
    description: str = ""

    def to_yaml(self) -> str:
        data: dict[str, Any] = {
            "task": self.task.value,
            "modality": self.modality.value,
            "classes": list(self.classes),
            "ignore_index": self.ignore_index,
        }
        if self.window is not None:
            data["window"] = {"center": self.window.center, "width": self.window.width}
        if self.description:
            data["description"] = self.description
        header = (
            "# TrainStudio dataset description\n"
            "# This file is optional; without it the layout is inferred from the folders.\n"
        )
        return header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def yaml_path(root: str | Path) -> Path | None:
    """The description file at the root, matched case-insensitively; None if absent."""
    root = Path(root)
    for name in YAML_ALTS:
        p = root / name
        try:
            if p.is_file():
                return p
        except OSError:
            return None

    wanted = {_norm(n): i for i, n in enumerate(YAML_ALTS)}
    hits: list[tuple[int, Path]] = []
    for f in listdir(root):
        rank = wanted.get(_norm(f.name))
        if rank is not None and not is_dir(f):
            hits.append((rank, f))
    return min(hits, key=lambda t: t[0])[1] if hits else None


def read_dataset_yaml(root: str | Path) -> DatasetYaml | None:
    """Read the description file at the root. Returns None if it is broken (the
    scan still works)."""
    p = yaml_path(root)
    if p is None:
        return None
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if isinstance(raw.get("classes"), dict):
            # Accept the YOLO style {0: 'bg', 1: 'lesion'} as well
            raw["classes"] = [raw["classes"][k] for k in sorted(raw["classes"])]
        return DatasetYaml.model_validate(raw)
    except Exception:
        return None


def write_dataset_yaml(root: str | Path, spec: DatasetYaml) -> Path:
    """Write (or overwrite) the description file, reusing whatever name is
    already there so a dataset does not end up with two of them."""
    target = yaml_path(root) or Path(root) / YAML_NAME
    target.write_text(spec.to_yaml(), encoding="utf-8")
    return target


# ─────────────────────────────────────────────────────────────────────────────
# Inferring the task from the structure
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Resolving the layout on disk
# ─────────────────────────────────────────────────────────────────────────────


def _norm(name: str) -> str:
    """`Ground Truth` / `ground-truth` / `ground_truth` all collapse to the same key."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def match_dir(base: str | Path, aliases: Sequence[str]) -> Path | None:
    """The first subdirectory of `base` whose name matches one of `aliases`.

    Matching goes exact → case-insensitive → punctuation-insensitive, so the
    documented layout keeps working unchanged while `Train/`, `VALID/` and
    `Ground Truth/` are accepted too. Earlier aliases win, which is what makes
    the alias tuples an explicit order of preference rather than a set.
    """
    base = Path(base)
    for name in aliases:                                  # fast path, no listing
        p = base / name
        if is_dir(p):
            return p

    present = subdirs(base)
    if not present:
        return None
    exact = {d.name.lower(): d for d in present}
    loose: dict[str, Path] = {}
    for d in present:                       # first writer wins → stable ordering
        loose.setdefault(_norm(d.name), d)

    for name in aliases:
        hit = exact.get(name.lower()) or loose.get(_norm(name))
        if hit is not None:
            return hit
    return None


def split_aliases(split: str) -> tuple[str, ...]:
    return SPLIT_ALIASES.get(split, (split,))


def split_dir(root: str | Path, split: str) -> Path:
    """The folder holding `split`, whatever it is actually called on disk.

    Falls back to the canonical `<root>/<split>` when there is no match, so the
    call sites can keep asking `.is_dir()` and get a straight "no".
    """
    base = Path(root)
    return match_dir(base, split_aliases(split)) or base / split


def images_dir(root: str | Path, split: str) -> Path:
    """The image folder for a split, in either of the two layouts in the wild:
    `<root>/<split>/images` (canonical) or `<root>/images/<split>` (YOLO-style)."""
    sdir = split_dir(root, split)
    nested = match_dir(sdir, IMAGE_DIR_ALIASES)
    if nested is not None:
        return nested

    top = match_dir(root, IMAGE_DIR_ALIASES)
    if top is not None:
        transposed = match_dir(top, split_aliases(split))
        if transposed is not None:
            return transposed
    return sdir / IMAGES_DIR


def mask_dir(root: str | Path, split: str) -> Path | None:
    """The mask/label folder for a split — `masks/`, `labels/`, `gt/`, … — in
    either layout. `None` when there is none."""
    nested = match_dir(split_dir(root, split), MASK_DIR_ALIASES)
    if nested is not None:
        return nested

    top = match_dir(root, MASK_DIR_ALIASES)
    if top is not None:
        return match_dir(top, split_aliases(split))
    return None


def layout_kind(root: str | Path) -> str | None:
    """`"nested"`, `"transposed"` or `None` — used only for explaining things
    back to the user."""
    if is_dir(split_dir(root, REQUIRED_SPLIT)):
        return "nested"
    if is_dir(images_dir(root, REQUIRED_SPLIT)):
        return "transposed"
    return None


def present_splits(root: str | Path) -> list[str]:
    """The canonical splits that exist, in `train, val, test` order."""
    return [
        s for s in SPLITS
        if is_dir(split_dir(root, s)) or is_dir(images_dir(root, s))
    ]


def renamed_splits(root: str | Path) -> dict[str, str]:
    """Splits whose folder is not called by its canonical name — reported so the
    user can see *which* folder was picked up as what."""
    out: dict[str, str] = {}
    for s in present_splits(root):
        actual = split_dir(root, s)
        if is_dir(actual) and actual.name.lower() != s:
            out[s] = actual.name
    return out


def infer_task(root: str | Path) -> Task | None:
    """Infer the task from the folder structure.

    An image folder plus a mask folder means segmentation; if the files inside
    are volumes it is 3D. Otherwise the subfolders of the train split are
    treated as classes.
    """
    root = Path(root)
    imgs = images_dir(root, REQUIRED_SPLIT)
    masks = mask_dir(root, REQUIRED_SPLIT)
    if is_dir(imgs) and masks is not None:
        sample = list_files(imgs, predicate=lambda p: is_readable_image(p) or is_volume(p))[:8]
        if sample and all(is_volume(p) for p in sample):
            return Task.SEGMENTATION3D
        return Task.SEGMENTATION

    train = split_dir(root, REQUIRED_SPLIT)
    if not is_dir(train):
        return None
    if subdirs(train):
        return Task.CLASSIFICATION
    return None


def looks_like_dataset(root: str | Path) -> bool:
    return infer_task(root) is not None


def candidate_roots(root: str | Path, limit: int = 8) -> list[Path]:
    """Child folders that are themselves dataset roots.

    Selecting the folder that *contains* the datasets instead of a dataset is
    the single most common mis-click, and it is recoverable: offer the children
    rather than only reporting that this folder is not a dataset.
    """
    return [d for d in subdirs(root) if looks_like_dataset(d)][:limit]


def infer_classes_from_folders(root: str | Path) -> list[str]:
    """In classification, class names are the subfolder names under the train split."""
    return [d.name for d in subdirs(split_dir(root, REQUIRED_SPLIT))]


def guess_modality(root: str | Path, task: Task) -> Modality:
    """Guess the modality by looking at a few sample files."""
    from data.readers import is_dicom, read_dicom_raw

    root = Path(root)
    if task == Task.CLASSIFICATION:
        folders = subdirs(split_dir(root, REQUIRED_SPLIT))
        sample = [p for d in folders[:3] for p in list_files(d)[:3]]
    else:
        d = images_dir(root, REQUIRED_SPLIT)
        sample = list_files(d, predicate=lambda p: is_readable_image(p) or is_volume(p))[:6]

    if not sample:
        return Modality.RGB

    if task == Task.SEGMENTATION3D:
        return Modality.CT

    dicoms = [p for p in sample if is_dicom(p)]
    if dicoms:
        try:
            _, meta = read_dicom_raw(dicoms[0])
            mod = (meta.get("modality") or "").upper()
            if mod.startswith("CT"):
                return Modality.CT
            if mod.startswith("MR"):
                return Modality.MR
        except Exception:
            pass
        return Modality.CT

    from data.readers import image_shape

    shapes = [image_shape(str(p)) for p in sample[:4]]
    channels = [s[2] for s in shapes if s]
    if channels and max(channels) == 1:
        return Modality.GRAYSCALE
    return Modality.RGB


def default_classes(n: int) -> list[str]:
    """Placeholders used when class names cannot be derived from the mask values."""
    return ["background"] + [f"class_{i}" for i in range(1, n)]
