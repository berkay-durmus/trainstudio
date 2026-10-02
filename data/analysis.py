"""A detailed look at a 2D dataset's images — beyond what the quick scan reports.

The scan (data/scan.py) samples a few dozen files so the Dataset page stays
instant. This reads every file's header instead — size, mode, bit depth, format
— which is cheap, plus pixels for a sample, and turns what it finds into
findings with a recommended action:

- sizes and aspect ratios, and how many images are smaller than common inputs
- mixed channel layouts, 16-bit images, mixed file formats, unreadable files
- byte-identical images, and whether any sit in two splits (leakage)
- per-channel mean / std (a "dataset" normalisation) and near-uniform images
- for segmentation: image/mask size mismatches, empty masks, class pixel shares

Nothing is written; DICOM and 3D volumes are left to the scan.
"""

from __future__ import annotations

import hashlib
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.schemas import Task
from data import spec as dspec
from data.readers import image_shape, is_dicom, is_plain_image, list_files, read_image, read_mask
from data.scan import Issue, ScanResult
from data.splitter import read_splits

COMMON_INPUT = 224          # the usual smallest model input, for the "too small" count
BITS = {"1": 1, "L": 8, "P": 8, "RGB": 8, "RGBA": 8, "CMYK": 8, "YCbCr": 8, "LA": 8,
        "I;16": 16, "I;16B": 16, "I;16L": 16, "I": 32, "F": 32}


@dataclass
class Entry:
    path: Path
    split: str
    label: str | None = None        # the class, for classification
    mask: Path | None = None        # the paired mask, for segmentation


@dataclass
class Analysis:
    task: Task
    n_files: int = 0
    analysed: int = 0                                   # files whose header was read
    truncated: bool = False                             # max_files reached
    sizes: Counter = field(default_factory=Counter)     # (w, h) → count
    widths: list[int] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)
    aspects: list[float] = field(default_factory=list)  # w / h
    orientation: Counter = field(default_factory=Counter)
    modes: Counter = field(default_factory=Counter)
    bits: Counter = field(default_factory=Counter)
    formats: Counter = field(default_factory=Counter)
    total_bytes: int = 0
    median_bytes: int = 0
    unreadable: list[str] = field(default_factory=list)
    duplicates: list[list[str]] = field(default_factory=list)   # groups of identical files
    cross_split: list[list[str]] = field(default_factory=list)  # groups spanning splits
    small: int = 0                                      # short side < COMMON_INPUT
    pixel_sample: int = 0
    channel_mean: list[float] = field(default_factory=list)     # in [0, 1]
    channel_std: list[float] = field(default_factory=list)
    uniform: list[str] = field(default_factory=list)    # near-constant images
    class_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)  # median (w, h)
    mask_mismatch: list[str] = field(default_factory=list)
    mask_sample: int = 0
    empty_masks: int = 0
    class_pixels: dict[int, float] = field(default_factory=dict)  # label → share of pixels
    findings: list[Issue] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def uniform_size(self) -> bool:
        return len(self.sizes) <= 1

    @property
    def size_span(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if not self.widths:
            return None
        return (min(self.widths), min(self.heights)), (max(self.widths), max(self.heights))

    @property
    def median_long_side(self) -> int:
        return int(statistics.median(max(w, h) for (w, h) in self.sizes.elements())) \
            if self.sizes else 0

    def add(self, level, title, detail="", items=None) -> None:
        self.findings.append(Issue(level, title, detail, list(items or [])[:8]))


# ─────────────────────────────────────────────────────────────────────────────
# Which files, in which split
# ─────────────────────────────────────────────────────────────────────────────


def collect(res: ScanResult) -> list[Entry]:
    """Every image of a 2D dataset with its split — as splits.json assigns it, if any."""
    root = Path(res.root)
    out: list[Entry] = []
    for split in dspec.present_splits(root):
        if res.task == Task.CLASSIFICATION:
            for cd in dspec.subdirs(dspec.split_dir(root, split)):
                out += [Entry(p, split, label=cd.name) for p in list_files(cd)]
        elif res.task == Task.SEGMENTATION:
            mdir = dspec.mask_dir(root, split)
            masks = {p.stem: p for p in list_files(mdir)} if mdir else {}
            out += [Entry(p, split, mask=masks.get(p.stem))
                    for p in list_files(dspec.images_dir(root, split))]
    info = read_splits(root)
    if info:
        assigned = {rel: s for s, rels in info.get("splits", {}).items() for rel in rels}
        for e in out:
            e.split = assigned.get(str(e.path.relative_to(root)), e.split)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The analysis
# ─────────────────────────────────────────────────────────────────────────────


def analyze_dataset(res: ScanResult, max_files: int = 20_000, pixel_sample: int = 200,
                    mask_sample: int = 300, progress=None, seed: int = 0) -> Analysis:
    """Analyse a scanned 2D dataset. `progress(fraction, text)` is called as it goes."""
    t0 = time.perf_counter()
    a = Analysis(task=res.task)
    if res.task not in (Task.CLASSIFICATION, Task.SEGMENTATION):
        a.add("info", "Detailed analysis covers 2D images",
              "Volumes are summarised in the Statistics tab: shape, spacing and intensity.")
        return a

    entries = collect(res)
    a.n_files = len(entries)
    if len(entries) > max_files:
        a.truncated = True
        entries = random.Random(seed).sample(entries, max_files)
    report = progress or (lambda f, t: None)

    # ── Headers: every file ──────────────────────────────────────────────
    file_bytes: list[int] = []
    per_class: dict[str, list[tuple[int, int]]] = defaultdict(list)
    readable: list[Entry] = []
    for i, e in enumerate(entries):
        if i % 200 == 0:
            report(0.6 * i / max(1, len(entries)), f"Reading headers · {i:,}/{len(entries):,}")
        info = _header(e.path)
        if info is None:
            a.unreadable.append(_rel(e.path, res.root))
            continue
        w, h, mode, fmt, nbytes = info
        readable.append(e)
        a.analysed += 1
        a.sizes[(w, h)] += 1
        a.widths.append(w)
        a.heights.append(h)
        a.aspects.append(w / h if h else 1.0)
        a.orientation["square" if w == h else "landscape" if w > h else "portrait"] += 1
        a.modes[mode] += 1
        if mode in BITS:                # DICOM bit depth is not in its mode
            a.bits[BITS[mode]] += 1
        a.formats[fmt] += 1
        file_bytes.append(nbytes)
        if min(w, h) < COMMON_INPUT:
            a.small += 1
        if e.label is not None:
            per_class[e.label].append((w, h))
        if e.mask is not None:
            m = _header(e.mask)
            if m is not None and (m[0], m[1]) != (w, h):
                a.mask_mismatch.append(f"{e.path.name}: image {w}×{h}, mask {m[0]}×{m[1]}")
    a.total_bytes = sum(file_bytes)
    a.median_bytes = int(statistics.median(file_bytes)) if file_bytes else 0
    a.class_sizes = {c: (int(statistics.median(x for x, _ in v)),
                         int(statistics.median(y for _, y in v)))
                     for c, v in sorted(per_class.items())}

    # ── Identical files: only same-sized files can be, so only those are hashed ──
    report(0.65, "Looking for identical images")
    by_size: dict[int, list[Entry]] = defaultdict(list)
    for e, n in zip(readable, file_bytes):
        by_size[n].append(e)
    groups: dict[str, list[Entry]] = defaultdict(list)
    for same in (g for g in by_size.values() if len(g) > 1):
        for e in same:
            groups[_md5(e.path)].append(e)
    for g in (g for g in groups.values() if len(g) > 1):
        # The path names the split folder; one assigned by splits.json is spelled out
        names = [r if e.split in Path(r).parts else f"{r} ({e.split})"
                 for e in g for r in [_rel(e.path, res.root)]]
        a.duplicates.append(names)
        if len({e.split for e in g}) > 1:
            a.cross_split.append(names)

    # ── Pixels: a sample ─────────────────────────────────────────────────
    rng = random.Random(seed)
    sample = rng.sample(readable, min(pixel_sample, len(readable)))
    sums, sq, n_px = np.zeros(3), np.zeros(3), 0
    for i, e in enumerate(sample):
        if i % 20 == 0:
            report(0.7 + 0.2 * i / max(1, len(sample)), f"Sampling pixels · {i}/{len(sample)}")
        try:
            img = read_image(e.path).astype(np.float64) / 255.0
        except Exception:
            continue
        flat = img.reshape(-1, 3)
        sums += flat.sum(0)
        sq += (flat ** 2).sum(0)
        n_px += len(flat)
        a.pixel_sample += 1
        if flat.std() < 2 / 255:
            a.uniform.append(_rel(e.path, res.root))
    if n_px:
        mean = sums / n_px
        a.channel_mean = [round(float(x), 4) for x in mean]
        a.channel_std = [round(float(x), 4) for x in np.sqrt(np.maximum(sq / n_px - mean ** 2, 0))]

    # ── Masks: a sample ──────────────────────────────────────────────────
    if res.task == Task.SEGMENTATION:
        paired = [e for e in readable if e.mask is not None]
        msample = rng.sample(paired, min(mask_sample, len(paired)))
        counts: Counter = Counter()
        for i, e in enumerate(msample):
            if i % 25 == 0:
                report(0.9 + 0.1 * i / max(1, len(msample)), f"Reading masks · {i}/{len(msample)}")
            try:
                m = read_mask(e.mask)
            except Exception:
                continue
            a.mask_sample += 1
            values, n = np.unique(m, return_counts=True)
            counts.update(dict(zip(values.tolist(), n.tolist())))
            if not (m > 0).any():
                a.empty_masks += 1
        total = sum(counts.values())
        a.class_pixels = {int(k): v / total for k, v in sorted(counts.items())} if total else {}

    _findings(a)
    report(1.0, "Done")
    a.elapsed = time.perf_counter() - t0
    return a


def _findings(a: Analysis) -> None:
    """Turn the numbers into findings, each with what to do about it."""
    n = max(1, a.analysed)
    if a.unreadable:
        a.add("error", f"{len(a.unreadable)} file(s) could not be opened",
              "They are replaced with blank images during training; remove or re-export them.",
              a.unreadable)
    if a.mask_mismatch:
        a.add("error", f"{len(a.mask_mismatch)} mask(s) differ in size from their image",
              "Training resizes both to the same size, which shifts the labels on these. "
              "Re-export the masks at the image size.", a.mask_mismatch)
    if a.cross_split:
        a.add("warning", f"{len(a.cross_split)} image(s) appear in more than one split",
              "The same picture in training and validation inflates the validation score. "
              "Remove one copy, or re-split the dataset.", [" = ".join(g) for g in a.cross_split])
    if len(a.sizes) > 1:
        (w0, h0), (w1, h1) = a.size_span
        a.add("warning", f"{len(a.sizes)} different image sizes, {w0}×{h0} to {w1}×{h1}",
              "Training resizes every image to one square input, stretching those whose "
              "shape differs. Standardise the sizes to keep their proportions (letterbox), "
              "and to load faster.")
    if a.aspects and max(a.aspects) / max(1e-9, min(a.aspects)) > 1.3:
        a.add("info", f"Aspect ratios range from {min(a.aspects):.2f} to {max(a.aspects):.2f}",
              "A plain resize distorts the shapes. The letterbox method pads instead.")
    if a.small:
        a.add("info", f"{a.small:,} image(s) ({a.small / n:.0%}) are under {COMMON_INPUT} px on "
              "the short side", "They are upscaled to the model input, which adds no detail; "
              "an input size near the native one is usually better.")
    if a.bits.get(16) or a.bits.get(32):
        hi = a.bits.get(16, 0) + a.bits.get(32, 0)
        a.add("warning", f"{hi:,} image(s) store more than 8 bits per pixel",
              "They are scaled to 8 bits per image when loaded, so brightness is not comparable "
              "between images. Standardising keeps the original bit depth in the copy, but "
              "training still reads 8 bits.")
    gray = any(m in ("L", "LA", "I;16", "I;16B", "I;16L", "I", "F") for m in a.modes)
    colour = any(m in ("RGB", "RGBA", "P", "CMYK", "YCbCr") for m in a.modes)
    if gray and colour:
        a.add("info", "Grayscale and colour images are mixed",
              "Every image is converted to 3 channels on loading; check that the grayscale "
              "ones really belong with the rest.", [f"{m}: {c:,}" for m, c in a.modes.items()])
    if a.modes.get("RGBA") or a.modes.get("LA"):
        a.add("info", "Some images have an alpha channel",
              "The alpha channel is dropped on loading; transparent areas become whatever "
              "colour lies underneath.")
    if len(a.formats) > 1:
        a.add("info", "Mixed file formats", ", ".join(f"{f}: {c:,}" for f, c in a.formats.items()))
    if a.uniform:
        a.add("warning", f"{len(a.uniform)} of {a.pixel_sample} sampled images are nearly uniform",
              "Blank or saturated images carry no signal; check how they were exported.", a.uniform)
    if a.duplicates and not a.cross_split:
        extra = sum(len(g) - 1 for g in a.duplicates)
        a.add("info", f"{extra} duplicate image(s) within a split",
              "Duplicates weigh those samples more heavily in training.",
              [" = ".join(g) for g in a.duplicates])
    if a.task == Task.SEGMENTATION and a.mask_sample:
        share = a.empty_masks / a.mask_sample
        if share >= 0.3:
            a.add("warning" if share >= 0.6 else "info",
                  f"{share:.0%} of the sampled masks are empty",
                  "Mostly-background data pushes the model towards predicting background; "
                  "a Dice-based loss or sampling the positive cases helps.")
    if a.channel_mean:
        a.add("info", "Per-channel statistics of this dataset",
              f"mean {a.channel_mean} · std {a.channel_std} (from {a.pixel_sample} images). "
              "If they are far from ImageNet's ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]), "
              "choose “Compute from the dataset” for normalisation in Settings.")
    if a.truncated:
        a.add("info", f"Analysed {a.analysed:,} of {a.n_files:,} files",
              "A random subset, to keep the analysis quick.")


def _header(path: Path):
    """(w, h, mode, format, bytes) from the file header, or None if it cannot be read."""
    try:
        nbytes = path.stat().st_size
        if not is_plain_image(path):
            if not is_dicom(path):
                return None
            shape = image_shape(str(path))           # DICOM has no cheap header path
            return (shape[1], shape[0], "DICOM", "DICOM", nbytes) if shape else None
        from PIL import Image

        with Image.open(path) as im:
            return im.size[0], im.size[1], im.mode, (im.format or path.suffix[1:]).upper(), nbytes
    except Exception:
        return None


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path, root) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name
