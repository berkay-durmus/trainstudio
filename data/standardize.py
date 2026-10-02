"""Write a copy of a 2D dataset in which every image has the same size.

The source is never touched: the copy goes to a new folder (by default next to
the source) that mirrors its layout file for file — same folder names, same file
names — so splits.json and dataset.yaml are copied as they are and the copy scans
exactly like the original.

Three ways to reach a size × size square:
- letterbox: scale the long side to `size`, pad the short side (keeps proportions)
- resize:    stretch to the square (what training does anyway, done once)
- crop:      scale the short side to `size`, cut the centre square out

Images keep their bit depth and channel layout (read and written with OpenCV
unchanged, never through the 8-bit training reader). Masks are resized with
nearest-neighbour through PIL, which keeps palette and index values exact; the
letterbox padding gets the ignore index, so it does not count as background —
unless 255 is itself a label (0/255 masks), in which case it gets 0.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from core.schemas import Task
from data.analysis import Analysis, collect
from data.readers import is_dicom
from data.scan import ScanResult

METHODS = {
    "letterbox": "Letterbox — keep the proportions, pad to a square",
    "resize": "Resize — stretch to a square",
    "crop": "Centre crop — fill the square, cut the edges",
}
MARKER = "standardization.json"
SIZES = (224, 256, 320, 384, 448, 512, 640, 768, 1024)


class StandardizeError(RuntimeError):
    pass


@dataclass
class Report:
    source: str
    out_root: str
    size: int
    method: str
    mask_pad: int | None
    images: int = 0
    masks: int = 0
    copied: int = 0                             # other files, carried over as they are
    unchanged: list[str] = field(default_factory=list)   # unreadable images, copied as-is
    created_at: str = ""
    elapsed: float = 0.0


def default_size(a: Analysis) -> int:
    """The median long side, rounded to a multiple of 32."""
    m = a.median_long_side or 256
    return int(min(2048, max(32, round(m / 32) * 32)))


def default_out(root: str | Path, size: int, method: str) -> Path:
    root = Path(root)
    return root.parent / f"{root.name}_std{size}_{method}"


def estimate_bytes(a: Analysis, size: int) -> int:
    """A rough size for the copy: bytes per pixel as now, times the new pixel count."""
    if not a.analysed or not a.widths:
        return 0
    pixels = sum(w * h for w, h in zip(a.widths, a.heights))
    per_pixel = a.total_bytes / max(1, pixels)
    return int(per_pixel * size * size * a.n_files * 1.1)


def check(res: ScanResult, out_root: str | Path, overwrite: bool = False,
          needed_bytes: int = 0) -> str | None:
    """Why the copy cannot be written there — or None when it can."""
    src, out = Path(res.root).resolve(), Path(out_root).expanduser().resolve()
    if res.task not in (Task.CLASSIFICATION, Task.SEGMENTATION):
        return "Standardising applies to 2D images; volumes are resampled during training."
    if out == src or src in out.parents:
        return "The copy cannot go inside the dataset itself."
    if out in src.parents:
        return "The copy cannot replace a folder that contains the dataset."
    if out.exists() and any(out.iterdir()):
        if not (out / MARKER).is_file():
            return f"`{out}` exists and is not an earlier standardised copy."
        if not overwrite:
            return f"`{out}` holds an earlier standardised copy; tick overwrite to replace it."
    parent = out.parent
    while not parent.exists():
        parent = parent.parent
    if not os.access(parent, os.W_OK):
        return f"`{parent}` is not writable."
    if needed_bytes and shutil.disk_usage(parent).free < needed_bytes:
        free = shutil.disk_usage(parent).free / 1024**3
        return f"Not enough free space: about {needed_bytes / 1024**3:.1f} GB needed, {free:.1f} GB free."
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Geometry — one function per method, shared by images and masks
# ─────────────────────────────────────────────────────────────────────────────


def _plan(w: int, h: int, size: int, method: str):
    """(scaled w, scaled h, crop box or None, pad box or None) for one image."""
    if method == "resize":
        return size, size, None, None
    scale = size / max(w, h) if method == "letterbox" else size / min(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    if method == "letterbox":
        left, top = (size - nw) // 2, (size - nh) // 2
        return nw, nh, None, (left, top)
    left, top = (nw - size) // 2, (nh - size) // 2
    return nw, nh, (left, top), None


def transform_image(img: np.ndarray, size: int, method: str) -> np.ndarray:
    h, w = img.shape[:2]
    nw, nh, crop, pad = _plan(w, h, size, method)
    interp = cv2.INTER_AREA if nw * nh < w * h else cv2.INTER_CUBIC
    out = cv2.resize(img, (nw, nh), interpolation=interp)
    if crop:
        x, y = crop
        out = out[y:y + size, x:x + size]
    if pad:
        x, y = pad
        canvas = np.zeros((size, size) + img.shape[2:], dtype=img.dtype)
        canvas[y:y + nh, x:x + nw] = out
        out = canvas
    return out


def transform_mask(path: Path, dst: Path, size: int, method: str, pad_value: int) -> None:
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        w, h = im.size
        nw, nh, crop, pad = _plan(w, h, size, method)
        out = im.resize((nw, nh), Image.Resampling.NEAREST)
        if crop:
            x, y = crop
            out = out.crop((x, y, x + size, y + size))
        if pad:
            bands = len(im.getbands())
            canvas = Image.new(im.mode, (size, size), pad_value if bands == 1 else (pad_value,) * bands)
            if im.mode == "P":
                canvas.putpalette(im.getpalette())
            canvas.paste(out, pad)
            out = canvas
        out.save(dst.with_suffix(".png") if dst.suffix.lower() not in (".png", ".tif", ".tiff") else dst)


def _write_image(src: Path, dst: Path, size: int, method: str) -> bool:
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    out = transform_image(img, size, method)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if dst.suffix.lower() in (".jpg", ".jpeg") else []
    return bool(cv2.imwrite(str(dst), out, params))


# ─────────────────────────────────────────────────────────────────────────────
# The copy
# ─────────────────────────────────────────────────────────────────────────────


def standardize(res: ScanResult, size: int, method: str, out_root: str | Path,
                overwrite: bool = False, ignore_index: int = 255,
                progress=None, workers: int | None = None) -> Report:
    """Write the standardised copy; raises StandardizeError when it cannot."""
    if method not in METHODS:
        raise StandardizeError(f"Unknown method `{method}`")
    if not 32 <= size <= 4096:
        raise StandardizeError("The size must be between 32 and 4096 px")
    problem = check(res, out_root, overwrite)
    if problem:
        raise StandardizeError(problem)

    t0 = time.perf_counter()
    # Absolute but not resolved: the file list below comes from the same, unresolved
    # root, and the two must match path for path (resolving one side of a symlinked
    # root — /var → /private/var on macOS — matched nothing, so nothing was resized)
    src = Path(os.path.abspath(res.root))
    out = Path(os.path.abspath(Path(out_root).expanduser()))
    entries = collect(res)
    if any(is_dicom(e.path) for e in entries[:50]):
        raise StandardizeError("DICOM files are not rewritten: their headers carry the "
                               "spacing and windowing training relies on.")
    images = {Path(os.path.abspath(e.path)) for e in entries}
    masks = {Path(os.path.abspath(e.mask)) for e in entries if e.mask is not None}
    # 0/255 masks use 255 as the foreground: padding with it would paint objects
    pad = None
    if res.task == Task.SEGMENTATION and method == "letterbox":
        pad = 0 if 255 in (res.mask_values or []) else ignore_index

    rep = Report(source=str(src), out_root=str(out), size=size, method=method, mask_pad=pad,
                 created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)

    jobs = []
    for p in sorted(src.rglob("*")):
        if p.is_dir() or any(part.startswith(".") for part in p.relative_to(src).parts):
            continue
        dst = partial / p.relative_to(src)
        jobs.append((p, dst, "image" if p in images else "mask" if p in masks else "copy"))
    for d in {dst.parent for _, dst, _ in jobs}:
        d.mkdir(parents=True, exist_ok=True)

    def run(job):
        p, dst, kind = job
        if kind == "image":
            if _write_image(p, dst, size, method):
                return "image", None
            shutil.copy2(p, dst)                  # unreadable: carried over as it was
            return "unchanged", str(p.relative_to(src))
        if kind == "mask":
            transform_mask(p, dst, size, method, pad if pad is not None else ignore_index)
            return "mask", None
        shutil.copy2(p, dst)
        return "copy", None

    report = progress or (lambda f, t: None)
    try:
        with ThreadPoolExecutor(max_workers=workers or min(8, (os.cpu_count() or 4))) as pool:
            for i, (kind, note) in enumerate(pool.map(run, jobs), start=1):
                if kind == "image":
                    rep.images += 1
                elif kind == "mask":
                    rep.masks += 1
                elif kind == "copy":
                    rep.copied += 1
                else:
                    rep.unchanged.append(note)
                if i % 25 == 0 or i == len(jobs):
                    report(i / max(1, len(jobs)), f"Writing · {i:,}/{len(jobs):,}")
    except Exception as exc:
        shutil.rmtree(partial, ignore_errors=True)
        raise StandardizeError(f"Writing the copy failed: {exc}") from exc

    rep.elapsed = time.perf_counter() - t0
    (partial / MARKER).write_text(json.dumps(asdict(rep), indent=2), encoding="utf-8")
    if out.exists():                              # an earlier copy, checked above
        shutil.rmtree(out)
    os.replace(partial, out)
    return rep
