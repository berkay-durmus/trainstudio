"""Reading PNG/JPG/TIFF, DICOM and NIfTI behind a single interface.

All 2D readers return an `np.ndarray`:
    image → HWC, uint8 (RGB) or HW float32 (raw CT/MR, not windowed)
    mask  → HW,  int32 (class index; 0 = background)

Heavy dependencies (pydicom, nibabel, SimpleITK) are imported only when needed,
so a user who only works with PNGs does not have to install them.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from core.schemas import  IMAGE_EXTS, VOLUME_EXTS, Modality, WindowSpec


class ReadError(RuntimeError):
    """An unreadable file — the caller skips it and collects a warning."""


# ─────────────────────────────────────────────────────────────────────────────
# File type detection
# ─────────────────────────────────────────────────────────────────────────────


def suffix_of(path: Path) -> str:
    """Returns double extensions such as `.nii.gz` correctly."""
    name = path.name.lower()
    for ext in VOLUME_EXTS:
        if name.endswith(ext):
            return ext
    return path.suffix.lower()


def is_volume(path: Path) -> bool:
    return suffix_of(path) in VOLUME_EXTS


def is_plain_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_dicom(path: Path, sniff: bool = True) -> bool:
    """True if the extension looks like DICOM, or the file carries the 'DICM' magic word."""
    if path.suffix.lower() in {".dcm", ".dicom", ".ima"}:
        return True
    if not sniff or not path.is_file():
        return False
    if path.suffix.lower() in IMAGE_EXTS or is_volume(path):
        return False
    try:
        with path.open("rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except Exception:
        return False


def is_readable_image(path: Path) -> bool:
    return is_plain_image(path) or is_dicom(path)


def list_files(directory: Path, predicate=is_readable_image, recursive: bool = False) -> list[Path]:
    """Readable files in a directory, sorted by name. Hidden files are skipped.

    Never raises: a directory the app user cannot traverse (and, when walking
    recursively, any unreadable subtree inside it) yields nothing instead of an
    `EACCES` that would take the whole page down. The caller decides what an
    empty listing means.
    """
    directory = Path(directory)
    try:
        if not directory.is_dir():
            return []
    except OSError:
        return []

    out: list[Path] = []
    stack = [directory]
    while stack:
        base = stack.pop()
        try:
            entries = list(base.iterdir())
        except OSError:
            continue
        for p in entries:
            if p.name.startswith("."):
                continue
            try:
                if p.is_dir():
                    # Symlinked subtrees are not followed: a self-referential
                    # link inside a dataset would otherwise walk forever.
                    if recursive and not p.is_symlink():
                        stack.append(p)
                elif p.is_file() and predicate(p):
                    out.append(p)
            except OSError:
                continue
    return sorted(out)


# ─────────────────────────────────────────────────────────────────────────────
# DICOM
# ─────────────────────────────────────────────────────────────────────────────


def read_dicom_raw(path: Path) -> tuple[np.ndarray, dict]:
    """Return DICOM pixels as float32 converted to physical units (HU for CT)."""
    try:
        import pydicom
    except ImportError as exc:                                  # pragma: no cover
        raise ReadError("Reading DICOM requires `pydicom` (pip install pydicom)") from exc

    try:
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
    except Exception as exc:
        raise ReadError(f"Could not read DICOM: {path.name} ({exc})") from exc

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr                                    # inverted contrast

    meta = {
        "modality": str(getattr(ds, "Modality", "") or ""),
        "window_center": _first_float(getattr(ds, "WindowCenter", None)),
        "window_width": _first_float(getattr(ds, "WindowWidth", None)),
        "spacing": _pixel_spacing(ds),
        "rows": int(getattr(ds, "Rows", arr.shape[0])),
        "cols": int(getattr(ds, "Columns", arr.shape[-1])),
    }
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr[arr.shape[0] // 2]                             # multi-frame → middle frame
    return arr, meta


def _first_float(v) -> float | None:
    if v is None:
        return None
    try:
        if isinstance(v, (list, tuple)) or hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
            v = list(v)[0]
        return float(v)
    except Exception:
        return None


def _pixel_spacing(ds) -> tuple[float, float] | None:
    try:
        ps = getattr(ds, "PixelSpacing", None)
        if ps:
            return (float(ps[0]), float(ps[1]))
    except Exception:
        pass
    return None


def apply_window(arr: np.ndarray, window: WindowSpec | None) -> np.ndarray:
    """Window an array in physical units and reduce it to uint8.

    Falls back to min–max normalisation when `window` is None.
    """
    a = arr.astype(np.float32)
    if window is not None:
        lo, hi = window.bounds
    else:
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return (a * 255.0).astype(np.uint8)


# Common CT window presets — offered on the Dataset page
CT_WINDOW_PRESETS: dict[str, WindowSpec] = {
    "Soft tissue": WindowSpec(center=40, width=400),
    "Lung": WindowSpec(center=-600, width=1500),
    "Bone": WindowSpec(center=300, width=1500),
    "Brain": WindowSpec(center=40, width=80),
    "Liver": WindowSpec(center=60, width=150),
    "Mediastinum": WindowSpec(center=50, width=350),
    "Angio": WindowSpec(center=300, width=600),
}


# ─────────────────────────────────────────────────────────────────────────────
# 2D image / mask
# ─────────────────────────────────────────────────────────────────────────────


def read_image(
    path: str | Path,
    modality: Modality = Modality.RGB,
    window: WindowSpec | None = None,
    channels: int = 3,
) -> np.ndarray:
    """Return a uint8 image for training and previews (HWC, with `channels` channels)."""
    path = Path(path)

    if is_dicom(path):
        raw, meta = read_dicom_raw(path)
        win = window
        if win is None and meta.get("window_center") and meta.get("window_width"):
            win = WindowSpec(center=meta["window_center"], width=meta["window_width"])
        img = apply_window(raw, win if modality.is_medical else None)
    else:
        img = _read_plain(path)

    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if channels == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif channels == 1 and img.shape[2] == 3:
        img = (img[:, :, :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
        img = img.astype(np.uint8)[:, :, None]
    return np.ascontiguousarray(img)


def _read_plain(path: Path) -> np.ndarray:
    """PNG/JPG/TIFF — 16-bit TIFFs are reduced to uint8 as well."""
    try:
        import cv2

        flag = cv2.IMREAD_UNCHANGED
        arr = cv2.imread(str(path), flag)
        if arr is None:
            raise ReadError(f"Could not read image: {path.name}")
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB if arr.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    except ImportError:                                          # pragma: no cover
        from PIL import Image

        arr = np.array(Image.open(path))

    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        arr = np.zeros_like(a, dtype=np.uint8) if hi <= lo else \
            (((a - lo) / (hi - lo)) * 255).astype(np.uint8)
    return arr


def read_mask(path: str | Path) -> np.ndarray:
    """A class-index mask (HW, int32). Colour masks are reduced to a single channel."""
    path = Path(path)
    try:
        import cv2

        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ReadError(f"Could not read mask: {path.name}")
    except ImportError:                                          # pragma: no cover
        from PIL import Image

        arr = np.array(Image.open(path))

    if arr.ndim == 3:
        # The common case where the same value was copied into 3 channels
        if arr.shape[2] >= 3 and np.array_equal(arr[:, :, 0], arr[:, :, 1]):
            arr = arr[:, :, 0]
        else:
            arr = arr[:, :, 0]
    return arr.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# 3D volume
# ─────────────────────────────────────────────────────────────────────────────


def read_volume(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return a NIfTI/MHA/NRRD volume (D,H,W float32) plus its voxel spacing."""
    path = Path(path)
    name = path.name.lower()

    if name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
        except ImportError as exc:                               # pragma: no cover
            raise ReadError("Reading NIfTI requires `nibabel`") from exc
        try:
            img = nib.load(str(path))
            arr = np.asanyarray(img.dataobj, dtype=np.float32)
            zooms = img.header.get_zooms()[:3]
        except Exception as exc:
            raise ReadError(f"Could not read NIfTI: {path.name} ({exc})") from exc
        # nibabel (X,Y,Z) → (D,H,W)
        arr = np.transpose(arr, (2, 1, 0))
        spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
        return np.ascontiguousarray(arr), spacing

    try:
        import SimpleITK as sitk
    except ImportError as exc:                                   # pragma: no cover
        raise ReadError(f"Reading {suffix_of(path)} requires `SimpleITK`") from exc
    try:
        im = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(im).astype(np.float32)      # already (D,H,W)
        sp = im.GetSpacing()                                     # (x,y,z)
        return arr, (float(sp[2]), float(sp[1]), float(sp[0]))
    except Exception as exc:
        raise ReadError(f"Could not read volume: {path.name} ({exc})") from exc


def volume_header(path: str | Path) -> dict:
    """Shape/spacing information without loading the volume — keeps scanning fast."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib

            img = nib.load(str(path))
            shape = tuple(int(s) for s in img.shape[:3])
            zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
            return {"shape": (shape[2], shape[1], shape[0]),
                    "spacing": (zooms[2], zooms[1], zooms[0])}
        except Exception as exc:
            raise ReadError(f"Could not read NIfTI header: {path.name} ({exc})") from exc
    try:
        import SimpleITK as sitk

        r = sitk.ImageFileReader()
        r.SetFileName(str(path))
        r.ReadImageInformation()
        size = r.GetSize()
        sp = r.GetSpacing()
        return {"shape": (size[2], size[1], size[0]), "spacing": (sp[2], sp[1], sp[0])}
    except Exception as exc:
        raise ReadError(f"Could not read volume header: {path.name} ({exc})") from exc


@functools.lru_cache(maxsize=256)
def image_shape(path_str: str) -> tuple[int, int, int] | None:
    """Image size (H, W, C) — without reading the pixels where possible."""
    path = Path(path_str)
    if is_plain_image(path):
        try:
            from PIL import Image

            with Image.open(path) as im:
                w, h = im.size
                c = len(im.getbands())
            return (h, w, c)
        except Exception:
            pass
    try:
        arr = read_image(path)
        return (arr.shape[0], arr.shape[1], arr.shape[2])
    except Exception:
        return None
