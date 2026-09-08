"""Generates synthetic datasets for development and verification.

    python scripts/make_dummy_dataset.py --out /tmp/ts_data --kinds cls seg seg3d dicom

The generated sets follow the canonical layout exactly; they are used to exercise
the application end to end without real data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _shape_image(rng, size=256, kind=0):
    """Produce a simple class-specific texture, so a model can actually learn it."""
    img = rng.normal(110, 18, (size, size, 3)).clip(0, 255)
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = rng.integers(size // 4, 3 * size // 4, 2)
    r = rng.integers(size // 10, size // 5)
    if kind == 0:                                   # circle
        m = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    elif kind == 1:                                 # square
        m = (np.abs(yy - cy) < r) & (np.abs(xx - cx) < r)
    else:                                           # diagonal stripe
        m = np.abs((yy - cy) + (xx - cx)) < r // 2
    img[m] = np.clip(img[m] + rng.normal(75, 10), 0, 255)
    return img.astype(np.uint8), m


def make_classification(out: Path, n_per_class=(140, 120, 40), size=256, seed=0):
    import cv2

    classes = ["circle", "square", "stripe"]
    rng = _rng(seed)
    for split, frac in (("train", 0.7), ("val", 0.2), ("test", 0.1)):
        for ci, (cname, total) in enumerate(zip(classes, n_per_class)):
            d = out / split / cname
            d.mkdir(parents=True, exist_ok=True)
            for i in range(max(1, int(total * frac))):
                img, _ = _shape_image(rng, size, ci)
                cv2.imwrite(str(d / f"{cname}_{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"  cls   → {out}  classes={classes} imbalance≈{max(n_per_class)/min(n_per_class):.1f}×")


def make_segmentation(out: Path, n=(140, 40, 20), size=256, seed=1):
    import cv2

    rng = _rng(seed)
    for split, count in zip(("train", "val", "test"), n):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img, m = _shape_image(rng, size, rng.integers(0, 2))
            mask = m.astype(np.uint8)                    # 0 = background, 1 = lesion
            cv2.imwrite(str(out / split / "images" / f"case_{i:04d}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / split / "masks" / f"case_{i:04d}.png"), mask)
    (out / "dataset.yaml").write_text(
        "task: segmentation\nmodality: rgb\nclasses:\n- background\n- lesion\nignore_index: 255\n",
        encoding="utf-8")
    print(f"  seg   → {out}  {sum(n)} pairs, 2 classes")


def make_dicom(out: Path, n=(80, 24, 12), size=256, seed=2):
    """CT-like DICOM slices on the HU scale (in the classification layout)."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    rng = _rng(seed)
    classes = ["healthy", "lesion"]
    for split, count in zip(("train", "val", "test"), n):
        for ci, cname in enumerate(classes):
            d = out / split / cname
            d.mkdir(parents=True, exist_ok=True)
            for i in range(count // 2):
                base = rng.normal(-50, 60, (size, size))
                yy, xx = np.mgrid[0:size, 0:size]
                body = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.42) ** 2
                vol = np.where(body, base + 90, -1000.0)
                if ci == 1:
                    cy, cx = rng.integers(size // 3, 2 * size // 3, 2)
                    r = rng.integers(12, 26)
                    les = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
                    vol[les] = rng.normal(190, 15, les.sum())

                stored = np.clip(vol + 1024, 0, 4095).astype(np.uint16)

                fm = FileMetaDataset()
                fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"   # CT Image Storage
                fm.MediaStorageSOPInstanceUID = generate_uid()
                fm.TransferSyntaxUID = ExplicitVRLittleEndian
                ds = Dataset()
                ds.file_meta = fm
                ds.SOPClassUID = fm.MediaStorageSOPClassUID
                ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
                ds.PatientID = f"SYNTHETIC_{ci}_{i:04d}"
                ds.PatientName = "Test^Synthetic"
                ds.Modality = "CT"
                ds.Rows, ds.Columns = size, size
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.BitsAllocated = 16
                ds.BitsStored = 12
                ds.HighBit = 11
                ds.PixelRepresentation = 0
                ds.RescaleIntercept = -1024.0
                ds.RescaleSlope = 1.0
                ds.WindowCenter = 40.0
                ds.WindowWidth = 400.0
                ds.PixelSpacing = [0.7, 0.7]
                ds.PixelData = stored.tobytes()
                ds.save_as(str(d / f"{cname}_{i:04d}.dcm"), enforce_file_format=True)
    print(f"  dicom → {out}  CT slices on the HU scale, window 40/400")


def make_segmentation3d(out: Path, n=(8, 3, 2), shape=(48, 96, 96), seed=3):
    import nibabel as nib

    rng = _rng(seed)
    d0, h, w = shape
    for split, count in zip(("train", "val", "test"), n):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(count):
            vol = rng.normal(-40, 45, shape).astype(np.float32)
            zz, yy, xx = np.mgrid[0:d0, 0:h, 0:w]
            cz, cy, cx = d0 // 2, rng.integers(h // 3, 2 * h // 3), rng.integers(w // 3, 2 * w // 3)
            r = rng.integers(8, 15)
            organ = ((zz - cz) / 1.6) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
            vol[organ] += 160
            lab = organ.astype(np.uint8)
            aff = np.diag([0.8, 0.8, 2.5, 1.0])
            # (D,H,W) → nibabel (X,Y,Z)
            nib.save(nib.Nifti1Image(np.transpose(vol, (2, 1, 0)), aff),
                     str(out / split / "images" / f"case_{i:03d}.nii.gz"))
            nib.save(nib.Nifti1Image(np.transpose(lab, (2, 1, 0)), aff),
                     str(out / split / "labels" / f"case_{i:03d}.nii.gz"))
    (out / "dataset.yaml").write_text(
        "task: segmentation3d\nmodality: ct\nclasses:\n- background\n- organ\nignore_index: 255\n",
        encoding="utf-8")
    print(f"  seg3d → {out}  {sum(n)} volumes, shape={shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generates synthetic test datasets")
    ap.add_argument("--out", default="/tmp/ts_data", help="the output root folder")
    ap.add_argument("--kinds", nargs="+", default=["cls", "seg"],
                    choices=["cls", "seg", "seg3d", "dicom"])
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    root = Path(args.out).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Synthetic datasets → {root}")
    if "cls" in args.kinds:
        make_classification(root / "cls_shapes", size=args.size)
    if "seg" in args.kinds:
        make_segmentation(root / "seg_shapes", size=args.size)
    if "dicom" in args.kinds:
        make_dicom(root / "cls_ct_dicom", size=args.size)
    if "seg3d" in args.kinds:
        make_segmentation3d(root / "seg3d_organ")
    print("Done.")


if __name__ == "__main__":
    main()
