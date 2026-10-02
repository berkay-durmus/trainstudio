"""Build every dataset the UI tests need, plus two very short training runs.

Only two datasets are generated from scratch (a classification and a
segmentation one, via scripts/make_dummy_dataset.py); every other case is
derived from those by copying and renaming, which is exactly what the layouts
under test differ by. Nothing here touches a real dataset.

    python tests/ui/fixtures.py            # build what is missing
    python tests/ui/fixtures.py --force    # rebuild from scratch

The fixtures land in .ui-fixtures/ at the project root unless TS_UI_FIXTURES
says otherwise. Roughly 40 MB and, with the two training runs, a couple of
minutes on a CPU.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("TS_UI_FIXTURES") or PROJ / ".ui-fixtures")

CLS = ROOT / "ts_data" / "cls_shapes"
SEG = ROOT / "ts_data" / "seg_shapes"


def log(msg: str) -> None:
    print(f"[fixtures] {msg}")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _chmod_readable(path: Path) -> None:
    """Undo the deliberately unreadable fixtures so they can be deleted."""
    for p in [path, *path.rglob("*")]:
        try:
            p.chmod(p.stat().st_mode | stat.S_IRWXU)
        except OSError:
            pass


def base_datasets(force: bool) -> None:
    if force or not (CLS.is_dir() and SEG.is_dir()):
        log("generating the classification and segmentation datasets")
        subprocess.run([sys.executable, str(PROJ / "scripts" / "make_dummy_dataset.py"),
                        "--out", str(ROOT / "ts_data"), "--kinds", "cls", "seg",
                        "--size", "40"], check=True, cwd=PROJ)
    extra = ROOT / "extra"
    if force or not (extra / "seg3d_organ").is_dir() or not (extra / "cls_ct_dicom").is_dir():
        log("generating the 3D and DICOM datasets")
        subprocess.run([sys.executable, str(PROJ / "scripts" / "make_dummy_dataset.py"),
                        "--out", str(extra), "--kinds", "seg3d", "dicom",
                        "--size", "32"], check=True, cwd=PROJ)


def derived() -> None:
    # ── classification layouts that differ only in folder naming ────────────
    log("deriving the renamed classification layouts")
    _copy(CLS / "train", ROOT / "aliassplit" / "train")
    _copy(CLS / "val",   ROOT / "aliassplit" / "valid")          # val -> valid
    _copy(CLS / "train", ROOT / "upper" / "Train")               # capitalised
    _copy(CLS / "val",   ROOT / "upper" / "Val")
    _copy(CLS / "train", ROOT / "trainonly" / "train")           # no val/test
    _copy(CLS, ROOT / "spacedir" / "My Data Set")                # space in the path

    # train/ holding the images directly, with no class subfolders
    flat = ROOT / "flat" / "train"
    flat.mkdir(parents=True, exist_ok=True)
    for img in (CLS / "train").rglob("*.png"):
        shutil.copy2(img, flat / img.name)

    # ── segmentation layouts ────────────────────────────────────────────────
    log("deriving the segmentation layouts")
    for name in ("gtnames", "yamlme"):          # Images/ + Ground Truth/
        _copy(SEG / "train" / "images", ROOT / name / "Train" / "Images")
        _copy(SEG / "train" / "masks",  ROOT / name / "Train" / "Ground Truth")
        _copy(SEG / "val" / "images",   ROOT / name / "Validation" / "Images")
        _copy(SEG / "val" / "masks",    ROOT / name / "Validation" / "Ground Truth")

    # the transposed YOLO layout: images/<split> and labels/<split>
    _copy(SEG / "train" / "images", ROOT / "yolostyle" / "images" / "train")
    _copy(SEG / "train" / "masks",  ROOT / "yolostyle" / "labels" / "train")
    _copy(SEG / "val" / "images",   ROOT / "yolostyle" / "images" / "valid")
    _copy(SEG / "val" / "masks",    ROOT / "yolostyle" / "labels" / "valid")

    # a dataset one level down, so the parent is not itself a dataset
    _copy(SEG, ROOT / "parentdir" / "seg1")

    # ── edge cases ──────────────────────────────────────────────────────────
    log("deriving the edge cases")
    (ROOT / "edge_empty").mkdir(parents=True, exist_ok=True)
    (ROOT / "edge_afile.txt").write_text("not a dataset\n")

    one = ROOT / "edge_oneclass"                      # a single class
    _copy(CLS / "train" / "circle", one / "train" / "only")
    _copy(CLS / "val" / "circle",   one / "val" / "only")

    mism = ROOT / "edge_mismatch"                     # masks missing for most images
    _copy(SEG / "train" / "images", mism / "train" / "images")
    _copy(SEG / "val", mism / "val")
    (mism / "train" / "masks").mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(sorted((SEG / "train" / "masks").glob("*.png"))):
        if i >= 5:
            break
        shutil.copy2(m, mism / "train" / "masks" / m.name)

    # ── unreadable folders, to prove permission errors are reported ─────────
    log("deriving the unreadable folders")
    for name in ("permroot", "permclass"):
        target = ROOT / name
        if target.exists():
            _chmod_readable(target)
            shutil.rmtree(target)
        _copy(CLS, target)
    (ROOT / "permclass" / "train" / "stripe").chmod(0o000)
    (ROOT / "permroot").chmod(0o000)

    # ── messy copies for the detailed analysis and standardising ────────────
    for task, src in (("cls", CLS), ("seg", SEG)):
        dst = ROOT / "messy" / task
        if not (dst / ".built").is_file():
            log(f"deriving the messy {task} dataset")
            _messy(src, dst, task)
            (dst / ".built").touch()


def _messy(src: Path, dst: Path, task: str) -> None:
    """A copy with every problem the analysis looks for, one or a few of each:
    mixed sizes and aspect ratios, a grayscale, an RGBA and a 16-bit TIFF image,
    a uniform image, an image in both train and val, a corrupt file and — for
    segmentation — a mask whose size differs from its image."""
    import cv2
    import numpy as np

    if dst.exists():
        shutil.rmtree(dst)
    _copy(src, dst)
    imgs = sorted(p for p in dst.rglob("*.png") if "masks" not in p.parts)

    def mask_of(p: Path) -> Path:
        return p.parent.parent / "masks" / p.name

    sizes = [(346, 346), (512, 649), (640, 480), (300, 520), (128, 96)]
    for i, p in enumerate(imgs):
        w, h = sizes[i % len(sizes)]
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(p), cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA))
        if task == "seg":
            m = cv2.imread(str(mask_of(p)), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(str(mask_of(p)), cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(imgs[1]), cv2.imread(str(imgs[1]), cv2.IMREAD_GRAYSCALE))
    cv2.imwrite(str(imgs[2]), cv2.cvtColor(cv2.imread(str(imgs[2])), cv2.COLOR_BGR2BGRA))
    tif = imgs[3].with_suffix(".tif")
    cv2.imwrite(str(tif), cv2.imread(str(imgs[3]), cv2.IMREAD_GRAYSCALE).astype(np.uint16) * 257)
    imgs[3].unlink()
    if task == "seg":
        mask_of(imgs[3]).rename(mask_of(tif).with_suffix(".png"))
    cv2.imwrite(str(imgs[4]), np.full((200, 200, 3), 128, np.uint8))
    if task == "seg":
        cv2.imwrite(str(mask_of(imgs[4])), np.zeros((200, 200), np.uint8))
    train = [p for p in imgs if "train" in p.parts and p.exists()]
    dup = next(p for p in imgs if "val" in p.parts)
    shutil.copy(train[5], dup)
    if task == "seg":
        shutil.copy(mask_of(train[5]), mask_of(dup))
    train[6].write_bytes(b"\x89PNG\r\n\x1a\nnot really a png")
    if task == "seg":
        cv2.imwrite(str(mask_of(train[7])), np.zeros((50, 60), np.uint8))


def training_runs(force: bool) -> None:
    runs = ROOT / "e2e" / "runs"
    if not force and (runs / "tv-cls" / "checkpoints" / "best.pt").is_file() \
            and (runs / "tv-seg" / "checkpoints" / "best.pt").is_file():
        log("the training runs are already there")
        return
    log("training two 2-epoch models (this is the slow part)")
    env = dict(os.environ, E2E=str(ROOT / "e2e"), TS_UI_DATA=str(ROOT / "ts_data"))
    subprocess.run([sys.executable, str(Path(__file__).parent / "_train_fixtures.py")],
                   check=True, cwd=PROJ, env=env)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild everything")
    ap.add_argument("--clean", action="store_true", help="delete the fixtures and stop")
    args = ap.parse_args()

    if args.clean or args.force:
        if ROOT.exists():
            log(f"removing {ROOT}")
            _chmod_readable(ROOT)
            shutil.rmtree(ROOT)
        if args.clean:
            return 0

    ROOT.mkdir(parents=True, exist_ok=True)
    base_datasets(args.force)
    derived()
    training_runs(args.force)
    log(f"ready at {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
