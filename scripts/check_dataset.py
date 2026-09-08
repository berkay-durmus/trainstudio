"""Report what the application makes of a dataset folder, without the UI.

    python scripts/check_dataset.py /media/$USER/storage/datasets/my_set

Prints the resolved layout, the detected task, the split and class counts and
every issue the Dataset page would show. Use it when the page rejects a folder:
it separates the three things that look identical from the browser — a path that
is not what you think it is, a folder the process cannot read, and a layout that
genuinely does not match.

Exit status is 0 when the dataset is usable for training, 1 otherwise, so it can
also be used as a check in a script.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.paths import can_list, deepest_existing, normalize_input   # noqa: E402
from core.schemas import Task                                        # noqa: E402
from data import spec as dspec                                       # noqa: E402
from data.scan import scan_dataset                                   # noqa: E402


def _line(key: str, value: object) -> None:
    print(f"  {key:<22} {value}")


def _describe_path(raw: str) -> Path | None:
    """Show what the raw argument turned into — the same cleaning the UI does."""
    target = normalize_input(raw)
    print("PATH")
    _line("as given", repr(raw))
    if target is None:
        print("  → empty after cleaning; nothing to check.")
        return None
    if str(target) != raw:
        _line("after cleaning", target)

    if not target.exists():
        _line("exists", "no")
        near = deepest_existing(target.parent)
        if near is not None:
            _line("deepest existing", near)
            print(f"\n  `{near}` exists but `{target}` does not. Check the spelling of the "
                  "part after it, and that the drive is mounted.")
        return None

    _line("exists", "yes")
    _line("is a folder", "yes" if target.is_dir() else "NO — this is a file")
    if not target.is_dir():
        return None
    _line("symlink", "yes → " + str(target.resolve()) if target.is_symlink() else "no")
    if not can_list(target):
        _line("readable", "NO — permission denied")
        whoami = f"uid={os.getuid()}" if hasattr(os, "getuid") else "this user"
        print(f"\n  The process cannot list this folder. Running as {whoami}. "
              f"Grant access with: sudo chmod -R a+rX {target}")
        return None
    _line("readable", f"yes ({len(dspec.listdir(target))} entries)")
    return target


def _describe_layout(root: Path) -> None:
    print("\nLAYOUT")
    _line("kind", dspec.layout_kind(root) or "not recognised")
    for split in dspec.SPLITS:
        sdir = dspec.split_dir(root, split)
        if not dspec.is_dir(sdir) and not dspec.is_dir(dspec.images_dir(root, split)):
            _line(split, "—")
            continue
        parts = [f"{sdir.name}/"] if dspec.is_dir(sdir) else []
        imgs = dspec.images_dir(root, split)
        if dspec.is_dir(imgs):
            parts.append(f"images → {imgs.relative_to(root)}")
        masks = dspec.mask_dir(root, split)
        if masks is not None:
            parts.append(f"masks → {masks.relative_to(root)}")
        _line(split, "  ".join(parts))

    yml = dspec.yaml_path(root)
    _line("dataset.yaml", yml.name if yml else "—")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="the dataset root folder")
    ap.add_argument("--task", choices=[t.value for t in Task],
                    help="force the task instead of detecting it")
    args = ap.parse_args()

    root = _describe_path(args.path)
    if root is None:
        return 1

    _describe_layout(root)

    task = Task(args.task) if args.task else None
    res = scan_dataset(root, task=task)

    print("\nSCAN")
    _line("task", res.task.label if res.task else "not detected")
    _line("modality", res.modality.label)
    _line("samples", f"{res.n_train} train · {res.n_val} val · {res.n_test} test")
    if res.classes:
        # Only classification counts files per class; for segmentation the class
        # list comes from the mask values, so a count would always read as 0.
        totals = res.totals_per_class() if res.class_counts else {}
        _line("classes", ", ".join(
            f"{c} ({totals[c]})" if c in totals else c for c in res.classes))
    if res.median_size:
        _line("median size", f"{res.median_size[0]} × {res.median_size[1]}")
    if res.mask_values:
        _line("mask values", res.mask_values)
    _line("scanned in", f"{res.elapsed * 1000:.0f} ms ({res.sampled} files sampled)")

    if res.issues:
        print("\nISSUES")
        for issue in res.issues:
            print(f"  {issue.icon} {issue.title}")
            if issue.detail:
                print(f"       {issue.detail}")
            if issue.items:
                print(f"       {', '.join(issue.items[:8])}")

    if res.candidates:
        print("\nDATASETS ONE LEVEL DOWN — select one of these instead")
        for cand in res.candidates:
            print(f"  {cand}")

    print("\n" + ("READY — this dataset can be used for training."
                  if res.ok else "NOT READY — resolve the issues above."))
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
