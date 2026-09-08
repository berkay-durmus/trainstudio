"""Run the whole UI test suite and print one tally.

    python tests/ui/run_all.py                 # build fixtures if needed, run all
    python tests/ui/run_all.py t1 t3           # run only these parts
    python tests/ui/run_all.py --rebuild       # regenerate the fixtures first

The tests drive the real pages through streamlit.testing.v1.AppTest: they click
the actual buttons, type into the actual text boxes and read what the page
renders. Nothing is mocked, and no real dataset is touched — everything runs
against the synthetic fixtures under .ui-fixtures/.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]

PARTS = [
    ("t1", "t1_picker.py",           "the folder picker: every form of typed path"),
    ("t2", "t2_layouts.py",          "layout detection, good and broken"),
    ("t3", "t3_dataset_controls.py", "modality, windowing, class names, dataset.yaml"),
    ("t4", "t4_flow.py",             "model catalogue, settings, results, inference"),
    ("t5", "t5_navigation.py",       "navigation and a full walk-through"),
    ("t6", "t6_actions.py",          "predictions, Grad-CAM, export, comparison"),
]

TALLY = re.compile(r"^UI TEST .*?: (\d+)/(\d+) passed$", re.M)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", help="which parts to run (default: all)")
    ap.add_argument("--rebuild", action="store_true", help="regenerate the fixtures")
    ap.add_argument("--quiet", action="store_true", help="only show the tally and failures")
    args = ap.parse_args()

    build = [sys.executable, str(HERE / "fixtures.py")]
    if args.rebuild:
        build.append("--force")
    if subprocess.run(build, cwd=PROJ).returncode != 0:
        print("The fixtures could not be built.", file=sys.stderr)
        return 2

    chosen = [p for p in PARTS if not args.parts or p[0] in args.parts]
    if not chosen:
        print(f"Nothing matched. Available: {', '.join(p[0] for p in PARTS)}")
        return 2

    total = passed = 0
    failures: list[str] = []
    for key, script, what in chosen:
        print(f"\n─── {key} · {what} " + "─" * max(0, 46 - len(what)))
        r = subprocess.run([sys.executable, str(HERE / script)],
                           cwd=PROJ, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONWARNINGS="ignore"))
        out = "\n".join(l for l in r.stdout.splitlines()
                        if "ScriptRunContext" not in l)
        if args.quiet:
            print("\n".join(l for l in out.splitlines()
                            if l.startswith(("UI TEST", "FAILED:", "  · ", "  [FAIL]"))))
        else:
            print(out)
        m = TALLY.search(out)
        if not m:
            failures.append(f"{key}: the suite did not finish — {r.stderr.strip()[-300:]}")
            continue
        p, a = int(m.group(1)), int(m.group(2))
        passed += p
        total += a
        failures += [f"{key}:{l.strip()[2:]}" for l in out.splitlines() if l.startswith("  · ")]

    print("\n" + "=" * 72)
    print(f"UI SUITE: {passed}/{total} assertions passed")
    if failures:
        print(f"{len(failures)} failure(s):")
        for f in failures:
            print(f"  · {f}")
    else:
        print("No failures.")
    print("=" * 72)
    return 1 if (failures or passed != total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
