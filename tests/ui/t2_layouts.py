"""UI test 2 — the Dataset page against every layout and every broken layout."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import T, dp, record, run_page, blob, messages, summary, ss_get

PAGE = "views/1_Dataset.py"

def open_with(path):
    return run_page(PAGE, lambda a: a.session_state.__setitem__(
        dp("dataset", "selected"), str(path)))

def p(*parts):
    return os.path.join(T, *parts)

# ── 1. Layouts that must be recognised ──────────────────────────────────────
print("\n1 · Layouts that must be accepted")
valid = [
    ("canonical classification", p("ts_data", "cls_shapes"),      "classification"),
    ("canonical segmentation",   p("ts_data", "seg_shapes"),      "segmentation"),
    ("alias split (train+valid)", p("aliassplit"),                "classification"),
    ("uppercase (Train/Val)",    p("upper"),                      "classification"),
    ("Images + Ground Truth",    p("gtnames"),                    "segmentation"),
    ("YOLO transposed",          p("yolostyle"),                  "segmentation"),
    ("path containing spaces",   p("spacedir", "My Data Set"),    "classification"),
    ("3D volumes",               p("extra", "seg3d_organ"),       "segmentation"),
    ("CT DICOM slices",          p("extra", "cls_ct_dicom"),      "classification"),
]
for name, path, expect_task in valid:
    at = open_with(path)
    txt = blob(at)
    exc = [e.value for e in at.exception]
    ok = not exc and "structure is valid" in txt
    detail = ""
    if not ok:
        detail = f"exc={exc[:1]} " if exc else ""
        detail += f"errors={messages(at,'error')[:1]}"
    record("layout", name, ok, detail)
    if ok:
        # the detected task must be the expected family
        got = "3d" if "3d" in txt and expect_task == "segmentation" else expect_task
        record("layout", f"  ↳ {name}: task looks like {expect_task}",
               expect_task[:5] in txt or "3d" in txt, "")

# ── 2. Broken layouts: a specific diagnosis, never a traceback ──────────────
print("\n2 · Broken layouts must diagnose, not crash")
broken = [
    ("empty folder",            p("edge_empty")),
    ("flat train/ (no classes)", p("flat")),
    ("parent of datasets",      p("parentdir")),
    ("images without masks",    p("edge_mismatch")),
    ("single class only",       p("edge_oneclass")),
]
for name, path in broken:
    at = open_with(path)
    exc = [e.value for e in at.exception]
    msgs = messages(at, "error") + messages(at, "warning") + messages(at, "info")
    ok = not exc and (len(msgs) > 0 or "structure is valid" in blob(at))
    record("broken", name, ok,
           "" if ok else (f"exc={exc[:1]}" if exc else "no message at all"))
    if not exc and msgs:
        record("broken", f"  ↳ {name}: message is specific",
               len(" ".join(msgs)) > 25, f"{msgs[0][:90]!r}")

# ── 3. Picking the parent must offer the datasets inside it ─────────────────
print("\n3 · Parent folder offers the datasets inside it")
at = open_with(p("parentdir"))
# candidate buttons are keyed cand0..candN; the picker's own subfolder
# buttons also start with a folder emoji, so match on the key instead.
cands = [b for b in at.button if b.proto.id and "cand" in b.proto.id.split("-")[-1]]
record("candidates", "child datasets offered as buttons", len(cands) >= 1,
       f"offered: {[b.label for b in cands]}")
if cands:
    at2 = cands[0].click().run()
    sel = str(ss_get(at2, dp("dataset", "selected")) or "")
    record("candidates", "clicking a candidate selects the child dataset",
           sel.rstrip("/").split("/")[-1] not in ("parentdir", ""),
           f"selected={sel}")
    record("candidates", "the selected child then scans as valid",
           "structure is valid" in blob(at2), "")

# ── 4. Train-only dataset offers the automatic split ────────────────────────
print("\n4 · A train-only dataset offers to split itself")
at = open_with(p("trainonly"))
txt = blob(at)
split_btn = [b for b in at.button if b.label and "Split" in b.label]
record("split", "the split control is offered", len(split_btn) == 1,
       f"buttons={[b.label for b in at.button if b.label]}")
if split_btn:
    at2 = split_btn[0].click().run()
    sj = os.path.join(p("trainonly"), "splits.json")
    record("split", "clicking it writes splits.json", os.path.isfile(sj),
           f"exists={os.path.isfile(sj)}")
    record("split", "the split is reported back",
           any("train" in s.lower() for s in messages(at2, "success")),
           f"{messages(at2,'success')[:1]}")
    if os.path.isfile(sj):
        os.remove(sj)          # leave the fixture as it was

# ── 5. Images directly in the split folders: one explanation, in a dialog ──
print("\n5 · Flat splits are explained once, in a dialog")
from streamlit.testing.v1 import AppTest
import core.schemas as S

for name, task in (("detected", None), ("forced to classification", S.Task.CLASSIFICATION.label)):
    def seed(a, task=task):
        a.session_state[dp("dataset", "selected")] = p("flatsplits")
        if task:
            a.session_state["force_task"] = task
    at = run_page(PAGE, seed)
    text = blob(at)
    # A phrase only the dialog uses: the issue list says "lie directly" too
    DIALOG = "file names are not read as labels"
    record("flat", f"{name}: the dialog explains it", DIALOG in text, "")
    record("flat", f"{name}: it shows what was found", "no class folders" in text, "")
    errs = messages(at, "error")
    record("flat", f"{name}: one error, not one per symptom",
           len([e for e in errs if "class" in e.lower() or "recognised" in e.lower()]) == 1
           and "outside class folders" not in text, f"{errs[:3]}")
    record("flat", f"{name}: no claim that there is no test set", "there is no test set" not in text, "")
    again = at.run()
    record("flat", f"{name}: the dialog does not reappear on the next rerun",
           DIALOG not in blob(again), "")

at = open_with(p("ts_data", "cls_shapes"))
record("flat", "a valid dataset opens no dialog", DIALOG not in blob(at), "")

sys.exit(summary("UI TEST 2 · layouts"))
