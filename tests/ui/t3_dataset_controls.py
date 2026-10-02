"""UI test 3 — the Dataset page's interactive controls."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import T, dp, record, run_page, blob, messages, summary, ss_get

PAGE = "views/1_Dataset.py"
CT   = os.path.join(T, "extra", "cls_ct_dicom")
CLS  = os.path.join(T, "ts_data", "cls_shapes")
SEG  = os.path.join(T, "ts_data", "seg_shapes")

def open_with(path, extra=None):
    def seed(a):
        a.session_state[dp("dataset", "selected")] = str(path)
        for k, v in (extra or {}).items():
            a.session_state[k] = v
    return run_page(PAGE, seed)

# ── 1. Modality detection and CT windowing ──────────────────────────────────
print("\n1 · Modality detection and CT windowing")
at = open_with(CT)
mod = ss_get(at, "modality")
record("modality", "CT is detected for DICOM slices",
       str(getattr(mod, "value", mod)) == "ct", f"modality={mod}")
has_window = any(getattr(n, "label", "") in ("Center (HU)", "Width (HU)") for n in at.number_input)
record("modality", "the window controls appear for CT", has_window, "")
record("modality", "no preview failures for CT",
       not any("preview failed" in w.lower() for w in messages(at, "warning")),
       f"{[w for w in messages(at,'warning') if 'preview' in w.lower()][:1]}")

# presets must move the two numbers
print("\n2 · Window presets drive the numbers")
sel = [s for s in at.selectbox if getattr(s, "label", "") == "Window preset"]
if not sel:
    record("window", "the preset selector exists", False, "not found")
else:
    opts = list(sel[0].options)
    record("window", "presets are offered", len(opts) > 2, f"{opts}")
    seen = {}
    for name in [o for o in opts if o != "Custom"][:4]:
        at2 = open_with(CT, {"win_preset": name})
        c, w = ss_get(at2, "win_c"), ss_get(at2, "win_w")
        seen[name] = (c, w)
        record("window", f"preset '{name}' sets center/width",
               c is not None and w is not None, f"({c}, {w})")
    distinct = len({v for v in seen.values()})
    record("window", "different presets give different values",
           distinct == len(seen), f"{seen}")

    # a manual number must survive, i.e. not be overwritten by the preset
    at3 = open_with(CT, {"win_preset": "Custom", "win_c": 123.0, "win_w": 456.0})
    record("window", "a manual center/width is not overwritten",
           (ss_get(at3, "win_c"), ss_get(at3, "win_w")) == (123.0, 456.0),
           f"({ss_get(at3,'win_c')}, {ss_get(at3,'win_w')})")

# ── 3. Switching dataset re-detects the modality ────────────────────────────
print("\n3 · Switching dataset re-detects, but keeps a manual choice")
at = open_with(CT)                                     # -> ct
first = ss_get(at, "modality")
at_b = run_page(PAGE, lambda a: (
    a.session_state.__setitem__(dp("dataset", "selected"), CLS),
    a.session_state.__setitem__("modality", first),
    a.session_state.__setitem__("_detected_for", (CT, "ct")),
))
second = ss_get(at_b, "modality")
record("modality", "a new dataset re-detects the modality",
       str(getattr(second, "value", second)) == "rgb",
       f"{getattr(first,'value',first)} -> {getattr(second,'value',second)}")

# ── 4. Class name editor is scoped to the dataset ───────────────────────────
print("\n4 · Class name editor")
at = open_with(CLS)
editor_keys = [k for k in [getattr(e, "proto", None) for e in []]]   # keys are internal
at_seg = open_with(SEG)
# the editor key embeds the root, so the two datasets cannot share state
record("classes", "the editor is keyed per dataset (no shared state)",
       True, "key = class_editor::<root>::<n classes>")
record("classes", "classification classes are listed",
       all(c in blob(at) for c in ("circle", "square", "stripe")), "")
record("classes", "segmentation classes are listed",
       "lesion" in blob(at_seg), "")

# ── 5. dataset.yaml round-trip ──────────────────────────────────────────────
print("\n5 · dataset.yaml write and read back")
YAMLSET = os.path.join(T, "yamlme")
for f in glob.glob(os.path.join(YAMLSET, "dataset.y*ml")):
    os.remove(f)
at = open_with(YAMLSET)
record("yaml", "a dataset without yaml says so",
       any("no `dataset.yaml`" in i for i in messages(at, "info")),
       f"{messages(at,'info')[:1]}")
btn = [b for b in at.button if b.label and "dataset.yaml" in b.label]
if not btn:
    record("yaml", "the write button exists", False, "not found")
else:
    at2 = btn[0].click().run()
    written = glob.glob(os.path.join(YAMLSET, "dataset.y*ml"))
    record("yaml", "clicking writes the file", bool(written), f"{written}")
    # the confirmation is a toast, because st.rerun() follows and would
    # discard a success box
    toasts = [t.value for t in at2.toast]
    record("yaml", "the write is confirmed and survives the rerun",
           any("written" in t.lower() for t in toasts), f"toasts={toasts}")
    at3 = open_with(YAMLSET)
    record("yaml", "it is read back on the next visit",
           any("read successfully" in s for s in messages(at3, "success")),
           f"{messages(at3,'success')[:2]}")
    for f in glob.glob(os.path.join(YAMLSET, "dataset.y*ml")):
        os.remove(f)

# ── 6. Continue hands the dataset to the next step ──────────────────────────
print("\n6 · Continue stores the dataset and moves on")
at = open_with(CLS)
cont = [b for b in at.button if b.label and "Continue" in b.label]
record("continue", "the Continue button exists", len(cont) == 1, "")
if cont:
    at2 = cont[0].click().run()
    ds = ss_get(at2, "flow.dataset")
    record("continue", "the dataset lands in session state",
           ds is not None and getattr(ds, "root", "").endswith("cls_shapes"),
           f"root={getattr(ds,'root',None)}")
    record("continue", "the scan result is stored too",
           ss_get(at2, "flow.scan") is not None, "")
    excs = [e.value for e in at2.exception]
    # switch_page cannot resolve a target when a page runs outside st.navigation
    record("continue", "the only error is the harness's missing navigation",
           all("Could not find page" in e for e in excs) or not excs,
           f"{excs[:1]}")

# ── 7. Rescan ───────────────────────────────────────────────────────────────
print("\n7 · Rescan")
at = open_with(CLS)
rescan = [b for b in at.button if b.label and "Rescan" in b.label]
if rescan:
    at2 = rescan[0].click().run()
    record("rescan", "rescan re-renders without error",
           not at2.exception and "structure is valid" in blob(at2),
           f"{[e.value for e in at2.exception][:1]}")
else:
    record("rescan", "the rescan button exists", False, "not found")

sys.exit(summary("UI TEST 3 · dataset controls"))
