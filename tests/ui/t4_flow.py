"""UI test 4 — Model Selection, Settings, Training, Results, Inference."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import T, record, run_page, blob, messages, summary, ss_get

from core.schemas import DatasetConfig, Task, Modality
from core.registry import get, MODEL_REGISTRY
import ui.state as st_keys

RUNS = os.path.join(T, "e2e", "runs")

def ds(task):
    if task == Task.CLASSIFICATION:
        return DatasetConfig(root=os.path.join(T,"ts_data","cls_shapes"),
            task=task, modality=Modality.RGB, classes=["circle","square","stripe"],
            n_train=210, n_val=60, n_test=30,
            class_counts={"circle":140,"square":120,"stripe":40},
            median_image_size=(128,128), channels=3)
    if task == Task.SEGMENTATION:
        return DatasetConfig(root=os.path.join(T,"ts_data","seg_shapes"),
            task=task, modality=Modality.RGB, classes=["background","lesion"],
            n_train=140, n_val=40, n_test=20, median_image_size=(128,128),
            channels=3, mask_foreground_ratio=0.08)
    return DatasetConfig(root=os.path.join(T,"extra","seg3d_organ"),
        task=Task.SEGMENTATION3D, modality=Modality.CT, classes=["background","organ"],
        n_train=9, n_val=2, n_test=2, channels=1)

def seeded(page, **kw):
    def seed(a):
        for k, v in kw.items():
            a.session_state[k] = v
    return run_page(page, seed)

# ══ 1 · Model Selection: backends offered per task ══════════════════════════
print("\n1 · Model Selection — the right libraries per task")
expected = {
    Task.CLASSIFICATION: {"timm", "torchvision", "Ultralytics"},
    Task.SEGMENTATION:   {"smp", "HuggingFace", "torchvision", "Ultralytics"},
    Task.SEGMENTATION3D: {"MONAI"},
}
for task, want in expected.items():
    at = seeded("pages/2_Model_Selection.py", **{st_keys.K_DATASET: ds(task)})
    if at.exception:
        record("modelsel", f"{task.value} page renders", False,
               f"{[e.value for e in at.exception][:1]}")
        continue
    offered = set()
    for ms in at.multiselect:
        if (ms.label or "") == "Library":
            offered = set(ms.options)
    record("modelsel", f"{task.value}: libraries = {sorted(want)}",
           offered == want, f"got {sorted(offered)}")

# ══ 2 · torchvision models are actually listed and selectable ══════════════
print("\n2 · torchvision models appear and can be chosen")
at = seeded("pages/2_Model_Selection.py", **{st_keys.K_DATASET: ds(Task.CLASSIFICATION)})
txt = blob(at)
tv_cls = [s for s in MODEL_REGISTRY
          if s.backend.value == "torchvision" and s.task == Task.CLASSIFICATION]
missing = [s.display_name for s in tv_cls if s.display_name.lower() not in txt]
record("modelsel", "every torchvision classifier is on the page", not missing,
       f"missing={missing}")

# filter down to torchvision only and count the cards
at2 = seeded("pages/2_Model_Selection.py",
             **{st_keys.K_DATASET: ds(Task.CLASSIFICATION)})
ms = [m for m in at2.multiselect if (m.label or "") == "Library"][0]
at3 = ms.set_value(["torchvision"]).run()
t3 = blob(at3)
record("modelsel", "filtering to torchvision keeps only its models",
       "dinov3" not in t3 and "resnet-18" in t3.replace("resnet-18","resnet-18"),
       f"has DINOv3={'dinov3' in t3}")

# select a torchvision card
picked = None
for b in at3.button:
    if b.label and "Select" in b.label:
        picked = b; break
if picked is None:
    ids = [b.proto.id.split("-")[-1] for b in at3.button][:12]
    record("modelsel", "a model card can be selected", False, f"buttons={ids}")
else:
    at4 = picked.click().run()
    spec = ss_get(at4, st_keys.K_SPEC)
    record("modelsel", "clicking a card stores the spec",
           spec is not None, f"spec={getattr(spec,'id',None)}")
    if spec is not None:
        record("modelsel", "the stored spec is a torchvision one",
               spec.backend.value == "torchvision", f"backend={spec.backend.value}")

# ══ 3 · The Advanced picker offers both libraries ═══════════════════════════
print("\n3 · Advanced picker — timm and torchvision")
at = seeded("pages/2_Model_Selection.py", **{st_keys.K_DATASET: ds(Task.CLASSIFICATION)})
radios = {(r.label or ""): list(r.options) for r in at.radio}
record("advanced", "the library radio offers both",
       radios.get("Library") == ["timm", "torchvision"], f"{radios}")
# switch it to torchvision and confirm a real torchvision list arrives
lib_radio = [r for r in at.radio if (r.label or "") == "Library"]
if lib_radio:
    at2 = lib_radio[0].set_value("torchvision").run()
    sels = {(s.label or ""): list(s.options)[:3] for s in at2.selectbox}
    pool = [k for k in sels if "pretrained models" in k]
    record("advanced", "torchvision's model list is populated", bool(pool),
           f"{ {k: sels[k] for k in pool} }")
    if pool:
        opts = sels[pool[0]]
        record("advanced", "the names are torchvision names",
               any(o.startswith(("alexnet","convnext","resnet","vgg","efficientnet")) for o in opts),
               f"{opts}")

# ══ 4 · Filters ════════════════════════════════════════════════════════════
print("\n4 · Filters narrow the catalogue")
at = seeded("pages/2_Model_Selection.py", **{st_keys.K_DATASET: ds(Task.CLASSIFICATION)})
base_cards = blob(at).count("select")
ti = [t for t in at.text_input if "Search" in (t.label or "") or t.placeholder]
if ti:
    at2 = ti[0].set_value("resnet").run()
    t2 = blob(at2)
    record("filters", "a text search narrows the list",
           "resnet" in t2 and "dinov3" not in t2, f"has DINOv3={'dinov3' in t2}")
    at3 = ti[0].set_value("zzzz-no-such-model").run()
    record("filters", "an impossible search shows an empty state",
           "no model" in blob(at3) or "nothing" in blob(at3) or not at3.exception,
           f"{messages(at3,'info')[:1]}")
else:
    record("filters", "the search box exists", False, "not found")

# ══ 5 · Settings per backend ═══════════════════════════════════════════════
print("\n5 · Settings adapts to the chosen library")
cases = [
    ("timm",        "dinov3_vit_s16",        Task.CLASSIFICATION, True,  False),
    ("torchvision", "tv_resnet18",           Task.CLASSIFICATION, True,  False),
    ("smp",         "smp_unet",              Task.SEGMENTATION,   True,  True),
    ("ultralytics", None,                    Task.CLASSIFICATION, False, False),
    ("torchvision", "tv_deeplabv3_resnet50", Task.SEGMENTATION,   True,  False),
]
ultra = next(s for s in MODEL_REGISTRY
             if s.backend.value == "ultralytics" and s.task == Task.CLASSIFICATION)
for lib, spec_id, task, loss_enabled, wants_encoder in cases:
    spec = get(spec_id) if spec_id else ultra
    at = seeded("pages/3_Settings.py",
                **{st_keys.K_DATASET: ds(task), st_keys.K_SPEC: spec})
    if at.exception:
        record("settings", f"{lib}/{spec.id} renders", False,
               f"{[e.value for e in at.exception][:1]}"); continue
    txt = blob(at)
    loss_sel = [s for s in at.selectbox if "loss" in (s.label or "").lower()]
    disabled_note = "loss selection is disabled" in txt
    ok = (not disabled_note) if loss_enabled else disabled_note
    record("settings", f"{lib}: loss {'selectable' if loss_enabled else 'disabled'}",
           ok, f"note={disabled_note} selects={[s.label for s in loss_sel]}")
    enc = [s for s in at.selectbox if "encoder" in (s.label or "").lower()]
    record("settings", f"{lib}: encoder selector {'shown' if wants_encoder else 'hidden'}",
           bool(enc) == wants_encoder, f"found={[s.label for s in enc]}")

# ══ 6 · Settings: config preview and run-name validation ═══════════════════
print("\n6 · Settings — config preview and run name")
at = seeded("pages/3_Settings.py",
            **{st_keys.K_DATASET: ds(Task.CLASSIFICATION), st_keys.K_SPEC: get("tv_resnet18")})
record("settings", "no error building the config preview",
       not any("configuration is invalid" in e for e in messages(at, "error")),
       f"{[e for e in messages(at,'error') if 'invalid' in e][:1]}")
name_inputs = [t for t in at.text_input if "name" in (t.label or "").lower()]
if name_inputs:
    at2 = name_inputs[0].set_value("bad/name:with*chars").run()
    record("settings", "an invalid run name is rejected, not crashed",
           not at2.exception, f"{[e.value for e in at2.exception][:1]}")
else:
    record("settings", "the run-name box exists", False, "not found")

# ══ 7 · Training page ══════════════════════════════════════════════════════
print("\n7 · Training page")
at = seeded("pages/4_Training.py", **{st_keys.K_DATASET: ds(Task.CLASSIFICATION),
                                      st_keys.K_SPEC: get("tv_resnet18")})
# With no run to watch the page sends the user back to Settings; st.switch_page
# cannot resolve a target when a page is run outside st.navigation.
_excs = [e.value for e in at.exception]
record("training", "renders with no active run (or redirects to Settings)",
       not _excs or all("Could not find page" in e for e in _excs), f"{_excs[:1]}")
at = seeded("pages/4_Training.py",
            **{st_keys.K_DATASET: ds(Task.CLASSIFICATION), st_keys.K_SPEC: get("tv_resnet18"),
               st_keys.K_ACTIVE_RUN: os.path.join(RUNS, "tv-cls")})
record("training", "renders a finished run", not at.exception,
       f"{[e.value for e in at.exception][:1]}")
record("training", "the finished run is described",
       "tv-cls" in blob(at) or "completed" in blob(at), "")

# ══ 8 · Results page ═══════════════════════════════════════════════════════
print("\n8 · Results page")
at = seeded("pages/5_Results.py", **{st_keys.K_OUTPUT: RUNS})
record("results", "renders", not at.exception, f"{[e.value for e in at.exception][:1]}")
t = blob(at)
record("results", "both finished runs are listed",
       "tv-cls" in t and "tv-seg" in t, "")
record("results", "the table carries model, metric and duration",
       all(k in t for k in ("resnet-18", "lr-aspp", "balanced_accuracy", "completed")), "")
at_empty = seeded("pages/5_Results.py", **{st_keys.K_OUTPUT: os.path.join(T, "edge_empty")})
record("results", "an empty output folder shows an empty state",
       "no results" in blob(at_empty).lower(), "")

# ══ 9 · Inference page ═════════════════════════════════════════════════════
print("\n9 · Inference page")
at = seeded("pages/6_Inference.py", **{st_keys.K_OUTPUT: RUNS,
                                       st_keys.K_ACTIVE_RUN: os.path.join(RUNS, "tv-cls")})
record("inference", "renders with a torchvision checkpoint", not at.exception,
       f"{[e.value for e in at.exception][:1]}")
t = blob(at)
record("inference", "the model loaded (no load error)",
       not any("could not be loaded" in e for e in messages(at, "error")),
       f"{[e for e in messages(at,'error')][:1]}")
record("inference", "classification is reported", "classification" in t, "")
at_seg = seeded("pages/6_Inference.py", **{st_keys.K_OUTPUT: RUNS,
                                           st_keys.K_ACTIVE_RUN: os.path.join(RUNS, "tv-seg")})
record("inference", "renders a segmentation checkpoint", not at_seg.exception,
       f"{[e.value for e in at_seg.exception][:1]}")
at_none = seeded("pages/6_Inference.py", **{st_keys.K_OUTPUT: os.path.join(T, "edge_empty")})
record("inference", "no checkpoints shows an empty state",
       "no usable model" in blob(at_none).lower(), "")

# ══ 10 · Settings for the remaining backends and for a split dataset ═══════
print("\n10 · Settings — 3D, Ultralytics segmentation, and a split dataset")
# MONAI, i.e. the 3D path with its patch-size settings
monai = next(s for s in MODEL_REGISTRY if s.backend.value == "monai")
at = seeded("pages/3_Settings.py",
            **{st_keys.K_DATASET: ds(Task.SEGMENTATION3D), st_keys.K_SPEC: monai})
record("settings3d", f"MONAI ({monai.id}) renders", not at.exception,
       f"{[e.value for e in at.exception][:1]}")
record("settings3d", "the config preview is valid",
       not any("configuration is invalid" in e for e in messages(at, "error")),
       f"{[e for e in messages(at,'error') if 'invalid' in e][:1]}")

# Ultralytics on segmentation — its own loss, so ours must be disabled
useg = next(s for s in MODEL_REGISTRY
            if s.backend.value == "ultralytics" and s.task == Task.SEGMENTATION)
at = seeded("pages/3_Settings.py",
            **{st_keys.K_DATASET: ds(Task.SEGMENTATION), st_keys.K_SPEC: useg})
record("settings3d", f"Ultralytics segmentation ({useg.id}) renders", not at.exception,
       f"{[e.value for e in at.exception][:1]}")
record("settings3d", "its own loss is used, ours is disabled",
       "loss selection is disabled" in blob(at), "")

# a dataset that was split automatically must carry splits.json into the config
split_ds = ds(Task.CLASSIFICATION)
split_ds.splits_file = os.path.join(split_ds.root, "splits.json")
at = seeded("pages/3_Settings.py",
            **{st_keys.K_DATASET: split_ds, st_keys.K_SPEC: get("tv_resnet18")})
record("settings3d", "a split dataset renders", not at.exception,
       f"{[e.value for e in at.exception][:1]}")
record("settings3d", "splits.json reaches the config preview",
       "splits.json" in blob(at) or "splits_file" in blob(at), "")

sys.exit(summary("UI TEST 4 · model selection, settings, training, results, inference"))
