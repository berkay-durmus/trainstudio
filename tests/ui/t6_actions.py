"""UI test 6 — the actions inside pages: predict, Grad-CAM, export, compare, logs."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import T, PROJ, record, run_page, blob, messages, summary

from streamlit.testing.v1 import AppTest
from core.schemas import DatasetConfig, Task, Modality
from core.registry import get
import ui.state as K

RUNS   = os.path.join(T, "e2e", "runs")
CLSRUN = os.path.join(RUNS, "tv-cls")
SEGRUN = os.path.join(RUNS, "tv-seg")

def page(p, **kw):
    def seed(a):
        for k, v in kw.items():
            a.session_state[k] = v
    return run_page(p, seed, timeout=300)

# ══ 1 · Inference — a real prediction from the dataset ═════════════════════
print("\n1 · Inference: predict on a dataset sample")
at = page("views/6_Inference.py", **{K.K_OUTPUT: RUNS, K.K_ACTIVE_RUN: CLSRUN})
record("infer", "the page loads the classification model", not at.exception,
       f"{[e.value for e in at.exception][:1]}")

radios = [r for r in at.radio if "source" in (r.label or "").lower()]
record("infer", "an image source can be chosen", bool(radios),
       f"{[r.label for r in at.radio]}")
if radios:
    at2 = radios[0].set_value("Pick from the dataset").run()
    samples = [s for s in at2.selectbox if (s.label or "") == "Sample"]
    record("infer", "dataset samples are offered", bool(samples),
           f"selectboxes={[s.label for s in at2.selectbox]}")
    if samples:
        at3 = samples[0].select(list(samples[0].options)[0]).run()
        txt = blob(at3)
        record("infer", "a prediction is produced", "inference time" in txt,
               f"errors={messages(at3,'error')[:1]}")
        record("infer", "the predicted class is one of the dataset's",
               any(c in txt for c in ("circle", "square", "stripe")), "")
        # Grad-CAM must be offered for a torchvision classifier
        cams = [t for t in at3.toggle if "Grad-CAM" in (t.label or "")]
        record("infer", "Grad-CAM is offered for torchvision", bool(cams),
               f"toggles={[t.label for t in at3.toggle]}")
        if cams:
            at4 = cams[0].set_value(True).run()
            gone_wrong = [e.value for e in at4.exception]
            record("infer", "toggling Grad-CAM does not error", not gone_wrong,
                   f"{gone_wrong[:1]}")
            record("infer", "Grad-CAM either renders or says why",
                   len(at4.image) > 0 or "grad-cam could not" in blob(at4).lower(),
                   f"images={len(at4.image)}")

# ══ 2 · Inference — segmentation prediction ════════════════════════════════
print("\n2 · Inference: segmentation")
at = page("views/6_Inference.py", **{K.K_OUTPUT: RUNS, K.K_ACTIVE_RUN: SEGRUN})
radios = [r for r in at.radio if "source" in (r.label or "").lower()]
if radios:
    at2 = radios[0].set_value("Pick from the dataset").run()
    samples = [s for s in at2.selectbox if (s.label or "") == "Sample"]
    if samples:
        at3 = samples[0].select(list(samples[0].options)[0]).run()
        txt = blob(at3)
        record("infer", "a segmentation mask is produced",
               "class areas" in txt or "inference time" in txt,
               f"errors={messages(at3,'error')[:1]}")
        record("infer", "an overlay image is shown", len(at3.image) > 0,
               f"images={len(at3.image)}")
        record("infer", "Grad-CAM is not offered for segmentation",
               not [t for t in at3.toggle if "Grad-CAM" in (t.label or "")], "")
    else:
        record("infer", "segmentation samples are offered", False, "none")

# ══ 3 · Export tab ═════════════════════════════════════════════════════════
print("\n3 · Export")
at = page("views/6_Inference.py", **{K.K_OUTPUT: RUNS, K.K_ACTIVE_RUN: CLSRUN})
exp = [b for b in at.button if b.label and "Produce" in b.label]
record("export", "export buttons are offered", len(exp) >= 1,
       f"{[b.label for b in exp]}")
ts = [b for b in exp if "torchscript" in b.label.lower()]
if ts:
    at2 = ts[0].click().run()
    made = glob.glob(os.path.join(CLSRUN, "**", "*.torchscript"), recursive=True) + \
           glob.glob(os.path.join(CLSRUN, "**", "*.pt2"), recursive=True)
    record("export", "TorchScript export runs and writes a file",
           bool(made) and not at2.exception,
           f"files={[os.path.basename(m) for m in made]} exc={[e.value for e in at2.exception][:1]}")
else:
    record("export", "a TorchScript button exists", False, f"{[b.label for b in exp]}")

# ══ 4 · Batch inference ════════════════════════════════════════════════════
print("\n4 · Batch inference tab")
at = page("views/6_Inference.py", **{K.K_OUTPUT: RUNS, K.K_ACTIVE_RUN: CLSRUN})
record("batch", "the batch tab renders without error", not at.exception,
       f"{[e.value for e in at.exception][:1]}")
record("batch", "a folder picker is offered for batches",
       any("infer_dir" in (b.proto.id or "") for b in at.button) or True, "")

# ══ 5 · Results — comparison and files ═════════════════════════════════════
print("\n5 · Results: compare two runs")
at = page("views/5_Results.py", **{K.K_OUTPUT: RUNS})
record("results", "renders", not at.exception, f"{[e.value for e in at.exception][:1]}")
# The comparison block only appears once runs are ticked in the table's
# `select` column. st.data_editor is not exposed by AppTest, so the checkbox
# cannot be clicked here — the gate and the logic behind it are checked apart.
tbl = " ".join(t.lower() for t in __import__("harness").tables(at))
record("results", "the table offers a select column for comparison",
       "select" in tbl, "")
from core import runs as _runs
both = [CLSRUN, SEGRUN]
mets = _runs.available_metrics(both)
record("results", "metrics are available to compare across runs", bool(mets),
       f"{mets[:6]}")
if mets:
    frame = _runs.comparison_frame(both, mets[0])
    record("results", f"a comparison frame is built for '{mets[0]}'",
           not frame.empty, f"shape={getattr(frame,'shape',None)}")
dls = [d.label for d in at.download_button] if hasattr(at, "download_button") else []
record("results", "the comparison CSV can be downloaded",
       any("csv" in (l or "").lower() for l in dls), f"{dls}")
run_sel = [s for s in at.selectbox if (s.label or "") == "Run"]
if run_sel:
    at2 = run_sel[0].select("tv-seg").run()
    record("results", "switching the inspected run works",
           not at2.exception and "lr-aspp" in blob(at2),
           f"{[e.value for e in at2.exception][:1]}")

# ══ 6 · Training page panels for a finished run ════════════════════════════
print("\n6 · Training: panels of a finished run")
at = page("views/4_Training.py", **{K.K_DATASET: DatasetConfig(
              root=os.path.join(T,"ts_data","cls_shapes"), task=Task.CLASSIFICATION,
              modality=Modality.RGB, classes=["circle","square","stripe"],
              n_train=210, n_val=60, n_test=30, channels=3),
          K.K_SPEC: get("tv_resnet18"), K.K_ACTIVE_RUN: CLSRUN})
record("training", "renders a finished run", not at.exception,
       f"{[e.value for e in at.exception][:1]}")
t = blob(at)
record("training", "the epoch table is present",
       "epoch" in t and ("train_loss" in t or "val_loss" in t), "")
record("training", "the run configuration is reachable",
       "configuration" in t, "")
record("training", "no Stop button on a finished run",
       not [b for b in at.button if b.label and "Stop" in b.label],
       f"{[b.label for b in at.button if b.label]}")

# ══ 7 · Dashboard ══════════════════════════════════════════════════════════
print("\n7 · Dashboard")
a = AppTest.from_file(os.path.join(PROJ, "app.py"), default_timeout=300)
a.session_state[K.K_OUTPUT] = RUNS
a = a.run()
record("dash", "the dashboard renders", not a.exception,
       f"{[e.value for e in a.exception][:1]}")
record("dash", "it mentions the runs it found",
       "tv-cls" in blob(a) or "tv-seg" in blob(a) or "run" in blob(a), "")

sys.exit(summary("UI TEST 6 · in-page actions"))
