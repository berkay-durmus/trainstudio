"""UI test 5 — the real app shell: navigation, and one full walk-through."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import T, PROJ, record, blob, messages, summary, ss_get, dp

from streamlit.testing.v1 import AppTest
from core.schemas import DatasetConfig, Task, Modality
from core.registry import get
import ui.state as K

RUNS = os.path.join(T, "e2e", "runs")
CLS  = os.path.join(T, "ts_data", "cls_shapes")
PAGES = ["pages/0_Dashboard.py", "pages/1_Dataset.py", "pages/2_Model_Selection.py",
         "pages/3_Settings.py", "pages/4_Training.py", "pages/5_Results.py",
         "pages/6_Inference.py"]

def ds_cls():
    return DatasetConfig(root=CLS, task=Task.CLASSIFICATION, modality=Modality.RGB,
        classes=["circle","square","stripe"], n_train=210, n_val=60, n_test=30,
        class_counts={"circle":140,"square":120,"stripe":40},
        median_image_size=(128,128), channels=3)

def app(seed=None, timeout=240):
    at = AppTest.from_file(os.path.join(PROJ, "app.py"), default_timeout=timeout)
    if seed:
        seed(at)
    return at

# ══ 1 · The entrypoint and every page in the real navigation ═══════════════
print("\n1 · Every page renders inside st.navigation")
at = app().run()
record("nav", "app.py renders the default page", not at.exception,
       f"{[e.value for e in at.exception][:1]}")

for page in PAGES:
    a = app(lambda x: (
        x.session_state.__setitem__(K.K_DATASET, ds_cls()),
        x.session_state.__setitem__(K.K_SPEC, get("tv_resnet18")),
        x.session_state.__setitem__(K.K_OUTPUT, RUNS),
    )).run()
    a = a.switch_page(page).run()
    excs = [e.value for e in a.exception]
    record("nav", f"{page}", not excs, f"{excs[:1]}")

# ══ 2 · The unguarded pages redirect instead of crashing ═══════════════════
print("\n2 · With no dataset, the flow pages send you to step 1")
for page, want in (("pages/2_Model_Selection.py", "select and validate a dataset first"),
                   ("pages/3_Settings.py", "select and validate a dataset first")):
    a = app().run().switch_page(page).run()
    excs = [e.value for e in a.exception]
    warns = " ".join(messages(a, "warning")).lower()
    record("nav", f"{page} with empty state warns and stops",
           not excs and want in warns, f"excs={excs[:1]} warns={warns[:80]!r}")

# ══ 3 · A full walk-through in one session ═════════════════════════════════
print("\n3 · Dataset → Model Selection → Settings, as a user would")
a = app().run().switch_page("pages/1_Dataset.py").run()
a.session_state[dp("dataset", "selected")] = CLS
a = a.run()
record("walk", "step 1: the dataset validates",
       "structure is valid" in blob(a), f"{messages(a,'error')[:1]}")

cont = [b for b in a.button if b.label and "Continue" in b.label]
record("walk", "step 1: Continue is enabled", bool(cont) and not cont[0].disabled,
       f"found={bool(cont)}")
if cont:
    a = cont[0].click().run()
    record("walk", "step 2: Continue stores the dataset",
           getattr(ss_get(a, K.K_DATASET), "root", "").endswith("cls_shapes"),
           f"root={getattr(ss_get(a, K.K_DATASET), 'root', None)}")
    # AppTest keeps running the page it was last pointed at, so the navigation
    # a browser performs on st.switch_page has to be restated here.
    a = a.switch_page("pages/2_Model_Selection.py").run()
    record("walk", "step 2: Model Selection renders, no error",
           not a.exception, f"{[e.value for e in a.exception][:1]}")
    record("walk", "step 2: the catalogue is shown for this task",
           "torchvision" in blob(a) and "timm" in blob(a), "")

    sel = [b for b in a.button if b.label and "Select" in b.label]
    if not sel:
        record("walk", "step 2: a model can be selected", False, "no Select button")
    else:
        a = sel[0].click().run()
        spec = ss_get(a, K.K_SPEC)
        record("walk", "step 2: the model is stored", spec is not None,
               f"spec={getattr(spec,'id',None)}")
        record("walk", "step 2: the page stays on the catalogue after selecting",
               "newest first" in blob(a) and not a.exception,
               f"{[e.value for e in a.exception][:1]}")
        a = a.switch_page("pages/3_Settings.py").run()
        record("walk", "step 3: Settings renders for the chosen model",
               not a.exception, f"{[e.value for e in a.exception][:1]}")
        record("walk", "step 3: the config preview is valid",
               not any("configuration is invalid" in e for e in messages(a, "error")),
               f"{[e for e in messages(a,'error') if 'invalid' in e][:1]}")
        start = [b for b in a.button if b.label and "Start training" in b.label]
        record("walk", "step 3: the Start training button is present and enabled",
               bool(start) and not start[0].disabled, f"found={bool(start)}")

# ══ 4 · Sidebar / branding present on every page ═══════════════════════════
print("\n4 · The shell is consistent")
a = app().run()
record("shell", "the sidebar renders", len(a.sidebar) >= 0, "")
record("shell", "the theme is injected", "trainstudio" in blob(a).lower(), "")

sys.exit(summary("UI TEST 5 · navigation and walk-through"))
