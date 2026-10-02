"""UI test 7 — run management: deleting runs from every page that lists them.

Everything here writes, so it works on throwaway copies: the runs are copied out
of the shared fixtures, and TRAINSTUDIO_HOME points at a temporary folder so the
user's own ~/.trainstudio preferences are never read or written.
"""
import json, os, shutil, sys, tempfile

TMP = tempfile.mkdtemp(prefix="ts_t7_")
# Before anything imports core.prefs, which reads it once at import time
os.environ["TRAINSTUDIO_HOME"] = os.path.join(TMP, "home")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import T, PROJ, record, blob, ss_get, run_page, summary

from streamlit.testing.v1 import AppTest
import ui.state as K
from core import runs

SRC = os.path.join(T, "e2e", "runs")


def copy_run(src_name, new_name, root):
    """A private copy of a fixture run, renamed so each copy is told apart."""
    dst = os.path.join(root, new_name)
    shutil.copytree(os.path.join(SRC, src_name), dst)
    cfg_path = os.path.join(dst, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["run_name"], cfg["output_dir"] = new_name, root
    json.dump(cfg, open(cfg_path, "w"))
    return dst


def make_live(run_dir):
    """state.json says 'running' with a PID that is alive: this very process."""
    p = os.path.join(run_dir, "state.json")
    st = json.load(open(p))
    st.update(status="running", pid=os.getpid())
    json.dump(st, open(p, "w"))


def app(**state):
    at = AppTest.from_file(os.path.join(PROJ, "app.py"), default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def page(p, **state):
    return run_page(p, lambda a: [a.session_state.__setitem__(k, v) for k, v in state.items()],
                    timeout=300)


def click(at, key):
    """Click a button by key; returns the rerun app, or None if it is not there."""
    try:
        return at.button(key=key).click().run()
    except KeyError:
        return None


ROOT = os.path.join(TMP, "runs")
os.makedirs(ROOT)

# ══ 1 · Core: what may be deleted ══════════════════════════════════════════
print("\n1 · delete_run refuses what it must")
live = copy_run("tv-cls", "live-run", ROOT)
make_live(live)
ok, why = runs.delete_run(live)
record("core", "a run that is still training is not deleted",
       not ok and os.path.isdir(live), why)
ok, why = runs.delete_run(TMP)
record("core", "a folder that is not a run is not deleted",
       not ok and os.path.isdir(TMP), why)

# ══ 2 · Dashboard ══════════════════════════════════════════════════════════
print("\n2 · Deleting from the Dashboard's recent runs")
dash = copy_run("tv-cls", "dash-run", ROOT)
a = app(**{K.K_OUTPUT: ROOT, K.K_ACTIVE_RUN: dash})
record("dash", "the dashboard renders with a delete control per run", not a.exception,
       f"{[e.value for e in a.exception][:1]}")
a2 = click(a, f"del_dash_{dash}")
record("dash", "the delete button is there", a2 is not None, "")
if a2 is not None:
    record("dash", "the run folder is gone", not os.path.exists(dash), "")
    record("dash", "the active run pointing at it was cleared",
           ss_get(a2, K.K_ACTIVE_RUN) is None, f"{ss_get(a2, K.K_ACTIVE_RUN)}")
    record("dash", "the page re-renders without it", not a2.exception
           and "dash-run" not in blob(a2), "")
a3 = click(app(**{K.K_OUTPUT: ROOT}), f"del_dash_{live}")
record("dash", "a live run survives a click on its delete button",
       os.path.isdir(live), "" if a3 is not None else "button not found")

# ══ 3 · Training page ══════════════════════════════════════════════════════
print("\n3 · Deleting the watched run from the Training page")
tr = copy_run("tv-seg", "train-run", ROOT)
at = page("views/4_Training.py", **{K.K_OUTPUT: ROOT, K.K_ACTIVE_RUN: tr})
at2 = click(at, f"del_train_{tr}")
record("training", "a finished run offers a delete button", at2 is not None,
       f"{[b.key for b in at.button][:8]}")
if at2 is not None:
    record("training", "the run folder is gone", not os.path.exists(tr), "")
    # With no run left the page shows a link to Settings, which AppTest cannot
    # resolve outside st.navigation (see t4)
    _excs = [e.value for e in at2.exception]
    record("training", "the page falls back without crashing",
           all("Could not find page" in e or "url_pathname" in e for e in _excs), f"{_excs[:1]}")
at = page("views/4_Training.py", **{K.K_OUTPUT: ROOT, K.K_ACTIVE_RUN: live})
record("training", "a live run offers no delete button",
       not [b for b in at.button if b.key == f"del_train_{live}"], "")

# ══ 4 · Results ════════════════════════════════════════════════════════════
print("\n4 · Results: single and bulk delete")
r1 = copy_run("tv-cls", "res-one", ROOT)
r2 = copy_run("tv-seg", "res-two", ROOT)
at = page("views/5_Results.py", **{K.K_OUTPUT: ROOT})
record("results", "the page renders", not at.exception,
       f"{[e.value for e in at.exception][:1]}")
at = at.selectbox(key="detail_run").select(r1).run()      # not the live one
detail = [b for b in at.button if b.key and b.key.startswith("del_res_")]
record("results", "the run detail has a delete button", bool(detail), "")
if detail:
    target = detail[0].key[len("del_res_"):]
    at2 = detail[0].click().run()
    record("results", "the run shown in detail is deleted", not os.path.exists(target), target)
    record("results", "and the page re-renders", not at2.exception,
           f"{[e.value for e in at2.exception][:1]}")
# AppTest cannot tick st.data_editor checkboxes; bulk delete is the same call
others = [d for d in (r1, r2) if os.path.exists(d)]
deleted, kept = runs.delete_runs(others + [live])
record("results", "bulk delete removes the finished runs",
       sorted(deleted) == sorted(others) and not any(os.path.exists(d) for d in others), "")
record("results", "and leaves the live one, saying why",
       live in kept and os.path.isdir(live), f"{kept}")

# ══ 5 · Settings: regularisation and the fields that were hidden ═══════════
print("\n5 · Settings: dropout, stochastic depth, EMA and per-backend support")
from core.registry import get
from core.schemas import AUG_PRESETS, DatasetConfig, Modality, Task

CLS_DS = DatasetConfig(root=os.path.join(T, "ts_data", "cls_shapes"), task=Task.CLASSIFICATION,
                       modality=Modality.RGB, classes=["circle", "square", "stripe"],
                       n_train=210, n_val=60, n_test=30,
                       class_counts={"circle": 140, "square": 120, "stripe": 40},
                       median_image_size=(128, 128), channels=3)
SEG_DS = DatasetConfig(root=os.path.join(T, "ts_data", "seg_shapes"), task=Task.SEGMENTATION,
                       modality=Modality.RGB, classes=["background", "lesion"],
                       n_train=140, n_val=40, n_test=20, median_image_size=(128, 128),
                       channels=3, mask_foreground_ratio=0.08)


def settings(spec_id, ds=CLS_DS):
    return page("views/3_Settings.py", **{K.K_DATASET: ds, K.K_SPEC: get(spec_id)})


def widget(at, kind, key):
    try:
        return getattr(at, kind)(key=key)
    except KeyError:
        return None


at = settings("efficientnet_b0")
record("settings", "the page renders", not at.exception, f"{[e.value for e in at.exception][:1]}")
hp = ss_get(at, K.K_HP)
record("settings", "the monitored metric defaults to balanced accuracy",
       hp.monitor_metric == "balanced_accuracy", hp.monitor_metric)
record("settings", "a fresh page marks no field as changed",
       not ss_get(at, K.K_TOUCHED), f"{ss_get(at, K.K_TOUCHED)}")
for key in ("f_drop_rate", "f_drop_path_rate"):
    w = widget(at, "number_input", key)
    record("settings", f"{key} is offered and enabled for timm", w is not None and not w.disabled, "")
at = at.number_input(key="f_drop_rate").set_value(0.3).run()
record("settings", "a dropout value reaches the hyperparameters",
       abs(ss_get(at, K.K_HP).drop_rate - 0.3) < 1e-9, f"{ss_get(at, K.K_HP).drop_rate}")
record("settings", "and the config.json preview", '"drop_rate": 0.3' in str(at.json[0].value), "")
at = at.toggle(key="f_ema").set_value(True).run()
w = widget(at, "number_input", "f_ema_decay")
record("settings", "turning EMA on reveals its decay", w is not None, "")
if w is not None:
    at = w.set_value(0.999).run()
    record("settings", "the decay reaches the hyperparameters",
           abs(ss_get(at, K.K_HP).ema_decay - 0.999) < 1e-9, f"{ss_get(at, K.K_HP).ema_decay}")

at = settings("yolo26n_cls")
w = widget(at, "toggle", "f_ema")
record("settings", "Ultralytics: EMA is disabled (it keeps its own)", w is not None and w.disabled, "")
w = widget(at, "number_input", "f_drop_path_rate")
record("settings", "Ultralytics: no stochastic depth", w is not None and w.disabled, "")
w = widget(at, "number_input", "f_drop_rate")
record("settings", "Ultralytics classification: dropout is available", w is not None and not w.disabled, "")
w = widget(settings("tv_resnet50"), "number_input", "f_drop_rate")
record("settings", "torchvision ResNet: dropout is disabled (it has none)",
       w is not None and w.disabled, "")
at = settings("smp_fpn", SEG_DS)
hp = ss_get(at, K.K_HP)
record("settings", "smp FPN: dropout starts from the library's own 0.2",
       abs(hp.drop_rate - 0.2) < 1e-9, f"{hp.drop_rate}")
record("settings", "segmentation monitors mean Dice", hp.monitor_metric == "dice_macro", hp.monitor_metric)

at = settings("efficientnet_b0")
cur = ss_get(at, K.K_AUG)
target = next(p for p in ("light", "heavy") if p != cur.preset)
# A field whose value the two presets disagree on, so the check can fail
probe = next(f for f in ("rotate_limit", "affine_p", "brightness_p", "blur_p")
             if AUG_PRESETS[target].get(f, 0) != getattr(cur, f))
at = at.radio[0].set_value(target).run()
w = widget(at, "slider", f"f_{probe}")
want = AUG_PRESETS[target].get(probe, 0)
record("settings", f"switching the augmentation preset to {target} moves its sliders",
       w is not None and abs(w.value - want) < 1e-9 and abs(getattr(ss_get(at, K.K_AUG), probe) - want) < 1e-9,
       f"{probe}: slider={getattr(w, 'value', None)} aug={getattr(ss_get(at, K.K_AUG), probe)} want={want}")
at = at.slider(key="f_vflip").set_value(0.35).run()
record("settings", "an augmentation slider is tracked as a change",
       "vflip" in (ss_get(at, K.K_TOUCHED) or set()) and abs(ss_get(at, K.K_AUG).vflip - 0.35) < 1e-9, "")

# One model per backend: the page must render with the new fields everywhere
for spec_id, ds in [("tv_convnext_base", CLS_DS), ("smp_unet", SEG_DS), ("hf_segformer_b0", SEG_DS),
                    ("tv_lraspp_mobilenet_v3_large", SEG_DS), ("yolo26n_seg", SEG_DS)]:
    at = settings(spec_id, ds)
    record("settings", f"{spec_id}: renders", not at.exception, f"{[e.value for e in at.exception][:1]}")

# ══ 6 · Presets ════════════════════════════════════════════════════════════
print("\n6 · Presets: save, apply to another model, apply from a past run")
from core import presets

at = settings("efficientnet_b0")
rec_bs = ss_get(at, K.K_HP).batch_size
at = at.number_input(key="f_epochs").set_value(77).run()
at = at.number_input(key="f_batch_size").set_value(rec_bs + 5).run()
at = at.number_input(key="f_drop_rate").set_value(0.3).run()
at = at.toggle(key="f_ema").set_value(True).run()
at = at.number_input(key="f_ema_decay").set_value(0.999).run()
at = at.text_input(key="preset_name").set_value("t7 recipe").run()
at = at.button(key="preset_save").click().run()
saved = os.path.join(TMP, "home", "presets", "t7-recipe.json")
record("presets", "saving writes a preset file under TRAINSTUDIO_HOME",
       os.path.isfile(saved) and not at.exception, saved)
record("presets", "it is listed", [p.name for p in presets.list_presets()] == ["t7 recipe"],
       f"{[p.name for p in presets.list_presets()]}")


def load(at, option, model_specific=False):
    at = at.selectbox(key="preset_pick").select(option).run()
    if model_specific:
        at = at.checkbox(key="preset_model_specific").check().run()
    return at.button(key="preset_apply").click().run()


at = settings("tv_efficientnet_v2_s")
tv_bs = ss_get(at, K.K_HP).batch_size
at = load(at, ("preset", "t7-recipe"))
hp = ss_get(at, K.K_HP)
record("presets", "another model takes the recipe (epochs, dropout, EMA)",
       hp.epochs == 77 and abs(hp.drop_rate - 0.3) < 1e-9 and hp.ema and abs(hp.ema_decay - 0.999) < 1e-9,
       f"epochs={hp.epochs} drop={hp.drop_rate} ema={hp.ema}/{hp.ema_decay}")
record("presets", "but keeps its own batch size by default", hp.batch_size == tv_bs,
       f"{hp.batch_size} vs recommended {tv_bs}")
record("presets", "applied fields are marked as changed",
       {"epochs", "drop_rate", "ema"} <= (ss_get(at, K.K_TOUCHED) or set()), "")
record("presets", "the widgets show the applied values",
       at.number_input(key="f_epochs").value == 77, f"{at.number_input(key='f_epochs').value}")
record("presets", "what was left out is listed", any("kept as they were" in e.label for e in at.expander),
       f"{[e.label for e in at.expander][:4]}")

at = load(settings("tv_efficientnet_v2_s"), ("preset", "t7-recipe"), model_specific=True)
record("presets", "ticking the box copies the model-specific values too",
       ss_get(at, K.K_HP).batch_size == rec_bs + 5, f"{ss_get(at, K.K_HP).batch_size}")

at = settings("smp_unet", SEG_DS)
seg_before = ss_get(at, K.K_HP)
at = load(at, ("preset", "t7-recipe"))
hp = ss_get(at, K.K_HP)
record("presets", "a different task keeps its own loss and monitored metric",
       hp.loss == seg_before.loss and hp.monitor_metric == "dice_macro", f"{hp.loss} {hp.monitor_metric}")
record("presets", "and skips dropout U-Net cannot apply", hp.drop_rate == 0.0, f"{hp.drop_rate}")
record("presets", "while the shared recipe still arrives", hp.epochs == 77 and hp.ema, "")

past = copy_run("tv-cls", "preset-source", ROOT)
at = page("views/3_Settings.py", **{K.K_DATASET: CLS_DS, K.K_SPEC: get("efficientnet_b0"),
                                    K.K_OUTPUT: ROOT})
at = load(at, ("run", past))
record("presets", "a past run's configuration can be applied",
       ss_get(at, K.K_HP).epochs == runs.load_summary(past).config.hp.epochs and not at.exception,
       f"{ss_get(at, K.K_HP).epochs}")
at = at.selectbox(key="preset_pick").select(("preset", "t7-recipe")).run()
at = at.button(key="preset_delete").click().run()
record("presets", "a preset can be deleted", not os.path.exists(saved), "")

shutil.rmtree(TMP, ignore_errors=True)
sys.exit(summary("UI TEST 7 · new features"))
