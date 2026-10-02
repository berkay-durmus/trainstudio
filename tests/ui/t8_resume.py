"""UI test 8 — pausing, stopping and resuming a run.

These train for real: a small timm model on the synthetic classification set, on
the CPU, a few seconds per epoch. The runs live in a temporary folder and
TRAINSTUDIO_HOME points there too, as in t7.
"""
import json, os, shutil, signal, sys, tempfile, time

# Resolved, as RunConfig resolves output_dir: run paths then match the run list's
TMP = os.path.realpath(tempfile.mkdtemp(prefix="ts_t8_"))
os.environ["TRAINSTUDIO_HOME"] = os.path.join(TMP, "home")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import T, PROJ, record, blob, run_page, summary

from streamlit.testing.v1 import AppTest

import pandas as pd
import ui.state as K
from core import launcher, runs
from core.events import MetricAccumulator, read_all_events
from core.hardware import detect
from core.recommend import recommend
from core.registry import get
from core.schemas import DatasetConfig, Layout, ModelSelection, RunConfig, RunStatus, Task
from data.scan import scan_dataset

DATA = os.path.join(T, "ts_data", "cls_shapes")
ROOT = os.path.join(TMP, "runs")
os.makedirs(ROOT)
EPOCHS = 3


def config(name, epochs=EPOCHS):
    spec = get("efficientnet_b0")
    res = scan_dataset(DATA, task=Task.CLASSIFICATION)
    ds = DatasetConfig(root=DATA, task=res.task, modality=res.modality, classes=res.classes,
                       n_train=res.n_train, n_val=res.n_val, n_test=res.n_test,
                       class_counts=res.totals_per_class(), median_image_size=res.median_size,
                       channels=3)
    rec = recommend(spec, ds, detect())
    hp = rec.hp
    hp.epochs, hp.batch_size, hp.img_size = epochs, 4, 64
    hp.num_workers, hp.amp, hp.channels_last = 0, False, False
    hp.early_stopping, hp.preview_every_n_epochs = False, 1
    hp.pretrained, hp.deterministic, hp.ema, hp.log_every_n_steps = False, True, True, 1
    return RunConfig(
        run_name=name, output_dir=ROOT, created_at=time.strftime("%F %T"), dataset=ds,
        model=ModelSelection(spec_id=spec.id, backend=spec.backend, arch=spec.arch,
                             display_name=spec.display_name),
        hp=hp, aug=rec.aug, device="cpu")


def wait(run_dir, on_tick=None, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = runs.load_summary(run_dir)
        if on_tick:
            on_tick(s)
        if s.status.is_terminal and not runs.process_alive(s.pid):
            return s
        time.sleep(0.3)
    raise TimeoutError(run_dir)


def metrics(run_dir):
    return pd.read_csv(os.path.join(run_dir, Layout.METRICS_CSV))


def epochs_seen(run_dir):
    acc = MetricAccumulator()
    acc.feed(read_all_events(os.path.join(run_dir, Layout.EVENTS)))
    return [r["epoch"] for r in acc.epochs]


def batch_seen(run_dir, epoch):
    return any(e.get("type") == "batch" and e.get("epoch") == epoch
               for e in read_all_events(os.path.join(run_dir, Layout.EVENTS)))


def page(p, **state):
    return run_page(p, lambda a: [a.session_state.__setitem__(k, v) for k, v in state.items()],
                    timeout=300)


def has_button(at, key):
    try:
        at.button(key=key)
        return True
    except KeyError:
        return False


# ══ 1 · Pause, then resume: the same run as an uninterrupted one ═══════════
print("\n1 · Pause after an epoch, resume — identical to an uninterrupted run")
a_cfg = config("straight")
launcher.start_run(a_cfg)
a = wait(a_cfg.run_dir)
record("pause", "the uninterrupted run completes", a.status == RunStatus.COMPLETED, a.error or "")
record("pause", "a completed run keeps no resume checkpoint",
       not (a_cfg.run_dir / Layout.RESUME).exists())
ok, why, _ = runs.resume_info(a)
record("pause", "a completed run cannot be resumed", not ok and "Completed" in why, why)

b_cfg = config("paused")
t_start = time.time()
launcher.start_run(b_cfg)
launcher.request_pause(b_cfg.run_dir)       # read once the first epoch ends
b = wait(b_cfg.run_dir)
record("pause", "a pause stops the run after epoch 1",
       b.status == RunStatus.STOPPED and b.epoch == 1 and b.resume_epoch == 1,
       f"{b.status.value} epoch={b.epoch} resume_epoch={b.resume_epoch}")
record("pause", "the PAUSE file is cleared when the run stops",
       not (b_cfg.run_dir / Layout.PAUSE).exists())
ok, why, nxt = runs.resume_info(b)
record("pause", "the paused run can be resumed from epoch 2", ok and nxt == 2, why)

gap = 4.0
time.sleep(gap)
launcher.resume_run(b_cfg.run_dir)
b = wait(b_cfg.run_dir)
t_total = time.time() - t_start
record("pause", "the resumed run completes every epoch",
       b.status == RunStatus.COMPLETED and b.epoch == EPOCHS, f"{b.status.value} {b.epoch}")
da, db = metrics(a_cfg.run_dir), metrics(b_cfg.run_dir)
cols = [c for c in da.select_dtypes("number").columns if c != "epoch_time"]
diff = float((da[cols] - db[cols]).abs().max().max()) if list(da.epoch) == list(db.epoch) else None
record("pause", "every metric equals the uninterrupted run's",
       diff is not None and diff < 1e-6, f"max difference {diff}")
record("pause", "the epoch history lists each epoch once",
       list(db.epoch) == [1, 2, 3] and epochs_seen(b_cfg.run_dir) == [1, 2, 3],
       f"csv={list(db.epoch)} events={epochs_seen(b_cfg.run_dir)}")
record("pause", "the duration leaves out the time spent paused",
       b.duration_s is not None and b.duration_s < t_total - gap + 0.5,
       f"duration={b.duration_s:.1f}s, wall={t_total:.1f}s, paused {gap}s")
log = (b_cfg.run_dir / Layout.LOG).read_text()
record("pause", "the log says where it resumed", "Resumed after epoch 1 of 3" in log)
try:
    launcher.resume_run(b_cfg.run_dir)
    refused = False
except launcher.LaunchError:
    refused = True
record("pause", "resume_run refuses a completed run itself", refused)

# ══ 2 · Stop now, in the middle of an epoch ════════════════════════════════
print("\n2 · Stop now in the middle of epoch 2")
c_cfg = config("stopped")
launcher.start_run(c_cfg)
fired = []


def stop_in_epoch_2(s):
    if not fired and batch_seen(c_cfg.run_dir, 2):
        fired.append(1)
        launcher.request_stop(c_cfg.run_dir)


c = wait(c_cfg.run_dir, stop_in_epoch_2)
record("stop", "Stop now ends the run during epoch 2",
       c.status == RunStatus.STOPPED and c.resume_epoch == 1,
       f"{c.status.value} resume_epoch={c.resume_epoch}")
ok, why, nxt = runs.resume_info(c)
record("stop", "it resumes by repeating epoch 2", ok and nxt == 2, why)

# ══ 3 · The process killed ═════════════════════════════════════════════════
print("\n3 · The process killed in the middle of epoch 2")
k_cfg = config("killed")
launcher.start_run(k_cfg)
killed = []


def kill_in_epoch_2(s):
    if not killed and batch_seen(k_cfg.run_dir, 2) and s.pid:
        killed.append(1)
        os.kill(s.pid, signal.SIGKILL)


k = wait(k_cfg.run_dir, kill_in_epoch_2)
ok, why, nxt = runs.resume_info(k)
record("kill", "a killed run is reported failed and can be resumed",
       k.status == RunStatus.FAILED and ok and nxt == 2, f"{k.status.value} {why}")
launcher.resume_run(k_cfg.run_dir)
k = wait(k_cfg.run_dir)
record("kill", "the killed run, resumed, completes with each epoch once",
       k.status == RunStatus.COMPLETED and list(metrics(k_cfg.run_dir).epoch) == [1, 2, 3]
       and epochs_seen(k_cfg.run_dir) == [1, 2, 3],
       f"{k.status.value} events={epochs_seen(k_cfg.run_dir)}")

# ══ 4 · What is not resumed ════════════════════════════════════════════════
print("\n4 · Runs that cannot be resumed")
old = os.path.join(ROOT, "old-run")
shutil.copytree(os.path.join(T, "e2e", "runs", "tv-cls"), old)
p = os.path.join(old, Layout.STATE)
st = json.load(open(p))
st.update(status="stopped")
json.dump(st, open(p, "w"))
ok, why, _ = runs.resume_info(runs.load_summary(old))
record("refuse", "a run from before resume support is refused, and says so",
       not ok and "before resuming" in why, why)

live = os.path.join(ROOT, "live-run")
shutil.copytree(c_cfg.run_dir, live)
cfg = json.load(open(os.path.join(live, Layout.CONFIG)))
cfg["run_name"] = "live-run"
json.dump(cfg, open(os.path.join(live, Layout.CONFIG), "w"))
st = json.load(open(os.path.join(live, Layout.STATE)))
st.update(status="running", pid=os.getpid())
json.dump(st, open(os.path.join(live, Layout.STATE), "w"))
ok, why, _ = runs.resume_info(runs.load_summary(live))
record("refuse", "a run that is still training is refused", not ok, why)

# ══ 5 · The pages ══════════════════════════════════════════════════════════
print("\n5 · The Training page and the Dashboard")
at = page("views/4_Training.py", **{K.K_OUTPUT: ROOT, K.K_ACTIVE_RUN: live})
record("ui", "a running run shows Pause and Stop now",
       not at.exception and has_button(at, "pause") and has_button(at, "stop"),
       str(at.exception) if at.exception else "")
at = at.button(key="pause").click().run()
record("ui", "Pause writes the PAUSE file", os.path.exists(os.path.join(live, Layout.PAUSE)))
record("ui", "a pending pause can be cancelled", has_button(at, "pause_cancel"))
at = at.button(key="pause_cancel").click().run()
record("ui", "Cancel removes the PAUSE file", not os.path.exists(os.path.join(live, Layout.PAUSE)))
shutil.rmtree(live)

rkey = f"resume_train_{c_cfg.run_dir}"
at = page("views/4_Training.py", **{K.K_OUTPUT: ROOT, K.K_ACTIVE_RUN: str(c_cfg.run_dir)})
record("ui", "a stopped run shows Resume and where it continues",
       not at.exception and has_button(at, rkey) and "continues from epoch 2 of 3" in blob(at),
       str(at.exception) if at.exception else "")
record("ui", "and no Pause or Stop", not has_button(at, "pause") and not has_button(at, "stop"))

at = AppTest.from_file(os.path.join(PROJ, "app.py"), default_timeout=300)
at.session_state[K.K_OUTPUT] = ROOT
at = at.run()
record("ui", "the Dashboard offers Resume for the stopped run only",
       has_button(at, f"resume_dash_{c_cfg.run_dir}")
       and not has_button(at, f"resume_dash_{a_cfg.run_dir}"))

at = page("views/4_Training.py", **{K.K_OUTPUT: ROOT, K.K_ACTIVE_RUN: str(c_cfg.run_dir)})
at = at.button(key=rkey).click().run()
c = wait(c_cfg.run_dir)
record("ui", "Resume on the Training page continues the run to the end",
       c.status == RunStatus.COMPLETED and list(metrics(c_cfg.run_dir).epoch) == [1, 2, 3]
       and epochs_seen(c_cfg.run_dir) == [1, 2, 3],
       f"{c.status.value} events={epochs_seen(c_cfg.run_dir)}")

shutil.rmtree(TMP, ignore_errors=True)
sys.exit(summary("UI TEST 8 · pausing and resuming"))
