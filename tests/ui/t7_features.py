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

shutil.rmtree(TMP, ignore_errors=True)
sys.exit(summary("UI TEST 7 · new features"))
