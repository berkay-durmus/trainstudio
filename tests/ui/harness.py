"""Shared helpers for driving TrainStudio pages through streamlit's AppTest."""
import os, pathlib, sys, tempfile, warnings
warnings.filterwarnings("ignore")
# Never the user's own ~/.trainstudio: its recent output folders would put their real
# runs on the pages under test. Parts that need a known home set it before this import.
os.environ.setdefault("TRAINSTUDIO_HOME", tempfile.mkdtemp(prefix="ts_home_"))
# The project root is two levels up from tests/ui/. Pages are loaded by path,
# and several of them read relative paths, so the working directory has to be
# the project root as it is when `streamlit run app.py` starts.
PROJ = str(pathlib.Path(__file__).resolve().parents[2])
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
os.chdir(PROJ)

# Where the synthetic datasets and the two short training runs live. Built by
# fixtures.py; override with TS_UI_FIXTURES to keep them outside the repo.
T = os.environ.get("TS_UI_FIXTURES") or os.path.join(PROJ, ".ui-fixtures")

from streamlit.testing.v1 import AppTest       # noqa: E402

# Keys the folder picker registers (ui/dir_picker._state_key)
def dp(key, name):
    return f"_dp::{key}::{name}"

RESULTS = []

def record(group, name, ok, detail=""):
    RESULTS.append((group, name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))

def texts(at):
    """Every piece of user-visible text the page produced."""
    out = []
    for attr in ("markdown", "error", "warning", "info", "success", "caption",
                 "text", "code", "header", "subheader", "title", "metric"):
        try:
            for el in getattr(at, attr):
                v = getattr(el, "value", None)
                if isinstance(v, str):
                    out.append(v)
                lab = getattr(el, "label", None)
                if isinstance(lab, str):
                    out.append(lab)
        except Exception:
            pass
    return out

def tables(at):
    """Text of every dataframe on the page — st.dataframe content is not markdown."""
    out = []
    try:
        for d in at.dataframe:
            v = d.value
            out.append(v.to_string() if hasattr(v, "to_string") else str(v))
    except Exception:
        pass
    return out


def json_blocks(at):
    """Text of st.json elements — the config preview is rendered as one."""
    out = []
    try:
        for j in at.json:
            v = j.value
            out.append(v if isinstance(v, str) else str(v))
    except Exception:
        pass
    return out


def containers(at):
    """Labels of expanders and tabs — collapsed content is still reachable UI."""
    out = []
    for attr in ("expander", "tabs"):
        try:
            for c in getattr(at, attr):
                lab = getattr(c, "label", None)
                if isinstance(lab, str):
                    out.append(lab)
        except Exception:
            pass
    return out


def widget_options(at):
    out = []
    for attr in ("selectbox", "multiselect", "radio"):
        try:
            for w in getattr(at, attr):
                out.extend(str(o) for o in (w.options or []))
        except Exception:
            pass
    return out


def blob(at):
    """Everything a user could read on the page, including tables and options."""
    return "\n".join(texts(at) + tables(at) + widget_options(at)
                      + containers(at) + json_blocks(at)).lower()

def messages(at, kind):
    try:
        return [e.value for e in getattr(at, kind)]
    except Exception:
        return []

def ss_get(at, key, default=None):
    """AppTest's session_state proxy has no .get(); only item access works."""
    try:
        return at.session_state[key]
    except Exception:
        return default


def ss_set(at, key, value):
    at.session_state[key] = value


def run_page(page, seed=None, timeout=180):
    at = AppTest.from_file(os.path.join(PROJ, page), default_timeout=timeout)
    if seed:
        seed(at)
    return at.run()

def summary(title):
    total = len(RESULTS)
    failed = [r for r in RESULTS if not r[2]]
    print(f"\n{'='*72}\n{title}: {total - len(failed)}/{total} passed")
    if failed:
        print("FAILED:")
        for g, n, _, d in failed:
            print(f"  · [{g}] {n}  {d}")
    print("="*72)
    return 1 if failed else 0
