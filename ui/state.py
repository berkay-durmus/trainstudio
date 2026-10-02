"""The session state of the wizard flow.

The only things carried between pages are the keys defined here; pages never
touch `st.session_state` directly. That keeps the flow — where each step's output
is the next step's input — traceable from one place.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from core import prefs
from core.recommend import Recommendation
from core.registry import ModelSpec
from core.schemas import AugConfig, DatasetConfig, Hyperparams, WindowSpec

K_SCAN = "flow.scan"                # data.scan.ScanResult
K_DATASET = "flow.dataset"          # DatasetConfig
K_WINDOW = "flow.window"            # WindowSpec | None
K_SPEC = "flow.spec"                # ModelSpec
K_ENCODER = "flow.encoder"          # str | None
K_HP = "flow.hp"                    # Hyperparams
K_AUG = "flow.aug"                  # AugConfig
K_REC = "flow.rec"                  # Recommendation
K_TOUCHED = "flow.touched"          # the fields the user changed by hand
K_OUTPUT = "flow.output_dir"        # str
K_RUN_NAME = "flow.run_name"        # str
K_ACTIVE_RUN = "flow.active_run"    # str — the run directory being watched


# ─────────────────────────────────────────────────────────────────────────────
# General access
# ─────────────────────────────────────────────────────────────────────────────


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def put(key: str, value: Any) -> None:
    st.session_state[key] = value


def clear(*keys: str) -> None:
    for k in keys:
        st.session_state.pop(k, None)


# ─────────────────────────────────────────────────────────────────────────────
# Flow steps
# ─────────────────────────────────────────────────────────────────────────────


def dataset() -> DatasetConfig | None:
    return get(K_DATASET)


def spec() -> ModelSpec | None:
    return get(K_SPEC)


def hp() -> Hyperparams | None:
    return get(K_HP)


def aug() -> AugConfig | None:
    return get(K_AUG)


def recommendation() -> Recommendation | None:
    return get(K_REC)


def output_dir() -> str:
    """Where runs are written. The env var matters in a container: the working
    directory there is the read-only application directory, so falling back to
    `cwd/runs` would write results into a path nobody has mounted."""
    return (get(K_OUTPUT)
            or prefs.get("output_dir")
            or os.environ.get("TRAINSTUDIO_RUNS_DIR")
            or str(Path.cwd() / "runs"))


def set_dataset(ds: DatasetConfig, window: WindowSpec | None = None) -> None:
    """If the dataset changes, the model and parameter choices become invalid."""
    prev = dataset()
    put(K_DATASET, ds)
    put(K_WINDOW, window)
    if prev is None or prev.task != ds.task or prev.root != ds.root:
        clear(K_SPEC, K_HP, K_AUG, K_REC, K_TOUCHED, K_ENCODER)


def set_spec(s: ModelSpec) -> None:
    """If the model changes, the recommendations must be recomputed."""
    prev = spec()
    put(K_SPEC, s)
    if prev is None or prev.id != s.id:
        clear(K_HP, K_AUG, K_REC, K_TOUCHED)


def touched() -> set[str]:
    return get(K_TOUCHED) or set()


def mark_touched(field: str) -> None:
    t = touched()
    t.add(field)
    put(K_TOUCHED, t)


def reset_touched() -> None:
    put(K_TOUCHED, set())


def done_steps() -> set[int]:
    """The completed steps shown in the flow indicator."""
    done: set[int] = set()
    if dataset() is not None:
        done.add(1)
    if spec() is not None:
        done.add(2)
    if hp() is not None:
        done.add(3)
    if get(K_ACTIVE_RUN):
        done.add(4)
    return done


def ready_to_configure() -> bool:
    return dataset() is not None and spec() is not None


# ─────────────────────────────────────────────────────────────────────────────
# Shared guards for the start of a page
# ─────────────────────────────────────────────────────────────────────────────


def require_dataset() -> DatasetConfig | None:
    ds = dataset()
    if ds is None:
        st.warning("Select and validate a dataset first.")
        st.page_link("views/1_Dataset.py", label="→ Go to the Dataset page", icon="📁")
        return None
    return ds


def require_spec() -> ModelSpec | None:
    s = spec()
    if s is None:
        st.warning("Select a model first.")
        st.page_link("views/2_Model_Selection.py", label="→ Go to the Model Selection page", icon="🧠")
        return None
    return s
