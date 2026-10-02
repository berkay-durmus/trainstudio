"""The Inference page's "Explain this prediction" panel: Grad-CAM, LIME and SHAP.

Grad-CAM is instant and runs as soon as it is picked. LIME and SHAP evaluate the
model hundreds of times, so they run on a button that says beforehand roughly how
long that will take on this machine. Results are kept per image, method, class
and effort, so moving to another method and back does not recompute them.
"""

from __future__ import annotations

import hashlib

import numpy as np
import streamlit as st

from core.schemas import Task
from export import explain as xai
from ui.components import faint

_KEY = "_xai"          # {(run, ckpt, image md5, method, class, effort): Explanation}
_SPEED = "_xai_speed"  # {(run, ckpt): seconds per image}
_KEEP = 24


def _target_default(loaded, pred) -> int:
    if pred.task == Task.CLASSIFICATION and pred.probs is not None:
        return int(pred.probs.argmax())
    if pred.mask is not None:
        counts = np.bincount(pred.mask.ravel(), minlength=len(loaded.classes))
        if len(counts) > 1 and counts[1:].any():
            return int(counts[1:].argmax()) + 1      # the largest class that is not background
    return 0


def _speed(loaded, ident) -> float:
    held = st.session_state.setdefault(_SPEED, {})
    if ident not in held:
        held[ident] = xai.seconds_per_image(loaded)
    return held[ident]


def _eta(seconds: float) -> str:
    if seconds < 1.5:
        return "a second"
    if seconds < 90:
        return f"about {seconds:.0f} s"
    return f"about {seconds / 60:.0f} min"


def _cached(ident, image, method, target, effort):
    key = (*ident, hashlib.md5(image.tobytes()).hexdigest(), method, target,
           effort if method != "gradcam" else None)
    return key, st.session_state.setdefault(_KEY, {}).get(key)


def _compute(loaded, image, key, method, target, effort):
    held = st.session_state.setdefault(_KEY, {})
    with st.spinner(f"Running {xai.METHODS[method]['label']}…"):
        try:
            result = xai.explain(loaded, image, method, target, effort)
        except Exception as exc:          # an explanation must never break the page
            result = exc
    held[key] = result
    while len(held) > _KEEP:
        held.pop(next(iter(held)))
    return result


def _show(result, method: str) -> None:
    label = xai.METHODS[method]["label"]
    if isinstance(result, Exception):
        st.warning(f"{label} could not be produced: {result}")
        return
    st.image(result.overlay, caption=f"{label} · {result.elapsed_s:.1f} s", width="stretch")
    if method == "gradcam" and float(result.heat.max()) <= 0:
        st.caption("No region raises this class's score at the watched layer — the model "
                   "finds no evidence for it here.")
    faint(result.note)


def render(loaded, image: np.ndarray, pred, run_dir: str, which: str, key: str) -> None:
    if loaded.task not in (Task.CLASSIFICATION, Task.SEGMENTATION):
        return
    # A toggle, not an expander: an expander runs its body even while closed
    if st.toggle("🔍 Explain this prediction — Grad-CAM, LIME, SHAP", key=f"xai_on_{key}"):
        _panel(loaded, image, pred, run_dir, which, key)


def _panel(loaded, image: np.ndarray, pred, run_dir: str, which: str, key: str) -> None:
    ok = xai.available(loaded)
    ident = (run_dir, which)

    c1, c2, c3 = st.columns([2.2, 2, 1.4])
    method = c1.radio("Method", list(xai.METHODS), horizontal=True, key=f"xai_m_{key}",
                      format_func=lambda m: xai.METHODS[m]["label"])
    names = loaded.classes
    target = c2.selectbox("Explain class", list(range(len(names))),
                          index=min(_target_default(loaded, pred), len(names) - 1),
                          key=f"xai_c_{key}",
                          format_func=lambda i: names[i])
    compare = c3.toggle("Compare all", key=f"xai_all_{key}",
                        help="All three methods side by side")
    effort = "balanced"
    if compare or method != "gradcam":
        effort = st.radio("Effort", list(xai.BUDGETS), index=1, horizontal=True,
                          key=f"xai_e_{key}", help="LIME samples and SHAP evaluations — "
                          "more is steadier and slower")

    methods = list(xai.METHODS) if compare else [method]
    if not compare:
        faint(xai.METHODS[method]["about"])

    missing = [m for m in methods if not ok[m][0]]
    for m in missing:
        st.info(ok[m][1])
    methods = [m for m in methods if ok[m][0]]
    if not methods:
        return

    keys, results = {}, {}
    for m in methods:
        keys[m], results[m] = _cached(ident, image, m, target, effort)
    # Grad-CAM is instant; the others wait for the button
    if "gradcam" in methods and results["gradcam"] is None:
        results["gradcam"] = _compute(loaded, image, keys["gradcam"], "gradcam", target, effort)
    pending = [m for m in methods if results[m] is None]
    if pending:
        n = sum(xai.evaluations(m, effort) for m in pending)
        secs = n * _speed(loaded, ident)
        names_ = " and ".join(xai.METHODS[m]["label"] for m in pending)
        if st.button(f"▶ Run {names_}", key=f"xai_go_{key}", type="primary",
                     help=f"{n:,} model evaluations"):
            for m in pending:
                results[m] = _compute(loaded, image, keys[m], m, target, effort)
        else:
            st.caption(f"{n:,} model evaluations — {_eta(secs)} on {loaded.device}.")

    cols = st.columns(len(methods)) if compare else [st.container()]
    for col, m in zip(cols, methods):
        with col:
            if compare:
                st.markdown(f"**{xai.METHODS[m]['label']}**")
            if results[m] is not None:
                _show(results[m], m)
            elif compare:
                st.caption("Not run yet.")
