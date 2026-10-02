"""Dashboard — hardware status, where you are in the flow, and recent runs."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core import prefs, runs
from core.hardware import detect, missing_packages, telemetry
from core.registry import MODEL_REGISTRY
from core.schemas import RunStatus
from ui import state
from ui.components import (
    badges,
    dim,
    empty_state,
    fmt_duration,
    fmt_metric,
    page_header,
    status_pill,
)

page_header("Dashboard", "System status, a summary of the flow and recent runs")

# ─────────────────────────────────────────────────────────────────────────────
# Hardware
# ─────────────────────────────────────────────────────────────────────────────

dev = detect()
tel = telemetry()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Compute device", {"cuda": "NVIDIA GPU", "mps": "Apple GPU", "cpu": "CPU"}[dev.kind])
c2.metric(
    "Memory",
    f"{dev.total_memory_gb:.0f} GB",
    f"{tel['gpu_mem_used_gb']:.1f} GB in use" if "gpu_mem_used_gb" in tel else None,
    delta_color="off",
)
c3.metric("Models in the catalogue", len(MODEL_REGISTRY))
c4.metric("RAM usage", f"{tel.get('ram_pct', 0):.0f}%" if tel else "—")

st.markdown(
    f"<div class='ts-card ts-card-tight'><span class='ts-dim'>{dev.label}</span>"
    + badges([
        ("Mixed precision (AMP)" if dev.supports_amp else "AMP off",
         "accent" if dev.supports_amp else ""),
        f"{dev.count} devices" if dev.count > 1 else dev.torch_device,
    ])
    + "</div>",
    unsafe_allow_html=True,
)

missing = missing_packages()
if missing:
    st.error(
        "**Some packages are missing** — training cannot start.\n\n"
        f"`pip install {' '.join(missing)}`\n\n"
        "Or set up the prepared environment: `conda env create -f environment.yml && conda activate trainui`"
    )

if dev.kind == "cpu":
    st.warning(
        "No GPU was found. Training on the CPU will be very slow — "
        "to try it out, pick one of the smallest models in the catalogue "
        "(EfficientNet-B0, YOLO26-n)."
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Flow status
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## Training flow")

ds = state.dataset()
spec = state.spec()
hp = state.hp()

s1, s2, s3, s4 = st.columns(4)

with s1:
    if ds:
        st.success(f"**Dataset**\n\n{Path(ds.root).name}")
        dim(f"{ds.task.label} · {ds.n_total:,} samples · {ds.num_classes} classes")
    else:
        st.info("**1 · Dataset**\n\nNot selected yet")
    st.page_link("views/1_Dataset.py", label="Dataset", icon="📁", width="stretch")

with s2:
    if spec:
        st.success(f"**Model**\n\n{spec.display_name}")
        dim(f"{spec.family} · {spec.params_label} parameters · {spec.released}")
    else:
        st.info("**2 · Model**\n\nNot selected yet")
    st.page_link("views/2_Model_Selection.py", label="Model Selection", icon="🧠",
                 width="stretch", disabled=ds is None)

with s3:
    if hp:
        st.success("**Settings**\n\nReady")
        dim(f"{hp.epochs} epochs · batch {hp.batch_size} · lr {hp.lr:.1e}")
    else:
        st.info("**3 · Settings**\n\nNot configured yet")
    st.page_link("views/3_Settings.py", label="Settings", icon="⚙️",
                 width="stretch", disabled=not state.ready_to_configure())

with s4:
    active = state.get(state.K_ACTIVE_RUN)
    if active:
        summary = runs.load_summary(active)
        if summary:
            st.markdown(status_pill(summary.status.value, summary.status.label),
                        unsafe_allow_html=True)
            dim(summary.run_name)
        else:
            st.info("**4 · Training**\n\nRun not found")
    else:
        st.info("**4 · Training**\n\nNot started")
    st.page_link("views/4_Training.py", label="Training", icon="🚀",
                 width="stretch", disabled=hp is None and active is None)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Recent runs
# ─────────────────────────────────────────────────────────────────────────────

head, ctrl = st.columns([3, 1])
head.markdown("## Recent runs")

roots = [state.output_dir()] + [p for p in prefs.recent("output") if p != state.output_dir()]
recent_runs = runs.list_runs_multi(roots)[:8]

if ctrl.button("🔄 Refresh", width="stretch"):
    st.rerun()

if not recent_runs:
    empty_state(
        "🗂️", "No runs yet",
        f"Searched in: {state.output_dir()}",
        "Pick a dataset and start your first run.",
    )
else:
    live = [r for r in recent_runs if r.is_live]
    if live:
        st.caption(f"{len(live)} run(s) currently in progress.")

    for r in recent_runs:
        with st.container(border=True):
            a, b, c, d, e = st.columns([3, 2, 2, 2, 1.4])
            with a:
                st.markdown(
                    status_pill(r.status.value, r.status.label) +
                    f" <span class='ts-mono'>{r.run_name}</span>",
                    unsafe_allow_html=True,
                )
                dim(f"{r.model} · {r.dataset}")
            b.metric("Progress", f"{r.epoch}/{r.total_epochs}" if r.total_epochs else "—")
            c.metric(
                r.monitor_metric or "metric",
                fmt_metric(r.best_value),
                f"epoch {r.best_epoch}" if r.best_epoch else None,
                delta_color="off",
            )
            d.metric("Duration", fmt_duration(r.duration_s))
            with e:
                if st.button("Open", key=f"open_{r.run_dir}", width="stretch"):
                    state.put(state.K_ACTIVE_RUN, r.run_dir)
                    st.switch_page("views/4_Training.py")
            if r.total_epochs:
                st.progress(r.progress)
            if r.error and r.status == RunStatus.FAILED:
                st.caption(f"🛑 {r.error[:180]}")

    st.page_link("views/5_Results.py", label="Compare all results", icon="📊")
