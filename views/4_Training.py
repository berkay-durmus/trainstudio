"""Training — live monitoring.

This page never touches the training process; it only reads `events.jsonl` from
the last byte offset it consumed. That is what lets the browser be refreshed, a
tab be closed, or the same run be opened in several tabs without affecting
training and without the display becoming inconsistent.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from core import runs
from core.events import MetricAccumulator, tail_events, tail_log
from core.launcher import force_kill, request_stop, runner_stdout, stop_requested
from core.schemas import Layout, RunStatus, metric_mode
from ui import state
from ui.charts import curves, sparkline
from ui.components import (
    delete_run_control,
    dim,
    empty_state,
    faint,
    fmt_duration,
    fmt_metric,
    page_header,
    status_pill,
)

REFRESH_SECONDS = 2

page_header("Training", "Live metrics, sample predictions and logs.",
            active_step=4, done_steps=state.done_steps())

# ─────────────────────────────────────────────────────────────────────────────
# Which run are we watching?
# ─────────────────────────────────────────────────────────────────────────────

active = state.get(state.K_ACTIVE_RUN)
all_runs = runs.list_runs_multi([state.output_dir()])

if not active and all_runs:
    active = all_runs[0].run_dir

if not active:
    empty_state("🚀", "There is no run to watch",
                f"Searched in: {state.output_dir()}",
                "Start a run from the Settings page.")
    st.page_link("views/3_Settings.py", label="→ Settings", icon="⚙️")
    st.stop()

if len(all_runs) > 1:
    options = [r.run_dir for r in all_runs]
    idx = options.index(active) if active in options else 0
    labels = {r.run_dir: f"{r.status.label} · {r.run_name}" for r in all_runs}
    picked = st.selectbox("Run being watched", options, index=idx,
                          format_func=lambda d: labels.get(d, Path(d).name))
    if picked != active:
        state.put(state.K_ACTIVE_RUN, picked)
        st.session_state.pop("train.acc", None)
        st.rerun()
    active = picked

run_dir = Path(active)
summary = runs.load_summary(run_dir)
if summary is None:
    st.error(f"Not a valid run directory: `{run_dir}`")
    st.stop()

cfg = summary.config
monitor = summary.monitor_metric or "val_loss"
mode = metric_mode(monitor)

# Reset the accumulator when the run changes
if st.session_state.get("train.acc_dir") != str(run_dir):
    st.session_state["train.acc"] = MetricAccumulator()
    st.session_state["train.acc_dir"] = str(run_dir)


def read_new_events() -> MetricAccumulator:
    """Read only the new lines and feed the accumulator."""
    acc: MetricAccumulator = st.session_state["train.acc"]
    result = tail_events(run_dir / Layout.EVENTS, acc.offset)
    if result.truncated and result.offset == 0:
        acc.reset()
    acc.feed(result.events)
    acc.offset = result.offset
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# The live section — refreshes itself every 2 seconds
# ─────────────────────────────────────────────────────────────────────────────


@st.fragment(run_every=REFRESH_SECONDS)
def live_panel() -> None:
    fresh = runs.load_summary(run_dir) or summary
    acc = read_new_events()
    running = fresh.status == RunStatus.RUNNING and runs.process_alive(fresh.pid)

    # ── Status strip ─────────────────────────────────────────────────────
    head, ctrl = st.columns([4, 1.2])
    with head:
        stopping = stop_requested(run_dir)
        label = "Stopping…" if (running and stopping) else fresh.status.label
        st.markdown(
            status_pill(fresh.status.value, label) +
            f" &nbsp; <span class='ts-mono'>{fresh.run_name}</span>",
            unsafe_allow_html=True)
        dim(f"{fresh.model} · {fresh.dataset} · monitored metric: {monitor}")
    with ctrl:
        if running:
            if st.button("⏹ Stop", width="stretch", type="primary"):
                request_stop(run_dir)
                st.toast("Stop requested — the model will be saved and the run will close.")
                st.rerun()
        elif fresh.status == RunStatus.RUNNING:
            if st.button("⚠️ Terminate the process", width="stretch"):
                force_kill(run_dir)
                st.rerun()

    # ── KPIs ─────────────────────────────────────────────────────────────
    last = acc.epochs[-1] if acc.epochs else {}
    eta = acc.eta_seconds()
    ips = (acc.last_batch or {}).get("ips")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Epoch", f"{fresh.epoch}/{fresh.total_epochs or '?'}")
    k2.metric(f"Best {monitor}", fmt_metric(fresh.best_value),
              f"epoch {fresh.best_epoch}" if fresh.best_epoch else None, delta_color="off")
    k3.metric("Validation loss", fmt_metric(last.get("val_loss")))
    k4.metric("Elapsed", fmt_duration(fresh.duration_s))
    k5.metric("Remaining (estimate)" if running else "Throughput",
              fmt_duration(eta) if running else (f"{ips:.0f} samples/s" if ips else "—"))

    # ── Progress ─────────────────────────────────────────────────────────
    if fresh.total_epochs:
        st.progress(fresh.progress, text=f"Epoch {fresh.epoch}/{fresh.total_epochs}")
    b = acc.last_batch
    if running and b and b.get("total_steps"):
        frac = min(1.0, b["step"] / b["total_steps"])
        st.progress(frac, text=f"Step {b['step']}/{b['total_steps']} · "
                               f"loss {b.get('loss', 0):.4f} · lr {b.get('lr', 0):.2e}")

    # ── Error / waiting states ───────────────────────────────────────────
    if fresh.status == RunStatus.FAILED:
        st.error(f"**Training ended with an error**\n\n{fresh.error or 'No details.'}")
        stdout = runner_stdout(run_dir)
        if stdout.strip():
            with st.expander("Process output (runner.out)"):
                st.code(stdout[-6000:], language="text")
    elif fresh.status == RunStatus.QUEUED:
        st.info("The process has started and the first epoch is being prepared… "
                "(the model may be downloading, or the data may still be scanning)")

    if not acc.epochs:
        if running or fresh.status == RunStatus.QUEUED:
            st.caption("The charts will appear here once the first epoch completes.")
        return

    # ── Charts ───────────────────────────────────────────────────────────
    st.markdown("")
    g1, g2 = st.columns(2)

    with g1:
        st.markdown("**Loss**")
        series = {}
        for key, label in (("train_loss", "Training"), ("val_loss", "Validation")):
            xs, ys = acc.series(key)
            if xs:
                series[label] = (xs, ys)
        if series:
            st.plotly_chart(curves(series, y_title="Loss", height=290, uirev="loss"),
                            width="stretch", config={"displayModeBar": False})

    with g2:
        keys = [k for k in acc.metric_keys()
                if k.startswith("val_") and not k.endswith("loss")]
        default = f"val_{monitor}" if f"val_{monitor}" in keys else (keys[0] if keys else None)
        if default:
            choice = st.selectbox("Metric", keys, index=keys.index(default),
                                  key="live_metric", label_visibility="collapsed")
            xs, ys = acc.series(choice)
            train_key = choice.replace("val_", "train_", 1)
            series = {"Validation": (xs, ys)}
            txs, tys = acc.series(train_key)
            if txs:
                series["Training"] = (txs, tys)
            best_row = acc.best_row(choice, metric_mode(choice.replace("val_", "")))
            hl = (best_row["epoch"], best_row[choice]) if best_row else None
            st.plotly_chart(
                curves(series, y_title=choice, height=290, uirev="metric", highlight=hl),
                width="stretch", config={"displayModeBar": False})

    # ── Detail tabs ──────────────────────────────────────────────────────
    t_prev, t_sys, t_table, t_log = st.tabs(
        ["🖼️ Previews", "🖥️ System", "📋 Epoch table", "📜 Log"])

    with t_prev:
        arts = [a for a in acc.artifacts if a.get("kind") == "preview"]
        if not arts:
            st.caption("No previews have been produced yet.")
        else:
            latest = arts[-1]
            path = run_dir / latest["path"]
            if path.is_file():
                st.image(str(path), caption=f"epoch {latest.get('epoch')}", width="stretch")
            if len(arts) > 1:
                pick = st.select_slider(
                    "Earlier epochs", options=[a.get("epoch") for a in arts],
                    value=arts[-1].get("epoch"), key="prev_slider")
                sel = next((a for a in arts if a.get("epoch") == pick), None)
                if sel and (run_dir / sel["path"]).is_file() and pick != latest.get("epoch"):
                    st.image(str(run_dir / sel["path"]), caption=f"epoch {pick}",
                             width="stretch")

    with t_sys:
        if not acc.system:
            st.caption("No system telemetry has arrived yet.")
        else:
            recent = acc.system[-120:]
            s1, s2, s3 = st.columns(3)
            with s1:
                vals = [e.get("cpu_pct", 0) for e in recent]
                st.metric("CPU", f"{vals[-1]:.0f}%" if vals else "—")
                st.plotly_chart(sparkline(vals, y_range=(0, 100)), width="stretch",
                                config={"displayModeBar": False})
            with s2:
                vals = [e.get("ram_pct", 0) for e in recent]
                st.metric("RAM", f"{vals[-1]:.0f}%" if vals else "—")
                st.plotly_chart(sparkline(vals, "#60A5FA", y_range=(0, 100)),
                                width="stretch", config={"displayModeBar": False})
            with s3:
                vals = [e.get("gpu_mem_used_gb") for e in recent if e.get("gpu_mem_used_gb")]
                gpu = [e.get("gpu_pct") for e in recent if e.get("gpu_pct") is not None]
                if gpu:
                    st.metric("GPU", f"{gpu[-1]:.0f}%")
                    st.plotly_chart(sparkline(gpu, "#F472B6", y_range=(0, 100)),
                                    width="stretch", config={"displayModeBar": False})
                elif vals:
                    st.metric("GPU memory", f"{vals[-1]:.1f} GB")
                    st.plotly_chart(sparkline(vals, "#F472B6"), width="stretch",
                                    config={"displayModeBar": False})

    with t_table:
        df = acc.to_dataframe()
        if not df.empty:
            cols = [c for c in df.columns if c != "best"]
            st.dataframe(df[cols].iloc[::-1], width="stretch", height=340,
                         hide_index=True)

    with t_log:
        st.code(tail_log(run_dir, 220) or "(no log yet)", language="text")

    if running:
        faint(f"Updating every {REFRESH_SECONDS} seconds · "
              f"last update {time.strftime('%H:%M:%S')}")


live_panel()

# ─────────────────────────────────────────────────────────────────────────────
# The static lower section
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
a, b, c, d = st.columns(4)

with a:
    st.markdown(f"<span class='ts-mono ts-faint'>{run_dir}</span>", unsafe_allow_html=True)
with b:
    if (run_dir / Layout.REPORT).is_file():
        st.link_button("📄 Open the HTML report", f"file://{run_dir / Layout.REPORT}",
                       width="stretch")
with c:
    ckpt = run_dir / Layout.BEST
    if ckpt.is_file():
        size = ckpt.stat().st_size / 1024**2
        st.caption(f"best.pt · {size:.1f} MB")
with d:
    if st.button("📊 Compare in Results", width="stretch"):
        st.switch_page("views/5_Results.py")
    if summary.status.is_terminal:
        delete_run_control(summary, key=f"train_{run_dir}", label="🗑️ Delete this run")

if cfg is not None:
    with st.expander("⚙️ This run's configuration"):
        st.json(cfg.model_dump(mode="json"))
