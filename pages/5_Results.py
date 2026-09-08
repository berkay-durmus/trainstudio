"""Results — list runs, compare them, and reach the reports and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core import prefs, runs
from core.schemas import Layout, RunStatus, metric_mode
from ui import state
from ui.charts import multi_run_curves
from ui.components import (
    dim,
    empty_state,
    faint,
    fmt_duration,
    fmt_metric,
    page_header,
    status_pill,
)
from ui.dir_picker import directory_picker

page_header("Results", "Compare runs and reach their reports and artifacts.",
            active_step=5, done_steps=state.done_steps())

# ─────────────────────────────────────────────────────────────────────────────
# Source folder(s)
# ─────────────────────────────────────────────────────────────────────────────

roots = [state.output_dir()] + [p for p in prefs.recent("output") if p != state.output_dir()]
with st.expander("📂 Folders being searched", expanded=False):
    picked = directory_picker(key="results_root", label="Add another output folder",
                              initial=state.output_dir(), recent_kind="output")
    if picked and picked not in roots:
        roots.insert(0, picked)
    st.caption("Searching: " + " · ".join(f"`{r}`" for r in roots))

top_l, top_r = st.columns([4, 1])
if top_r.button("🔄 Refresh", width="stretch"):
    st.rerun()

all_runs = runs.list_runs_multi(roots)
if not all_runs:
    empty_state("📊", "No results yet",
                "No completed runs were found in these folders.",
                "Once you start a run, the results will be listed here.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Filters + table
# ─────────────────────────────────────────────────────────────────────────────

f1, f2, f3 = st.columns([2, 2, 2])
tasks = sorted({r.task for r in all_runs if r.task})
statuses = sorted({r.status.value for r in all_runs})

sel_tasks = f1.multiselect("Task", tasks, default=tasks, placeholder="Task")
sel_status = f2.multiselect("Status", statuses, default=statuses,
                            format_func=lambda s: RunStatus(s).label,
                            placeholder="Status")
search = f3.text_input("Search", placeholder="run name, model or dataset…",
                       label_visibility="collapsed")

filtered = [
    r for r in all_runs
    if (not sel_tasks or r.task in sel_tasks)
    and (not sel_status or r.status.value in sel_status)
    and (not search or search.lower() in
         f"{r.run_name} {r.model} {r.dataset}".lower())
]

if not filtered:
    st.info("No runs match the filters.")
    st.stop()

table = pd.DataFrame([{
    "select": False,
    "status": r.status.label,
    "run": r.run_name,
    "model": r.model,
    "dataset": r.dataset,
    "metric": r.monitor_metric,
    "best": r.best_value,
    "epoch": f"{r.epoch}/{r.total_epochs}" if r.total_epochs else str(r.epoch),
    "duration": fmt_duration(r.duration_s),
    "_dir": r.run_dir,
} for r in filtered])

st.markdown(f"### {len(filtered)} runs")
edited = st.data_editor(
    table, width="stretch", hide_index=True, key="runs_table",
    column_config={
        "select": st.column_config.CheckboxColumn("", width="small",
                                                  help="Tick to include in the comparison"),
        "best": st.column_config.NumberColumn(format="%.4f"),
        "_dir": None,
    },
    disabled=[c for c in table.columns if c != "select"],
)

selected_dirs = edited.loc[edited["select"], "_dir"].tolist()

# ─────────────────────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────────────────────

if selected_dirs:
    st.divider()
    st.markdown(f"## Comparison — {len(selected_dirs)} runs")

    available = runs.available_metrics(selected_dirs)
    if not available:
        st.info("No plottable metric was found in the selected runs.")
    else:
        first = next((r for r in filtered if r.run_dir == selected_dirs[0]), None)
        preferred = f"val_{first.monitor_metric}" if first else None
        idx = available.index(preferred) if preferred in available else 0

        c1, c2 = st.columns([2, 1])
        metric = c1.selectbox("Metric being compared", available, index=idx)
        smooth = c2.toggle("Skip missing epochs", value=True,
                           help="Shows runs of different lengths on the same axis.")

        df = runs.comparison_frame(selected_dirs, metric)
        if df.empty:
            st.info("The selected runs do not have this metric.")
        else:
            st.plotly_chart(multi_run_curves(df, metric), width="stretch")

            # The leaderboard
            mode = metric_mode(metric.replace("val_", "").replace("train_", ""))
            board = []
            for d in selected_dirs:
                value, epoch = runs.best_of(d, metric.replace("val_", ""))
                s = next((r for r in filtered if r.run_dir == d), None)
                board.append({
                    "run": Path(d).name,
                    "model": s.model if s else "",
                    metric: value,
                    "best epoch": epoch,
                    "duration": fmt_duration(s.duration_s) if s else "—",
                })
            board_df = pd.DataFrame(board).sort_values(
                metric, ascending=(mode == "min"), na_position="last")
            st.markdown("**Leaderboard**")
            st.dataframe(board_df, width="stretch", hide_index=True,
                         column_config={metric: st.column_config.NumberColumn(format="%.4f")})

            csv = df.pivot_table(index="epoch", columns="run", values="value")
            st.download_button("⬇️ Download the comparison data as CSV",
                               csv.to_csv().encode("utf-8"),
                               file_name=f"comparison_{metric}.csv", mime="text/csv")

    # The configuration difference
    if len(selected_dirs) == 2:
        with st.expander("⚙️ Settings that differ between the two runs"):
            def flat(d: dict, prefix: str = "") -> dict:
                out = {}
                for k, v in d.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        out.update(flat(v, key))
                    else:
                        out[key] = v
                return out

            try:
                a, b = [flat(json.loads((Path(d) / Layout.CONFIG).read_text(encoding="utf-8")))
                        for d in selected_dirs]
                rows = [{"setting": k, Path(selected_dirs[0]).name: a.get(k),
                         Path(selected_dirs[1]).name: b.get(k)}
                        for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                else:
                    st.caption("The configurations are identical.")
            except Exception as exc:
                st.warning(f"The configurations could not be read: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# A single run in detail
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("## Run detail")

names = {r.run_dir: r.run_name for r in filtered}
chosen = st.selectbox("Run", list(names), format_func=lambda d: names[d],
                      key="detail_run")
r = next(x for x in filtered if x.run_dir == chosen)
run_dir = Path(chosen)

h1, h2 = st.columns([4, 1.4])
with h1:
    st.markdown(status_pill(r.status.value, r.status.label) +
                f" &nbsp;<span class='ts-mono'>{r.run_name}</span>", unsafe_allow_html=True)
    dim(f"{r.model} · {r.dataset} · {r.epoch}/{r.total_epochs} epochs · "
        f"{fmt_duration(r.duration_s)}")
with h2:
    if st.button("📈 Open on the live page", width="stretch"):
        state.put(state.K_ACTIVE_RUN, chosen)
        st.switch_page("pages/4_Training.py")

summary_path = run_dir / Layout.SUMMARY
if summary_path.is_file():
    try:
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        final = s.get("final_metrics") or {}
        if final:
            st.markdown(f"**Final evaluation — the `{s.get('final_split','val')}` split**")
            items = [(k, v) for k, v in final.items() if isinstance(v, (int, float))]
            for i in range(0, len(items), 5):
                cols = st.columns(5)
                for col, (k, v) in zip(cols, items[i:i + 5]):
                    col.metric(k, fmt_metric(v))
        ci = s.get("balanced_accuracy_ci95")
        if ci:
            faint(f"Balanced accuracy 95% confidence interval: "
                  f"{ci[0]:.4f} [{ci[1]:.4f} – {ci[2]:.4f}]")
    except Exception as exc:
        st.warning(f"summary.json could not be read: {exc}")

d1, d2, d3 = st.tabs(["📊 Plots", "📋 Tables", "📦 Files"])

with d1:
    plots = run_dir / Layout.PLOTS
    images = sorted(plots.glob("*.png")) if plots.is_dir() else []
    previews = sorted((run_dir / Layout.PREVIEWS).glob("*.png")) \
        if (run_dir / Layout.PREVIEWS).is_dir() else []
    if not images and not previews:
        st.caption("No plots were produced (the run may have ended early).")
    for i in range(0, len(images), 2):
        cols = st.columns(2)
        for col, p in zip(cols, images[i:i + 2]):
            col.image(str(p), caption=p.stem, width="stretch")
    if previews:
        st.markdown("**The latest preview**")
        st.image(str(previews[-1]), width="stretch")

with d2:
    per_class = run_dir / Layout.PER_CLASS
    if per_class.is_file():
        st.markdown("**Per class**")
        st.dataframe(pd.read_csv(per_class), width="stretch", hide_index=True)
    df = runs.epoch_table(run_dir)
    if not df.empty:
        st.markdown("**Epoch history**")
        st.dataframe(df.iloc[::-1], width="stretch", hide_index=True, height=320)

with d3:
    files = [
        (Layout.REPORT, "HTML report", "text/html"),
        (Layout.METRICS_CSV, "metrics.csv", "text/csv"),
        (Layout.METRICS_XLSX, "metrics.xlsx",
         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        (Layout.CONFIG, "config.json", "application/json"),
        (Layout.SUMMARY, "summary.json", "application/json"),
        (Layout.ENV, "env.json", "application/json"),
    ]
    cols = st.columns(3)
    for i, (rel, label, mime) in enumerate(files):
        p = run_dir / rel
        if p.is_file():
            with cols[i % 3]:
                st.download_button(f"⬇️ {label}", p.read_bytes(),
                                   file_name=f"{r.run_name}_{p.name}", mime=mime,
                                   width="stretch", key=f"dl_{rel}")

    ckpts = sorted((run_dir / Layout.CHECKPOINTS).glob("*.pt")) \
        if (run_dir / Layout.CHECKPOINTS).is_dir() else []
    exports = sorted((run_dir / Layout.EXPORTS).glob("*")) \
        if (run_dir / Layout.EXPORTS).is_dir() else []
    if ckpts or exports:
        st.markdown("**Model files**")
        for p in ckpts + exports:
            st.markdown(f"- `{p.name}` — {p.stat().st_size / 1024**2:.1f} MB")
        faint("Checkpoints can be too large to download through the browser; "
              "you can reach them directly at the path above.")

    st.markdown(f"<span class='ts-mono ts-faint'>{run_dir}</span>", unsafe_allow_html=True)
    st.caption(f"Total size: {runs.dir_size_mb(run_dir):.1f} MB")

    with st.expander("🗑️ Delete this run"):
        st.warning("This cannot be undone — every checkpoint and output will be deleted.")
        confirm = st.text_input("Type the run name to confirm", key="del_confirm")
        if st.button("Delete permanently", disabled=confirm != r.run_name):
            if runs.delete_run(run_dir):
                # Survives the rerun that removes the run from the list.
                st.toast("Run deleted.")
                st.rerun()
            else:
                st.error("Could not delete.")
