"""Inference & Export — try a trained model out and convert it to a portable format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from core import prefs, runs
from core.schemas import Backend, Layout, Task
from data.readers import is_readable_image, list_files
from ui import state
from ui.charts import horizontal_bars
from ui.components import dim, empty_state, faint, fmt_metric, page_header
from ui.dir_picker import directory_picker

page_header("Inference & Export",
            "Try a trained model on your own images, or export it to ONNX/TorchScript.")

# ─────────────────────────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────────────────────────

roots = [state.output_dir()] + [p for p in prefs.recent("output") if p != state.output_dir()]
candidates = [r for r in runs.list_runs_multi(roots) if r.has(Layout.BEST)]

if not candidates:
    empty_state("🔍", "There is no usable model",
                "No run containing a checkpoint (`checkpoints/best.pt`) was found.",
                "Complete a training run first.")
    st.stop()

c1, c2, c3 = st.columns([3, 1.2, 1.2])
labels = {r.run_dir: f"{r.run_name} — {r.model}" for r in candidates}
default = state.get(state.K_ACTIVE_RUN)
idx = list(labels).index(default) if default in labels else 0
run_dir = c1.selectbox("Model", list(labels), index=idx, format_func=lambda d: labels[d])
which = c2.selectbox("Checkpoint", ["best", "last"],
                     format_func=lambda w: {"best": "Best", "last": "Last"}[w])
summary = next(r for r in candidates if r.run_dir == run_dir)


@st.cache_resource(show_spinner="Loading the model…", max_entries=3)
def _load(path: str, which: str):
    from export.inference import load_checkpoint

    return load_checkpoint(path, which)


try:
    loaded = _load(run_dir, which)
except Exception as exc:
    st.error(f"The model could not be loaded: {type(exc).__name__}: {exc}")
    st.stop()

c3.metric(summary.monitor_metric or "metric", fmt_metric(summary.best_value))
dim(f"{loaded.task.label} · {len(loaded.classes)} classes · input {loaded.cfg.hp.img_size}px "
    f"· device {loaded.device}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────────────────────────────────────

tab_single, tab_batch, tab_export = st.tabs(
    ["🖼️ Single image", "📁 Folder", "📦 Export"])

if loaded.task == Task.SEGMENTATION3D:
    st.info("Inference from the UI is not supported for 3D models yet; "
            "you can use `checkpoints/best.pt` directly with MONAI.")


def run_and_show(image: np.ndarray, key: str) -> None:
    from export.inference import gradcam, predict

    with st.spinner("Predicting…"):
        pred = predict(loaded, image)

    left, right = st.columns([1.3, 1])
    with left:
        if pred.overlay is not None:
            st.image(pred.overlay, caption="Prediction (mask overlaid)", width="stretch")
        else:
            st.image(image, caption="Input", width="stretch")
    with right:
        if pred.task == Task.CLASSIFICATION and pred.probs is not None:
            st.metric("Prediction", pred.label, f"{pred.confidence * 100:.1f}% confidence",
                      delta_color="off")
            order = np.argsort(-pred.probs)[:8]
            st.plotly_chart(
                horizontal_bars(
                    [loaded.classes[i] if i < len(loaded.classes) else str(i) for i in order],
                    [float(pred.probs[i]) for i in order],
                    x_title="Probability", x_range=(0, 1)),
                width="stretch", config={"displayModeBar": False})
        elif pred.class_areas:
            st.markdown("**Class areas** (percentage of the image)")
            for name, frac in sorted(pred.class_areas.items(), key=lambda kv: -kv[1]):
                st.markdown(f"<div class='ts-kv'><span class='ts-kv-k'>{name}</span>"
                            f"<span class='ts-kv-v'>{frac * 100:.2f}%</span></div>",
                            unsafe_allow_html=True)
        faint(f"Inference time: {pred.inference_ms:.0f} ms")

    if pred.task == Task.CLASSIFICATION and loaded.backend in (
            Backend.TIMM, Backend.TORCHVISION):
        if st.toggle("🔥 Grad-CAM — where is the model looking?", key=f"cam_{key}"):
            cam = gradcam(loaded, image,
                          int(pred.probs.argmax()) if pred.probs is not None else None)
            if cam is None:
                st.caption("Grad-CAM could not be produced for this architecture "
                           "(`pip install grad-cam` may be required).")
            else:
                st.image(cam, caption="Grad-CAM heatmap", width="stretch")


with tab_single:
    src = st.radio("Image source", ["Upload a file", "Pick from the dataset"],
                   horizontal=True, label_visibility="collapsed")

    image = None
    if src == "Upload a file":
        up = st.file_uploader("Image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "dcm"])
        if up is not None:
            import tempfile

            from export.inference import read_input

            suffix = Path(up.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
                fh.write(up.getbuffer())
                tmp = fh.name
            try:
                image = read_input(tmp, loaded)
            except Exception as exc:
                st.error(f"The image could not be read: {exc}")
    else:
        ds_root = Path(loaded.cfg.dataset.root)
        pool: list[Path] = []
        for split in ("test", "val", "train"):
            base = ds_root / split
            if not base.is_dir():
                continue
            if loaded.task == Task.CLASSIFICATION:
                for cdir in sorted(d for d in base.iterdir() if d.is_dir()):
                    pool.extend(list_files(cdir)[:20])
            else:
                pool.extend(list_files(base / "images")[:60])
            if pool:
                break
        if not pool:
            st.caption("The dataset used for training was not found on this machine.")
        else:
            pick = st.selectbox("Sample", pool[:200], format_func=lambda p: p.name)
            from export.inference import read_input

            try:
                image = read_input(pick, loaded)
            except Exception as exc:
                st.error(f"Could not read it: {exc}")

    if image is not None:
        run_and_show(image, "single")

with tab_batch:
    st.caption("Runs the prediction over every image in a folder and offers the result as a CSV.")
    with st.expander("📂 Select folder", expanded=True):
        folder = directory_picker(key="infer_dir", label="Image folder",
                                  initial=loaded.cfg.dataset.root)
    if folder:
        files = list_files(Path(folder), predicate=is_readable_image)
        st.markdown(f"**{len(files)}** images found.")
        limit = st.number_input("Maximum number of images to process", 1, 5000,
                                min(200, max(1, len(files))))
        if files and st.button("▶️ Start batch inference", type="primary"):
            import pandas as pd

            from export.inference import predict, read_input

            rows, bar = [], st.progress(0.0)
            subset = files[: int(limit)]
            for i, path in enumerate(subset):
                try:
                    p = predict(loaded, read_input(path, loaded))
                    row = {"file": path.name, "time_ms": round(p.inference_ms, 1)}
                    if p.task == Task.CLASSIFICATION and p.probs is not None:
                        row["prediction"] = p.label
                        row["confidence"] = round(p.confidence, 4)
                        for k, name in enumerate(loaded.classes):
                            if k < len(p.probs):
                                row[f"p_{name}"] = round(float(p.probs[k]), 4)
                    else:
                        for name, frac in p.class_areas.items():
                            row[f"area_{name}"] = round(frac, 5)
                    rows.append(row)
                except Exception as exc:
                    rows.append({"file": path.name, "error": str(exc)[:120]})
                bar.progress((i + 1) / len(subset))

            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch", height=380, hide_index=True)
            st.download_button("⬇️ Download the results as CSV",
                               df.to_csv(index=False).encode("utf-8"),
                               file_name=f"{summary.run_name}_predictions.csv", mime="text/csv")
            if "prediction" in df.columns:
                st.markdown("**Prediction distribution**")
                st.bar_chart(df["prediction"].value_counts())

with tab_export:
    st.caption("Convert the model file into a portable format so it can be used in other "
               "environments (C++, mobile, a server).")
    e1, e2 = st.columns(2)
    for col, fmt, label, note in (
        (e1, "onnx", "ONNX", "Readable by most runtimes (ONNX Runtime, TensorRT)."),
        (e2, "torchscript", "TorchScript", "For PyTorch environments without Python, and for C++."),
    ):
        with col:
            st.markdown(f"**{label}**")
            faint(note)
            if st.button(f"Produce {label}", key=f"exp_{fmt}", width="stretch"):
                try:
                    from export.inference import export_to

                    with st.spinner(f"Producing {label}…"):
                        path = export_to(loaded, fmt)
                    st.success(f"Written: `{path}` ({path.stat().st_size / 1024**2:.1f} MB)")
                except Exception as exc:
                    st.error(f"Failed: {type(exc).__name__}: {exc}")

    existing = sorted((Path(run_dir) / Layout.EXPORTS).glob("*")) \
        if (Path(run_dir) / Layout.EXPORTS).is_dir() else []
    if existing:
        st.markdown("**Existing exports**")
        for p in existing:
            size = p.stat().st_size / 1024**2
            a, b = st.columns([3, 1])
            a.markdown(f"`{p.name}` — {size:.1f} MB")
            if size < 200:
                b.download_button("⬇️ Download", p.read_bytes(), file_name=p.name,
                                  key=f"dl_{p.name}", width="stretch")
            else:
                b.caption("Fetch it directly from the path")
