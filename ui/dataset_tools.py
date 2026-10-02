"""The Dataset page's Analysis tab: the detailed analysis and size standardisation.

Kept out of views/1_Dataset.py, which is long enough already. The analysis runs
on request (the scan above it has to stay instant) and is kept in session state
for as long as the folder it describes is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core.schemas import Task
from data import standardize as std
from data.analysis import Analysis, analyze_dataset
from data.readers import read_image
from data.scan import ScanResult
from ui.charts import histogram, size_scatter
from ui.components import dim, issue_list
from ui.dir_picker import set_selection

_KEY = "_analysis"          # (root, mtime) → Analysis, page-local


def standardised_source(root: str) -> dict | None:
    """The report of a standardised copy, when `root` is one."""
    try:
        return json.loads((Path(root) / std.MARKER).read_text(encoding="utf-8"))
    except Exception:
        return None


def copy_banner(root: str) -> None:
    """Above the scan of a standardised copy: what it is, and the way back."""
    info = standardised_source(root)
    if not info:
        return
    c1, c2 = st.columns([4, 1])
    c1.info(f"📐 A standardised copy of `{Path(info['source']).name}` — every image "
            f"{info['size']}×{info['size']} ({info['method']}), written {info['created_at']}.")
    if Path(info["source"]).is_dir() and c2.button("↩ Back to the original", width="stretch"):
        set_selection("dataset", info["source"])
        st.cache_data.clear()
        st.rerun()


def render(res: ScanResult, mtime: float) -> None:
    if res.task not in (Task.CLASSIFICATION, Task.SEGMENTATION):
        st.caption("The detailed analysis covers 2D images. Volumes are summarised in the "
                   "Statistics tab, and resampled to a common spacing during training.")
        return

    key = (res.root, mtime)
    held = st.session_state.get(_KEY)
    a: Analysis | None = held[1] if held and held[0] == key else None

    c1, c2 = st.columns([4, 1])
    c1.caption("Reads every image header — size, channels, bit depth, format — and samples "
               "pixels, to find what the quick scan cannot. Nothing is written.")
    if c2.button("🔬 Run detailed analysis" if a is None else "🔄 Analyse again",
                 width="stretch", type="primary" if a is None else "secondary"):
        bar = st.progress(0.0, text="Starting…")
        a = analyze_dataset(res, progress=lambda f, t: bar.progress(min(1.0, f), text=t))
        st.session_state[_KEY] = (key, a)
        st.rerun()          # so the button above reads "Analyse again"
    if a is None:
        return

    _summary(a)
    st.markdown("**Findings**")
    issue_list(a.findings)
    if a.analysed:
        st.divider()
        _standardise(res, a)


def _summary(a: Analysis) -> None:
    span = a.size_span
    m = st.columns(5)
    m[0].metric("Images analysed", f"{a.analysed:,}",
                f"of {a.n_files:,}" if a.truncated else None, delta_color="off")
    m[1].metric("Distinct sizes", f"{len(a.sizes):,}")
    m[2].metric("Aspect ratio (W/H)",
                f"{min(a.aspects):.2f} – {max(a.aspects):.2f}" if a.aspects else "—")
    m[3].metric("In two splits", f"{len(a.cross_split):,}")
    m[4].metric("Unreadable", f"{len(a.unreadable):,}")
    if span:
        (w0, h0), (w1, h1) = span
        dim(f"Width {w0}–{w1} px · height {h0}–{h1} px · "
            f"{a.orientation.get('landscape', 0):,} landscape, "
            f"{a.orientation.get('portrait', 0):,} portrait, "
            f"{a.orientation.get('square', 0):,} square · "
            f"{a.total_bytes / 1024**2:,.1f} MB in total · analysed in {a.elapsed:.1f} s")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Image sizes (W × H)**")
        st.plotly_chart(size_scatter(a.sizes), width="stretch", config={"displayModeBar": False})
    with g2:
        st.markdown("**Aspect ratios**")
        st.plotly_chart(histogram(a.aspects, "Width / height"), width="stretch",
                        config={"displayModeBar": False})

    t1, t2, t3, t4 = st.columns(4)
    for col, title, counter in ((t1, "Format", a.formats), (t2, "Mode", a.modes),
                                (t3, "Bits per pixel", a.bits)):
        with col:
            st.markdown(f"**{title}**")
            st.dataframe(pd.DataFrame(counter.most_common(), columns=[title, "Images"]),
                         hide_index=True, width="stretch")
    with t4:
        if a.class_sizes:
            st.markdown("**Median size per class**")
            st.dataframe(pd.DataFrame([(c, f"{w}×{h}") for c, (w, h) in a.class_sizes.items()],
                                      columns=["Class", "W × H"]), hide_index=True, width="stretch")
        elif a.class_pixels:
            st.markdown("**Pixels per label**")
            st.dataframe(pd.DataFrame([(k, f"{v:.1%}") for k, v in a.class_pixels.items()],
                                      columns=["Label", "Share"]), hide_index=True, width="stretch")
            dim(f"From {a.mask_sample:,} masks · {a.empty_masks:,} empty")


def _standardise(res: ScanResult, a: Analysis) -> None:
    st.markdown("### 📐 Standardise image size")
    if a.uniform_size:
        w, h = next(iter(a.sizes))
        st.caption(f"Every image is already {w}×{h}; there is nothing to standardise.")
        return
    dim("Writes a copy of the dataset in which every image has the same size, next to "
        "the original — which is left exactly as it is. The copy is then selected here.")

    c1, c2 = st.columns([3, 2])
    with c1:
        method = st.radio("Method", list(std.METHODS), format_func=std.METHODS.get,
                          key="std_method")
    with c2:
        auto = std.default_size(a)
        options = sorted(set(std.SIZES) | {auto})
        size = st.selectbox("Size (px, square)", options, index=options.index(auto),
                            key="std_size",
                            format_func=lambda s: f"{s} · median" if s == auto else str(s))
    _preview(res, a, size, method)

    default = str(std.default_out(res.root, size, method))
    out = st.text_input("Write the copy to", value=default, key=f"std_out_{size}_{method}")
    overwrite = False
    if (Path(out) / std.MARKER).is_file():
        overwrite = st.checkbox("Replace the earlier standardised copy there", key="std_over")
    need = std.estimate_bytes(a, size)
    problem = std.check(res, out, overwrite, needed_bytes=need)
    st.caption(f"About {need / 1024**2:,.0f} MB · {a.n_files:,} images"
               + (" and their masks" if res.task == Task.SEGMENTATION else "")
               + (" · the padding in masks is ignored by the loss"
                  if res.task == Task.SEGMENTATION and method == "letterbox" else ""))
    if problem:
        st.warning(problem)
    if st.button("📐 Standardise", type="primary", disabled=bool(problem), key="std_go"):
        bar = st.progress(0.0, text="Writing the copy…")
        try:
            rep = std.standardize(res, size, method, out, overwrite=overwrite,
                                  progress=lambda f, t: bar.progress(min(1.0, f), text=t))
        except std.StandardizeError as exc:
            bar.empty()
            st.error(str(exc))
            return
        msg = f"Standardised copy written: {rep.images:,} images at {size}×{size}"
        if rep.unchanged:
            msg += f" ({len(rep.unchanged)} unreadable file(s) copied unchanged)"
        st.toast(msg)
        set_selection("dataset", rep.out_root)
        st.session_state.pop(_KEY, None)
        st.cache_data.clear()
        st.rerun()


def _preview(res: ScanResult, a: Analysis, size: int, method: str) -> None:
    """One image before and after — the one whose shape is furthest from square."""
    from data.analysis import collect

    entries = collect(res)
    if not entries:
        return
    try:
        pick = max(entries[:400], key=lambda e: abs(_aspect(e.path) - 1))
        before = read_image(pick.path)
        after = std.transform_image(before, size, method)
    except Exception:
        return
    b1, b2, _ = st.columns([1, 1, 2])
    b1.image(before, caption=f"Before · {before.shape[1]}×{before.shape[0]}", width="stretch")
    b2.image(after, caption=f"After · {size}×{size}", width="stretch")


def _aspect(path: Path) -> float:
    from PIL import Image

    try:
        with Image.open(path) as im:
            w, h = im.size
        return w / h
    except Exception:
        return 1.0
