"""Dataset — folder selection, canonical structure validation, statistics and previews."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.schemas import Modality, Task, WindowSpec
from data import spec as dspec
from data.readers import CT_WINDOW_PRESETS, read_image, read_mask
from data.scan import scan_dataset
from data.splitter import make_splits, split_summary
from ui import dataset_tools, state
from ui.charts import class_distribution
from ui.components import (
    dim,
    empty_state,
    fmt_count,
    issue_list,
    page_header,
)
from ui.dir_picker import directory_picker, set_selection

page_header(
    "Dataset",
    "Select a dataset that already exists on this machine — the files are read in place, never copied.",
    active_step=1, done_steps=state.done_steps(),
)

# ─────────────────────────────────────────────────────────────────────────────
# 1 · Folder selection
# ─────────────────────────────────────────────────────────────────────────────

current = state.dataset()
picker_col, help_col = st.columns([3, 2], gap="large")

with picker_col:
    with st.expander("📁 Select folder", expanded=current is None):
        selected = directory_picker(
            key="dataset",
            label="Dataset root folder",
            initial=current.root if current else None,
            recent_kind="dataset",
            help_text="Select the root folder — the one that contains `train/`.",
        )
    if current and not selected:
        selected = current.root

with help_col:
    st.markdown("**Expected folder structure**")
    task_tab = st.radio(
        "Task", [t for t in Task], horizontal=True, label_visibility="collapsed",
        format_func=lambda t: t.label, key="layout_help_task",
    )
    st.code(dspec.LAYOUT_HELP[task_tab], language="text")
    dim("Give the application this structure and every model will work. For backends that "
        "want a different format, such as YOLO, the conversion is handled for you.")
    with st.expander("Do the folder names have to match exactly?"):
        st.markdown(dspec.ALIAS_HELP)

if not selected:
    empty_state("📁", "No dataset selected",
                "Use the folder picker above to select your dataset's root folder.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 2 · Scanning
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
head, opts, act = st.columns([3, 2, 1])
head.markdown(f"## Validation\n<span class='ts-mono ts-dim'>{selected}</span>",
              unsafe_allow_html=True)

force_task = opts.selectbox(
    "Task (detected automatically)",
    ["Automatic"] + [t.label for t in Task],
    key="force_task",
)
if act.button("🔄 Rescan", width="stretch"):
    st.cache_data.clear()


@st.cache_data(show_spinner="Scanning the dataset…", max_entries=8)
def _scan(root: str, task_label: str, mtime: float):
    task = next((t for t in Task if t.label == task_label), None)
    return scan_dataset(root, task=task)


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _root_mtime(root: str) -> float:
    """The modification time of the root and split folders — the cache key.

    Permission-safe on purpose: this runs *before* the scan, so an unreadable
    folder has to become a stale-but-harmless 0.0 here and be reported as an
    issue by the scan itself, not crash the page on the way in.
    """
    p = Path(root)
    stamps = [_mtime(p)]
    for s in dspec.SPLITS:
        d = dspec.split_dir(p, s)
        if dspec.is_dir(d):
            stamps.append(_mtime(d))
    return max(stamps)


res = _scan(selected, force_task, _root_mtime(selected))

dataset_tools.layout_popup(res, _root_mtime(selected))

if res.task is None:
    issue_list(res.issues)
    # Selecting the folder that *contains* the datasets is the usual mis-click;
    # offer those datasets instead of making the user walk the picker again.
    if res.candidates:
        st.markdown("**Datasets found inside this folder**")
        for i, cand in enumerate(res.candidates):
            if st.button(f"📁 {Path(cand).name}", key=f"cand{i}", width="stretch"):
                set_selection("dataset", cand)
                st.cache_data.clear()
                st.rerun()
    st.stop()

# ── Summary strip ────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Task", res.task.label)
m2.metric("Train", fmt_count(res.n_train))
m3.metric("Validation", fmt_count(res.n_val))
m4.metric("Test", fmt_count(res.n_test) if res.n_test else "—")
m5.metric("Classes", res.classes and len(res.classes) or "—")

# Written by the automatic split below, which has to rerun to pick the new
# splits up — and `st.rerun()` discards anything rendered before it, so the
# summary is handed over through session state and shown here instead.
split_note = st.session_state.pop("_split_summary", None)
if split_note:
    st.success(split_note)

dataset_tools.copy_banner(selected)
if res.ok:
    st.success(f"The structure is valid — {res.total:,} samples, {len(res.classes)} classes "
               f"(scanned in {res.elapsed * 1000:.0f} ms, {res.sampled} files sampled).")
else:
    st.error(f"Training cannot start until {len(res.errors)} issue(s) are resolved.")

issue_list(res.issues)

# ─────────────────────────────────────────────────────────────────────────────
# 3 · Details
# ─────────────────────────────────────────────────────────────────────────────

tab_stats, tab_analysis, tab_preview, tab_yaml = st.tabs(
    ["📊 Statistics", "🔬 Analysis", "🖼️ Preview", "📝 dataset.yaml"])

with tab_stats:
    left, right = st.columns([3, 2], gap="large")

    with left:
        if res.class_counts:
            st.markdown("**Class distribution**")
            st.plotly_chart(
                class_distribution(res.class_counts, res.classes),
                width="stretch", config={"displayModeBar": False},
            )
            if res.imbalance_ratio >= 2:
                dim(f"Imbalance ratio {res.imbalance_ratio:.1f}× — "
                    "the recommendation engine will tune the loss function accordingly.")
        elif res.mask_values:
            st.markdown("**Mask statistics**")
            st.markdown(f"- Label values found: `{res.mask_values}`")
            st.markdown(f"- Mean foreground ratio: **{(res.mask_fg_ratio or 0) * 100:.2f}%**")
            st.markdown(f"- Classes: {', '.join(f'`{i}` = {c}' for i, c in enumerate(res.classes))}")

    with right:
        st.markdown("**Image properties**")
        rows = [("Modality", res.modality.label)]
        # The scan keeps (H, W); shown W × H, like the Analysis tab
        if res.median_size:
            rows.append(("Median size (W × H)", f"{res.median_size[1]} × {res.median_size[0]}"))
        if res.size_range and res.size_range[0] != res.size_range[1]:
            lo, hi = res.size_range
            rows.append(("Size range (W × H)", f"{lo[1]}×{lo[0]} – {hi[1]}×{hi[0]}"))
        rows.append(("Channels", res.channels))
        if res.volume_shape:
            rows.append(("Volume shape (D×H×W)", "×".join(map(str, res.volume_shape))))
            rows.append(("Voxel spacing (mm)", " × ".join(f"{s:g}" for s in res.volume_spacing or ())))
        if res.intensity_range:
            rows.append(("Intensity (1st–99th pct)",
                         f"{res.intensity_range[0]:g} … {res.intensity_range[1]:g}"))
        for k, v in rows:
            st.markdown(f"<div class='ts-kv'><span class='ts-kv-k'>{k}</span>"
                        f"<span class='ts-kv-v'>{v}</span></div>", unsafe_allow_html=True)
        if res.size_range and res.size_range[0] != res.size_range[1]:
            dim("Sizes differ: the 🔬 Analysis tab shows how, and can write a copy "
                "in which they are all the same.")

# ── Modality / windowing ─────────────────────────────────────────────────────
# A keyed widget keeps its own value across reruns and ignores `index=`/`value=`
# from then on. That is right for a choice the user made and wrong for a
# detected default, so the detected values are written into the widget state
# whenever what they were detected *from* changes.
st.markdown("")
if st.session_state.get("_detected_for") != (res.root, res.modality.value):
    st.session_state["_detected_for"] = (res.root, res.modality.value)
    st.session_state["modality"] = res.modality

mod_col, win_col = st.columns([1, 3])
modality = mod_col.selectbox(
    "Modality", list(Modality), format_func=lambda m: m.label, key="modality",
    help="Augmentation and normalisation are tuned differently for medical modalities.",
)

window: WindowSpec | None = None
if modality.is_medical:
    with win_col:
        p1, p2, p3 = st.columns([2, 1, 1])
        presets = list(CT_WINDOW_PRESETS)
        preset = p1.selectbox("Window preset", presets + ["Custom"], key="win_preset")
        # Choosing a preset has to move the two number inputs; they are keyed, so
        # the only way to change them is through session state, before they render.
        if preset != "Custom" and st.session_state.get("_win_preset_applied") != preset:
            st.session_state["_win_preset_applied"] = preset
            base = CT_WINDOW_PRESETS[preset]
            st.session_state["win_c"] = float(base.center)
            st.session_state["win_w"] = float(base.width)
        st.session_state.setdefault("win_c", float(WindowSpec().center))
        st.session_state.setdefault("win_w", float(WindowSpec().width))

        center = p2.number_input("Center (HU)", step=10.0, key="win_c")
        width = p3.number_input("Width (HU)", step=25.0, min_value=1.0, key="win_w")
        window = WindowSpec(center=center, width=width)

with tab_preview:
    if not res.preview:
        st.caption("No samples were found to preview.")
    else:
        st.caption("Randomly selected samples — use these to confirm that the reading and "
                   "windowing settings are right.")
        cols = st.columns(min(4, len(res.preview)))
        for col, item in zip(cols, res.preview[:4]):
            with col:
                try:
                    if res.task == Task.SEGMENTATION3D:
                        from data.readers import read_volume
                        import numpy as np

                        vol, _ = read_volume(item["image"])
                        lab, _ = read_volume(item["mask"])
                        z = int(np.argmax((lab > 0).sum(axis=(1, 2)))) or vol.shape[0] // 2
                        from data.readers import apply_window

                        img = apply_window(vol[z], window)
                        img = np.stack([img] * 3, axis=-1)
                        msk = lab[z]
                    else:
                        img = read_image(item["image"], modality, window)
                        msk = read_mask(item["mask"]) if item.get("mask") else None

                    if msk is not None:
                        import numpy as np

                        overlay = img.copy()
                        fg = msk > 0
                        overlay[fg] = (0.55 * overlay[fg] +
                                       0.45 * np.array([45, 212, 191])).astype("uint8")
                        st.image(overlay, width="stretch")
                        st.caption(f"{item['label']} · foreground {100 * fg.mean():.2f}%")
                    else:
                        st.image(img, width="stretch")
                        st.caption(item["label"])
                except Exception as exc:
                    st.warning(f"The preview failed: {exc}")

with tab_analysis:
    dataset_tools.render(res, _root_mtime(selected))

with tab_yaml:
    existing = dspec.read_dataset_yaml(res.root)
    if existing:
        st.success("This dataset has a `dataset.yaml` and it was read successfully.")
        st.code(existing.to_yaml(), language="yaml")
    else:
        st.info("There is no `dataset.yaml` — the structure was inferred from the folders. "
                "You can create one to make the class names and modality permanent.")

    st.markdown("**Class names**")
    # The key carries the dataset identity: with a fixed key the editor would go
    # on showing the class names of whichever dataset was scanned first.
    edited = st.data_editor(
        [{"index": i, "class name": c} for i, c in enumerate(res.classes)],
        width="stretch", hide_index=True,
        key=f"class_editor::{res.root}::{len(res.classes)}",
        column_config={"index": st.column_config.NumberColumn(disabled=True)},
    )
    new_classes = [row["class name"] for row in edited]

    if st.button("💾 Create / update dataset.yaml", type="primary"):
        try:
            path = dspec.write_dataset_yaml(res.root, dspec.DatasetYaml(
                task=res.task, modality=modality, classes=new_classes, window=window,
            ))
            # `st.toast` survives the rerun below; `st.success` would not.
            st.toast(f"Written: {path}")
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Could not write the file: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# 4 · Automatic split
# ─────────────────────────────────────────────────────────────────────────────

if res.needs_split:
    st.divider()
    st.markdown("## Automatic split")
    st.caption("Only `train/` was found. The files are not copied — a `splits.json` is "
               "created at the root and training reads from it.")
    b1, b2, b3, b4 = st.columns([1, 1, 1, 1.4])
    val_frac = b1.slider("Validation share", 0.05, 0.4, 0.2, 0.05)
    test_frac = b2.slider("Test share", 0.0, 0.3, 0.1, 0.05)
    seed = b3.number_input("Random seed", value=42, step=1)
    with b4:
        st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        if st.button("✂️ Split and save", type="primary", width="stretch"):
            try:
                info = make_splits(res, val_frac=val_frac, test_frac=test_frac, seed=int(seed))
                st.session_state["_split_summary"] = split_summary(info)
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"The split failed: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# 5 · Confirmation
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
ok_col, btn_col = st.columns([3, 1])

with ok_col:
    if res.ok:
        dim(f"{res.total:,} samples ready · {res.task.label} · {modality.label}")
    else:
        st.error("Resolve the errors above, then rescan.")

with btn_col:
    if st.button("Continue →", type="primary", width="stretch", disabled=not res.ok):
        ds_cfg = res.to_dataset_config(window=window)
        ds_cfg.modality = modality
        if (Path(res.root) / "splits.json").is_file():
            ds_cfg.splits_file = str(Path(res.root) / "splits.json")
        state.set_dataset(ds_cfg, window)
        state.put(state.K_SCAN, res)
        set_selection("dataset", res.root)
        st.switch_page("views/2_Model_Selection.py")
