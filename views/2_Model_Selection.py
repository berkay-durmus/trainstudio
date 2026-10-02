"""Model Selection — the catalogue, with the most recent models in the literature on top."""

from __future__ import annotations

import streamlit as st

from core.hardware import detect, usable_vram_gb
from core.registry import (
    BACKEND_LABELS,
    DEFAULT_ENCODER,
    FEATURED_ENCODERS,
    SIZE_CLASS_LABELS,
    ModelSpec,
    backends_for,
    query,
)
from core.schemas import Backend, Task
from ui import state
from ui.components import badges, empty_state, faint, page_header

page_header(
    "Model Selection",
    "The catalogue is sorted by release date — the newest architectures come first.",
    active_step=2, done_steps=state.done_steps(),
)

ds = state.require_dataset()
if ds is None:
    st.stop()

dev = detect()
budget = usable_vram_gb(dev)

# ─────────────────────────────────────────────────────────────────────────────
# Filters
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    f"<div class='ts-card ts-card-tight'>Models suitable for the "
    f"<b>{ds.task.label}</b> task · dataset "
    f"<span class='ts-mono'>{ds.root.split('/')[-1]}</span> "
    f"· {ds.n_total:,} samples · {ds.num_classes} classes</div>",
    unsafe_allow_html=True,
)

f1, f2, f3, f4 = st.columns([2, 2, 2, 1.6])

search = f1.text_input("Search", placeholder="model, family or strength…",
                       label_visibility="collapsed")

avail_backends = backends_for(ds.task)
picked_backends = f2.multiselect(
    "Library", avail_backends, default=avail_backends,
    format_func=lambda b: BACKEND_LABELS[b], label_visibility="collapsed",
    placeholder="Library",
)

picked_sizes = f3.multiselect(
    "Size", list(SIZE_CLASS_LABELS), default=[],
    format_func=lambda s: SIZE_CLASS_LABELS[s], label_visibility="collapsed",
    placeholder="Model size",
)

fit_only = f4.toggle("Fits this hardware", value=False,
                     help=f"Only models that fit within a ~{budget:.0f} GB memory budget")

results = query(
    task=ds.task,
    backends=set(picked_backends) if picked_backends else None,
    size_classes=set(picked_sizes) if picked_sizes else None,
    max_vram_gb=budget if fit_only else None,
    text=search,
)

# ─────────────────────────────────────────────────────────────────────────────
# The selected model summary (sticky)
# ─────────────────────────────────────────────────────────────────────────────

chosen = state.spec()
if chosen and chosen.task == ds.task:
    box = st.container(border=True)
    with box:
        a, b, c = st.columns([4, 3, 1.4])
        a.markdown(f"**Selected:** {chosen.display_name}")
        a.markdown(badges([
            (chosen.released, "new" if chosen.is_recent else "accent"),
            f"{chosen.params_label} parameters",
            BACKEND_LABELS[chosen.backend],
            f"~{chosen.min_vram_gb:.0f} GB",
        ]), unsafe_allow_html=True)
        with b:
            if chosen.needs_encoder:
                enc_names = [e for e, _ in FEATURED_ENCODERS]
                cur = state.get(state.K_ENCODER) or DEFAULT_ENCODER
                idx = enc_names.index(cur) if cur in enc_names else enc_names.index(DEFAULT_ENCODER)
                enc = st.selectbox(
                    "Encoder (backbone)", enc_names, index=idx,
                    format_func=lambda e: dict(FEATURED_ENCODERS)[e],
                    key="encoder_select",
                    help="The feature-extraction backbone of the segmentation architecture. "
                         "The architecture and the backbone are chosen independently.",
                )
                state.put(state.K_ENCODER, enc)
            else:
                faint(chosen.summary)
        with c:
            if st.button("Continue →", type="primary", width="stretch"):
                st.switch_page("views/3_Settings.py")

st.caption(f"{len(results)} models · newest first")

# ─────────────────────────────────────────────────────────────────────────────
# The card gallery
# ─────────────────────────────────────────────────────────────────────────────


def render_card(spec: ModelSpec, col) -> None:
    fits = spec.min_vram_gb <= budget
    with col.container(border=True):
        title, pick = st.columns([3, 1.1])
        title.markdown(f"**{spec.display_name}**")
        with pick:
            is_sel = chosen is not None and chosen.id == spec.id
            if st.button("✓ Selected" if is_sel else "Select", key=f"pick_{spec.id}",
                         type="primary" if is_sel else "secondary",
                         width="stretch"):
                state.set_spec(spec)
                if spec.needs_encoder:
                    state.put(state.K_ENCODER, spec.rec.get("encoder") or DEFAULT_ENCODER)
                st.rerun()

        st.markdown(badges([
            (spec.released, "new" if spec.is_recent else ""),
            f"{spec.params_label} par.",
            BACKEND_LABELS[spec.backend],
            (f"~{spec.min_vram_gb:.0f} GB", "" if fits else "accent"),
        ]), unsafe_allow_html=True)

        st.markdown(f"<div class='ts-dim' style='min-height:3.2rem'>{spec.summary}</div>",
                    unsafe_allow_html=True)

        if spec.strengths:
            st.markdown(badges([(s, "accent") for s in spec.strengths[:3]]),
                        unsafe_allow_html=True)
        if not fits:
            st.caption(f"⚠️ Borderline on this machine (~{budget:.0f} GB) — the batch size will be small.")
        if spec.paper_url:
            st.markdown(f"<a class='ts-faint' href='{spec.paper_url}' target='_blank'>"
                        f"source ↗</a>", unsafe_allow_html=True)


if not results:
    empty_state("🔍", "No models match the filters",
                "Clear the search or turn off the hardware filter.")
else:
    st.markdown("<div class='ts-scroll' style='max-height:none'>", unsafe_allow_html=True)
    N = 3
    for i in range(0, len(results), N):
        cols = st.columns(N, gap="small")
        for spec, col in zip(results[i:i + N], cols):
            render_card(spec, col)
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Advanced: a model outside the catalogue
# ─────────────────────────────────────────────────────────────────────────────

if ds.task == Task.CLASSIFICATION:
    with st.expander("⚙️ Advanced — use a model outside the catalogue"):
        st.caption("The catalogue is a curated selection. Both libraries carry far more "
                   "pretrained models than are listed, so if you know the exact name you "
                   "can pick it here.")

        lib = st.radio("Library", ["timm", "torchvision"], horizontal=True,
                       key="custom_lib",
                       captions=["~1000 models, the widest choice",
                                 "the reference implementations"])

        @st.cache_data(show_spinner=False)
        def _custom_models(library: str) -> list[str]:
            """Pretrained classification model names the installed library offers."""
            try:
                if library == "timm":
                    import timm

                    return sorted(timm.list_models(pretrained=True))
                import torchvision.models as tvm

                # `list_models` on the classification module excludes the
                # detection/segmentation/video builders, which take different
                # arguments and are not classifiers.
                return sorted(tvm.list_models(module=tvm))
            except Exception:
                return []

        names = _custom_models(lib)
        if not names:
            st.warning(f"{lib} is not installed, or its model list could not be read.")
        else:
            placeholder = "convnext, eva, swin…" if lib == "timm" else "resnet, convnext, vit…"
            q = st.text_input("Search for a model name", placeholder=placeholder,
                              key=f"custom_search_{lib}")
            matches = [n for n in names if q.lower() in n.lower()][:200] if q else names[:200]
            picked = st.selectbox(f"Choose from {len(names):,} pretrained models", matches,
                                  key=f"custom_pick_{lib}")
            if st.button("Use this model"):
                backend = Backend.TIMM if lib == "timm" else Backend.TORCHVISION
                custom = ModelSpec(
                    id=f"{lib}::{picked}", display_name=picked, task=Task.CLASSIFICATION,
                    backend=backend, arch=picked, family=f"{lib} (custom)",
                    released="0000-00", params_m=25.0, default_img_size=224,
                    min_vram_gb=8.0, size_class="base",
                    # The size figures above are a placeholder: the real ones are
                    # only known once the model is built, and the recommendation
                    # engine treats them as a hint rather than a constraint.
                    summary=f"A model picked straight from {lib}, outside the catalogue.",
                    weights="DEFAULT" if backend == Backend.TORCHVISION else None,
                    rec=dict(optimizer="adamw", lr=3e-4, weight_decay=0.05),
                )
                state.set_spec(custom)
                # A toast outlives the rerun; a success box is discarded by it.
                st.toast(f"{picked} selected.")
                st.rerun()
