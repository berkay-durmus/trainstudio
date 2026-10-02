"""Settings — hyperparameters, augmentation, the output directory and starting a run.

Under each field you see the recommendation engine's rationale; when the user
changes a value the field is marked as "modified", so it is obvious at a glance
where the configuration deviates from the recommendation.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from core import prefs, presets, runs
from core.capabilities import supports
from core.hardware import detect
from core.launcher import LaunchError, start_run
from core.recommend import recommend
from core.registry import BACKEND_LABELS, FEATURED_ENCODERS
from core.schemas import (
    AUG_PRESETS,
    AugConfig,
    Backend,
    Hyperparams,
    LOSS_CHOICES,
    METRIC_HIGHER_IS_BETTER,
    ModelSelection,
    RunConfig,
    Task,
)
from ui import state
from ui.components import badges, hint, page_header
from ui.dir_picker import directory_picker

page_header(
    "Settings",
    "The recommendations come with their reasons — you can change any value you like.",
    active_step=3, done_steps=state.done_steps(),
)

ds = state.require_dataset()
spec = state.require_spec()
if ds is None or spec is None:
    st.stop()

dev = detect()

# ─────────────────────────────────────────────────────────────────────────────
# Compute / load the recommendation
# ─────────────────────────────────────────────────────────────────────────────

rec = state.recommendation()
if rec is None:
    rec = recommend(spec, ds, dev)
    state.put(state.K_REC, rec)
    state.put(state.K_HP, rec.hp.model_copy(deep=True))
    state.put(state.K_AUG, rec.aug.model_copy(deep=True))
    state.reset_touched()

hp: Hyperparams = state.hp()
aug: AugConfig = state.aug()
touched = state.touched()

if spec.needs_encoder:
    hp.encoder = state.get(state.K_ENCODER) or hp.encoder

# ── Presets ──────────────────────────────────────────────────────────────────


def preset_controls() -> None:
    """Save these settings under a name, or lay a saved preset / past run over them."""
    with st.popover("💾 Save preset", width="stretch"):
        name = st.text_input("Preset name", key="preset_name",
                             placeholder="e.g. strong-aug-cosine")
        clash = bool(name.strip()) and presets.exists(name)
        if clash:
            st.warning(f"A preset called “{name.strip()}” exists and will be replaced.")
        if st.button("Save", type="primary", key="preset_save", width="stretch",
                     disabled=not name.strip()):
            try:
                presets.save_preset(name, hp, aug, ds.task, spec.display_name)
                st.toast(f"Preset “{name.strip()}” saved")
            except (OSError, ValueError) as exc:
                st.error(f"Could not save: {exc}")

    with st.popover("📂 Load preset", width="stretch"):
        saved = presets.list_presets()
        past = [s for s in runs.list_runs_multi([state.output_dir()])
                if s.config is not None][:20]
        options = [("preset", p.slug) for p in saved] + [("run", s.run_dir) for s in past]
        if not options:
            st.caption("No saved presets or past runs yet. Save one with 💾.")
            return
        names = {("preset", p.slug): f"💾 {p.name} · {p.task.label} · {p.source_model}"
                 for p in saved}
        names.update({("run", s.run_dir): f"🗂️ {s.run_name} · {s.model}" for s in past})
        pick = st.selectbox("Load from", options, format_func=lambda o: names[o],
                            key="preset_pick")
        with_model = st.checkbox(f"Also apply the model-specific values ({presets.MODEL_SPECIFIC_LABEL})",
                                 value=False, key="preset_model_specific",
                                 help="Off: those values stay as recommended for "
                                      f"{spec.display_name}.")
        b1, b2 = st.columns(2)
        if b1.button("Apply", type="primary", key="preset_apply", width="stretch"):
            src = presets.load_preset(pick[1]) if pick[0] == "preset" else presets.from_run(pick[1])
            if src is None:
                st.error("That preset could not be read.")
                return
            new_hp, new_aug, applied, skipped = presets.apply_preset(
                src, hp, aug, spec=spec, task=ds.task, medical=ds.modality.is_medical,
                include_model_specific=with_model)
            state.put(state.K_HP, new_hp)
            state.put(state.K_AUG, new_aug)
            if "encoder" in applied:
                state.put(state.K_ENCODER, new_hp.encoder)
            for f in applied:
                state.mark_touched(f)
            msg = f"“{src.name}”: {len(applied)} setting(s) applied"
            if skipped:
                msg += f", {len(skipped)} kept as they were"
            st.toast(msg)
            state.put(state.K_PRESET_SKIPPED, skipped)
            st.rerun()
        if pick[0] == "preset" and b2.button("Delete", key="preset_delete", width="stretch"):
            presets.delete_preset(pick[1])
            st.rerun()


# ── Top strip ────────────────────────────────────────────────────────────────
top = st.container(border=True)
with top:
    a, b, c = st.columns([3, 3, 1.6])
    a.markdown(f"**{spec.display_name}**"
               + (f" · encoder `{hp.encoder}`" if spec.needs_encoder else ""))
    a.markdown(badges([
        (spec.released, "new" if spec.is_recent else ""),
        f"{spec.params_label} par.",
        BACKEND_LABELS[spec.backend],
        ds.task.label,
    ]), unsafe_allow_html=True)
    b.markdown(f"**{Path(ds.root).name}** · {ds.n_train:,} train / {ds.n_val:,} validation "
               f"· {ds.num_classes} classes")
    b.markdown(f"<span class='ts-faint'>{dev.label}</span>", unsafe_allow_html=True)
    with c:
        if st.button("✨ Restore recommendations", width="stretch",
                     help="Returns every field to the recommendation engine's values."):
            fresh = recommend(spec, ds, dev)
            state.put(state.K_REC, fresh)
            state.put(state.K_HP, fresh.hp.model_copy(deep=True))
            state.put(state.K_AUG, fresh.aug.model_copy(deep=True))
            state.reset_touched()
            st.rerun()
        preset_controls()

# Shown once, on the rerun right after a preset was applied
skipped = state.get(state.K_PRESET_SKIPPED)
if skipped:
    state.clear(state.K_PRESET_SKIPPED)
    with st.expander(f"ℹ️ {len(skipped)} setting(s) from the preset were kept as they were"):
        for f, why in sorted(skipped.items()):
            st.markdown(f"- `{f}` — {why}")

for w in rec.warnings:
    (st.warning if w.startswith("⚠️") else st.info)(w)

if touched:
    st.caption(f"Fields that deviate from the recommendation: {', '.join(sorted(touched))}")


# ─────────────────────────────────────────────────────────────────────────────
# Field helper — writes the value, shows the rationale, marks deviations
# ─────────────────────────────────────────────────────────────────────────────


def field(name: str, widget, *args, obj=None, why_key: str | None = None, **kwargs):
    """Draw one hyperparameter field and write any change back to the model.

    A field this backend cannot apply is shown disabled, with the reason in
    place of the recommendation's rationale.
    """
    target = obj if obj is not None else hp
    default = getattr(target, name)
    label = kwargs.pop("label", name)
    if name in touched:
        label = f"{label} ●"
    ok, why_not = supports(spec, name)
    if not ok:
        kwargs["disabled"] = True
    value = widget(label, *args, value=default, key=f"f_{name}", **kwargs)
    if ok and value != default:
        setattr(target, name, value)
        state.mark_touched(name)
    hint(why_not or rec.reason(why_key or name))
    return value


def select_field(name: str, options, *, obj=None, label: str, why_key: str | None = None,
                 format_func=str, help: str | None = None):
    target = obj if obj is not None else hp
    default = getattr(target, name)
    idx = list(options).index(default) if default in options else 0
    lbl = f"{label} ●" if name in touched else label
    ok, why_not = supports(spec, name)
    value = st.selectbox(lbl, options, index=idx, key=f"f_{name}",
                         format_func=format_func, help=help, disabled=not ok)
    if ok and value != default:
        setattr(target, name, value)
        state.mark_touched(name)
    hint(why_not or rec.reason(why_key or name))
    return value


def optional_field(name: str, label: str, widget, *, off: float, **kwargs):
    """A field whose model value is None when switched off; `off` stands for None."""
    current = getattr(hp, name)
    if name in touched:
        label = f"{label} ●"
    ok, why_not = supports(spec, name)
    value = widget(label, value=float(off if current is None else current),
                   key=f"f_{name}", disabled=not ok, **kwargs)
    new = None if abs(value - off) < 1e-9 else float(value)
    if ok and new != current:
        setattr(hp, name, new)
        state.mark_touched(name)
    hint(why_not or rec.reason(name))


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

t_basic, t_optim, t_loss, t_aug, t_runtime = st.tabs(
    ["🎯 Basics", "📉 Optimisation", "⚖️ Loss", "🔀 Augmentation", "🖥️ Runtime"]
)

with t_basic:
    c1, c2, c3 = st.columns(3)
    with c1:
        field("epochs", st.number_input, min_value=1, max_value=5000, step=1, label="Number of epochs")
        field("batch_size", st.number_input, min_value=1, max_value=1024, step=1,
              label="Batch size")
    with c2:
        field("img_size", st.number_input, min_value=32, max_value=2048, step=32,
              label="Input size (px)"
              if ds.task != Task.SEGMENTATION3D else "Patch size (voxels)")
        field("accumulate_grad_batches", st.number_input, min_value=1, max_value=128, step=1,
              label="Gradient accumulation")
    with c3:
        field("pretrained", st.toggle, label="Pretrained weights")
        hint("Training from scratch is almost always worse; only turn this off if the "
             "domain is very different.")
        field("freeze_backbone_epochs", st.number_input, min_value=0, max_value=1000, step=1,
              label="Freeze the backbone for (epochs)")

    if spec.needs_encoder:
        enc_names = [e for e, _ in FEATURED_ENCODERS]
        cur = hp.encoder or enc_names[0]
        idx = enc_names.index(cur) if cur in enc_names else 0
        enc = st.selectbox("Encoder (backbone)", enc_names, index=idx,
                           format_func=lambda e: dict(FEATURED_ENCODERS)[e])
        if enc != hp.encoder:
            hp.encoder = enc
            state.put(state.K_ENCODER, enc)
            state.mark_touched("encoder")
        hint(rec.reason("encoder"))

    eff = hp.batch_size * hp.accumulate_grad_batches
    steps = max(1, ds.n_train // max(1, hp.batch_size))
    st.markdown(
        f"<div class='ts-card ts-card-tight'>Effective batch <b>{eff}</b> · "
        f"<b>{steps:,}</b> steps per epoch · <b>{steps * hp.epochs:,}</b> steps in total</div>",
        unsafe_allow_html=True,
    )

with t_optim:
    c1, c2, c3 = st.columns(3)
    with c1:
        select_field("optimizer", ["adamw", "adam", "sgd", "rmsprop"], label="Optimizer")
        field("lr", st.number_input, min_value=1e-7, max_value=10.0, step=1e-5,
              format="%.2e", label="Learning rate")
        field("weight_decay", st.number_input, min_value=0.0, max_value=1.0, step=1e-4,
              format="%.5f", label="Weight decay")
    with c2:
        select_field("scheduler", ["cosine", "step", "plateau", "onecycle", "poly", "none"],
                     label="Learning rate schedule")
        field("warmup_epochs", st.number_input, min_value=0.0, max_value=100.0, step=0.5,
              label="Warmup (epochs)")
        field("min_lr", st.number_input, min_value=0.0, max_value=1.0, step=1e-7,
              format="%.2e", label="Minimum learning rate")
    with c3:
        if hp.optimizer in ("sgd", "rmsprop"):
            field("momentum", st.number_input, min_value=0.0, max_value=1.0, step=0.01,
                  label="Momentum")
        else:
            field("beta1", st.number_input, min_value=0.0, max_value=0.999, step=0.001,
                  format="%.3f", label="Beta 1")
            field("beta2", st.number_input, min_value=0.0, max_value=0.9999, step=0.0001,
                  format="%.4f", label="Beta 2")
        if hp.optimizer == "sgd":
            field("nesterov", st.toggle, label="Nesterov momentum")
        if hp.scheduler == "step":
            field("step_size", st.number_input, min_value=1, max_value=1000, step=1,
                  label="Step every (epochs)")
            field("step_gamma", st.number_input, min_value=0.01, max_value=1.0, step=0.05,
                  label="Step factor")
            hint("The learning rate is multiplied by this factor every step.")

    st.markdown("**🛡️ Regularisation & stability**")
    r1, r2, r3 = st.columns(3)
    with r1:
        field("drop_rate", st.number_input, min_value=0.0, max_value=0.9, step=0.05,
              format="%.2f", label="Dropout")
        field("drop_path_rate", st.number_input, min_value=0.0, max_value=0.9, step=0.05,
              format="%.2f", label="Stochastic depth (drop path)")
    with r2:
        field("ema", st.toggle, label="EMA of the weights")
        if hp.ema:
            field("ema_decay", st.number_input, min_value=0.9, max_value=0.99999,
                  step=0.0001, format="%.5f", label="EMA decay")
    with r3:
        optional_field("grad_clip", "Gradient clipping (0 = off)", st.number_input,
                       off=0.0, min_value=0.0, max_value=100.0, step=0.1)
        optional_field("layer_decay", "Layer-wise learning rate decay (1.0 = off)", st.slider,
                       off=1.0, min_value=0.3, max_value=1.0, step=0.05)

    with st.expander("Early stopping and monitoring"):
        e1, e2, e3 = st.columns(3)
        with e1:
            metrics = sorted(m for m in METRIC_HIGHER_IS_BETTER
                             if (m in ("accuracy", "balanced_accuracy", "f1_macro", "auroc")
                                 if ds.task == Task.CLASSIFICATION
                                 else m in ("dice_macro", "iou_macro", "dice", "iou"))
                             or m == "val_loss")
            select_field("monitor_metric", metrics, label="Monitored metric")
        with e2:
            field("early_stopping", st.toggle, label="Early stopping")
            field("patience", st.number_input, min_value=1, max_value=1000, step=1,
                  label="Patience (epochs)")
        with e3:
            field("val_interval", st.number_input, min_value=1, max_value=50, step=1,
                  label="Validation interval")
            field("save_last", st.toggle, label="Also keep the last checkpoint")

with t_loss:
    native = spec.backend in (Backend.HF, Backend.ULTRALYTICS)
    if native:
        st.info(f"{spec.family} uses its own internal loss function; "
                "loss selection is disabled for this backend.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if ds.task == Task.CLASSIFICATION:
                select_field("loss", list(LOSS_CHOICES[ds.task]), label="Loss function")
                select_field("class_weights", ["none", "balanced"], label="Class weights",
                             format_func=lambda v: {"none": "None", "balanced": "Balanced"}[v],
                             why_key="loss")
                field("label_smoothing", st.number_input, min_value=0.0, max_value=0.5,
                      step=0.01, label="Label smoothing")
            else:
                select_field("loss", list(LOSS_CHOICES[ds.task]),
                             label="Loss function")
                field("dice_weight", st.slider, min_value=0.0, max_value=1.0, step=0.05,
                      label="Dice weight", why_key="loss")
        with c2:
            if "focal" in hp.loss:
                field("focal_gamma", st.number_input, min_value=0.0, max_value=5.0, step=0.5,
                      label="Focal γ")
                hint("The larger γ is, the more the contribution of easy samples is "
                     "suppressed (typical: 2.0).")
            if hp.loss == "tversky":
                field("tversky_alpha", st.slider, min_value=0.0, max_value=1.0, step=0.05,
                      label="Tversky α (false positive penalty)")
                field("tversky_beta", st.slider, min_value=0.0, max_value=1.0, step=0.05,
                      label="Tversky β (false negative penalty)")
                hint("β > α penalises missed lesions more heavily than false alarms.")

        if ds.task == Task.CLASSIFICATION and ds.imbalance_ratio > 1.5:
            counts = ds.class_counts
            st.caption("Class distribution: " +
                       " · ".join(f"`{c}` {n:,}" for c, n in counts.items()))

with t_aug:
    # Not `presets`: that name is the saved-presets module imported above
    aug_presets = list(AUG_PRESETS) + ["custom"]
    p_idx = aug_presets.index(aug.preset) if aug.preset in aug_presets else 2
    preset = st.radio("Preset", aug_presets, index=p_idx, horizontal=True,
                      format_func=lambda p: {"none": "None", "light": "Light", "medium": "Medium",
                                             "heavy": "Heavy", "custom": "Custom"}[p])
    if preset != aug.preset and preset != "custom":
        new_aug = AugConfig(**AUG_PRESETS[preset])
        new_aug.normalize = aug.normalize
        new_aug.hu_shift, new_aug.hu_scale = aug.hu_shift, aug.hu_scale
        state.put(state.K_AUG, new_aug)
        state.mark_touched("aug_preset")
        st.rerun()
    hint(rec.reason("aug_preset"))
    if rec.reason("aug_modality"):
        st.caption(f"ℹ️ {rec.reason('aug_modality')}")
    if rec.reason("aug_elastic"):
        st.caption(f"ℹ️ {rec.reason('aug_elastic')}")

    def prob(name: str, label: str, hi: float = 1.0, step: float = 0.05, **kw):
        return field(name, st.slider, obj=aug, min_value=0.0, max_value=hi, step=step,
                     label=label, **kw)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**Geometric**")
        prob("hflip", "Horizontal flip")
        prob("vflip", "Vertical flip")
        prob("rot90", "90° rotation")
        prob("affine_p", "Affine transform probability")
        field("rotate_limit", st.slider, obj=aug, min_value=0, max_value=180, step=5,
              label="Rotation limit (°)")
        prob("scale_limit", "Scale limit", hi=0.5, step=0.01)
        prob("shift_limit", "Shift limit", hi=0.5, step=0.01)
    with g2:
        st.markdown("**Intensity**")
        prob("brightness_p", "Brightness/contrast probability")
        prob("brightness_contrast", "Brightness/contrast magnitude", hi=0.6)
        prob("gamma_p", "Gamma")
        prob("blur_p", "Blur")
        prob("noise_p", "Noise")
        prob("sharpen_p", "Sharpen")
        if ds.modality.is_medical:
            field("hu_shift", st.slider, obj=aug, min_value=0.0, max_value=100.0, step=5.0,
                  label="Intensity shift (HU)")
            prob("hu_scale", "Intensity scaling", hi=0.3, step=0.01)
    with g3:
        st.markdown("**Deformation and mixing**")
        prob("elastic_p", "Elastic deformation")
        prob("grid_distortion_p", "Grid distortion")
        if ds.task == Task.CLASSIFICATION:
            prob("coarse_dropout_p", "Region erasing (cutout)")
            prob("mixup", "MixUp α")
            prob("cutmix", "CutMix α")
        select_field("normalize", ["imagenet", "dataset", "minmax", "none"], obj=aug,
                     label="Normalisation",
                     format_func=lambda v: {"imagenet": "ImageNet statistics",
                                            "dataset": "Compute from the dataset",
                                            "minmax": "Min–max [0,1]", "none": "None"}[v])
    state.put(state.K_AUG, aug)

with t_runtime:
    c1, c2, c3 = st.columns(3)
    with c1:
        devices = ["auto"] + dev.all_devices
        cur_dev = state.get("flow.device", "auto")
        picked = st.selectbox("Device", devices,
                              index=devices.index(cur_dev) if cur_dev in devices else 0)
        state.put("flow.device", picked)
        hint(f"Automatic selection would use: {dev.torch_device} ({dev.label})")
        field("amp", st.toggle, label="Mixed precision (AMP)",
              disabled=not dev.supports_amp)
    with c2:
        field("num_workers", st.number_input, min_value=0, max_value=32, step=1,
              label="Data loading workers")
        field("seed", st.number_input, min_value=0, max_value=2**31 - 1, step=1,
              label="Random seed")
        hint("The same seed plus the same settings gives a reproducible result.")
    with c3:
        field("deterministic", st.toggle, label="Deterministic algorithms")
        hint("Guarantees full reproducibility but slows training down.")
        field("channels_last", st.toggle, label="channels_last memory format",
              disabled=not dev.supports_channels_last)
        field("preview_every_n_epochs", st.number_input, min_value=0, max_value=100, step=1,
              label="Preview frequency (0 = off)")
        field("compile_model", st.toggle, label="torch.compile",
              disabled=dev.kind != "cuda")
        hint("Compiles the model before training: faster steps after a slow first epoch. "
             "CUDA only." if dev.kind != "cuda" else
             "Faster steps after a slow first epoch; worth it for long runs.")

state.put(state.K_HP, hp)

# ─────────────────────────────────────────────────────────────────────────────
# Output and launch
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("## Output and launch")

out_col, name_col = st.columns([3, 2], gap="large")

with out_col:
    with st.expander("📂 Output folder", expanded=False):
        picked_out = directory_picker(
            key="output", label="The root folder the results are written to",
            initial=state.output_dir(), recent_kind="output", allow_create=True,
            help_text="Every run creates its own subfolder inside this folder.",
        )
    output_dir = picked_out or state.output_dir()
    state.put(state.K_OUTPUT, output_dir)
    st.markdown(f"<span class='ts-mono ts-dim'>{output_dir}</span>", unsafe_allow_html=True)

with name_col:
    default_name = state.get(state.K_RUN_NAME) or runs.suggest_run_name(
        spec.display_name, Path(ds.root).name, output_dir)
    run_name = st.text_input("Run name", value=default_name)
    state.put(state.K_RUN_NAME, run_name)
    notes = st.text_input("Note (optional)", placeholder="What are you testing in this run?")

run_dir = Path(output_dir) / run_name
exists = run_dir.exists() and any(run_dir.iterdir()) if run_dir.exists() else False
if exists:
    st.warning(f"`{run_dir}` already exists and is not empty.")
overwrite = st.checkbox("Overwrite", value=False, disabled=not exists)

ex1, ex2, ex3 = st.columns([1, 1, 2])
export_onnx = ex1.checkbox("Export to ONNX when training ends", value=False)
export_ts = ex2.checkbox("Export to TorchScript", value=False)


def build_config() -> RunConfig:
    return RunConfig(
        run_name=run_name,
        output_dir=output_dir,
        notes=notes,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        dataset=ds,
        model=ModelSelection(
            spec_id=spec.id, backend=spec.backend, arch=spec.arch,
            display_name=spec.display_name,
            encoder=hp.encoder if spec.needs_encoder else None,
            encoder_weights="imagenet" if spec.needs_encoder and hp.pretrained else None,
            weights=spec.weights,
        ),
        hp=hp, aug=aug,
        device=state.get("flow.device", "auto"),
        export_onnx=export_onnx, export_torchscript=export_ts,
    )


with st.expander("🔍 The config.json that will be created"):
    try:
        st.json(build_config().model_dump(mode="json"))
    except Exception as exc:
        st.error(f"The configuration is invalid: {exc}")

b1, b2 = st.columns([3, 1])
b1.markdown(
    f"<div class='ts-card ts-card-tight'><b>{spec.display_name}</b> · "
    f"{hp.epochs} epochs · batch {hp.batch_size} · lr {hp.lr:.2e} · {hp.loss} · "
    f"monitored metric <b>{hp.monitor_metric}</b></div>",
    unsafe_allow_html=True,
)

if b2.button("🚀 Start training", type="primary", width="stretch"):
    try:
        cfg = build_config()
        launched = start_run(cfg, overwrite=overwrite)
        prefs.set("output_dir", output_dir)
        prefs.push_recent("output", output_dir)
        state.put(state.K_ACTIVE_RUN, str(launched.run_dir))
        state.put(state.K_RUN_NAME, None)
        st.success(f"Started — PID {launched.pid}")
        st.switch_page("views/4_Training.py")
    except LaunchError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Could not start: {type(exc).__name__}: {exc}")
