"""Saved training configurations — presets — to start another model the same way.

A preset holds a run's hyperparameters and augmentation; it is stored as one JSON
file per preset under TRAINSTUDIO_HOME/presets (~/.trainstudio by default, the
mounted state volume in the container). Any past run is a preset too: its
config.json carries the same two objects.

Applying one to a different model is not a plain copy. Some values only make
sense for the model, machine or task they were chosen for, and are kept from the
current settings unless asked otherwise — see `apply_preset`.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from core import prefs
from core.capabilities import supports
from core.schemas import LOSS_CHOICES, AugConfig, Backend, Hyperparams, RunConfig, Task

PRESETS_DIR = prefs.PREFS_DIR / "presets"
PRESET_VERSION = 1

# Chosen for one architecture: copied only when asked for
MODEL_SPECIFIC = ("img_size", "batch_size", "lr", "min_lr", "encoder", "layer_decay",
                  "freeze_backbone_epochs", "accumulate_grad_batches")
# Chosen for one machine: never copied, the current recommendation stands
HARDWARE = ("amp", "channels_last", "num_workers", "compile_model")
# Meaningful for one task only
TASK_SPECIFIC = ("loss", "monitor_metric", "class_weights", "dice_weight", "label_smoothing",
                 "focal_gamma", "tversky_alpha", "tversky_beta")
AUG_CLASSIFICATION_ONLY = ("mixup", "cutmix", "coarse_dropout_p")
AUG_MEDICAL_ONLY = ("hu_shift", "hu_scale")

MODEL_SPECIFIC_LABEL = "batch size, learning rate, input size, encoder and the like"


@dataclass
class Preset:
    slug: str                   # file name for a saved preset, the run folder for a run
    name: str
    task: Task
    source_model: str
    hp: Hyperparams
    aug: AugConfig
    created_at: str = ""
    from_run: bool = False


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s[:60] or "preset"


def _path(slug: str) -> Path:
    return PRESETS_DIR / f"{slug}.json"


def exists(name: str) -> bool:
    return _path(slugify(name)).is_file()


def save_preset(name: str, hp: Hyperparams, aug: AugConfig, task: Task,
                source_model: str) -> Path:
    """Write (or overwrite) a preset; raises OSError if it cannot be written."""
    name = name.strip()
    if not name:
        raise ValueError("A preset needs a name")
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PRESET_VERSION, "name": name, "task": task.value,
        "source_model": source_model, "created_at": time.strftime("%Y-%m-%d %H:%M"),
        "hp": hp.model_dump(mode="json"), "aug": aug.model_dump(mode="json"),
    }
    path = _path(slugify(name))
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_preset(slug: str) -> Preset | None:
    try:
        d = json.loads(_path(slug).read_text(encoding="utf-8"))
        return Preset(slug=slug, name=d["name"], task=Task(d["task"]),
                      source_model=d.get("source_model", ""), created_at=d.get("created_at", ""),
                      hp=Hyperparams.model_validate(d["hp"]),
                      aug=AugConfig.model_validate(d["aug"]))
    except Exception:
        return None         # unreadable, or written by an incompatible version


def list_presets() -> list[Preset]:
    """Every readable preset, newest first."""
    if not PRESETS_DIR.is_dir():
        return []
    out = [p for f in PRESETS_DIR.glob("*.json") if (p := load_preset(f.stem)) is not None]
    return sorted(out, key=lambda p: p.created_at, reverse=True)


def delete_preset(slug: str) -> bool:
    try:
        _path(slug).unlink()
        return True
    except OSError:
        return False


def from_run(run_dir: str | Path) -> Preset | None:
    """A past run's configuration, as a preset."""
    try:
        cfg = RunConfig.load(Path(run_dir) / "config.json")
    except Exception:
        return None
    return Preset(slug=str(run_dir), name=cfg.run_name, task=cfg.dataset.task,
                  source_model=cfg.model.display_name or cfg.model.arch,
                  created_at=cfg.created_at, hp=cfg.hp, aug=cfg.aug, from_run=True)


def apply_preset(preset: Preset, hp: Hyperparams, aug: AugConfig, *, spec, task: Task,
                 medical: bool, include_model_specific: bool = False):
    """Lay a preset over the current settings.

    Returns `(hp, aug, applied, skipped)`: the new settings, the names of the
    fields that changed, and {field: reason} for preset values left out.
    """
    new_hp = hp.model_dump()
    new_aug = aug.model_dump()
    applied: list[str] = []
    skipped: dict[str, str] = {}
    native = spec.backend in (Backend.HF, Backend.ULTRALYTICS)

    def why_not(field: str, value) -> str | None:
        if field in HARDWARE:
            return "depends on this machine"
        if field in MODEL_SPECIFIC and not include_model_specific:
            return "model-specific"
        if field == "encoder" and not spec.needs_encoder:
            return f"{spec.display_name} has no encoder"
        if field in TASK_SPECIFIC and preset.task != task:
            return f"saved for {preset.task.label}"
        if field == "loss" and (native or value not in LOSS_CHOICES[task]):
            return f"not a loss {spec.display_name} can use"
        ok, reason = supports(spec, field)
        return None if ok else reason

    for field, value in preset.hp.model_dump().items():
        if value == new_hp.get(field):
            continue
        reason = why_not(field, value)
        if reason:
            skipped[field] = reason
        else:
            new_hp[field] = value
            applied.append(field)

    for field, value in preset.aug.model_dump().items():
        if value == new_aug.get(field):
            continue
        if field in AUG_CLASSIFICATION_ONLY and task != Task.CLASSIFICATION:
            skipped[field] = "classification only"
        elif field in AUG_MEDICAL_ONLY and not medical:
            skipped[field] = "CT/MR only"
        else:
            new_aug[field] = value
            applied.append(field)

    return (Hyperparams.model_validate(new_hp), AugConfig.model_validate(new_aug),
            applied, skipped)
