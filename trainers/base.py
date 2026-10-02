"""The training loop shared by every backend.

Subclasses define only three things: how to build the model, how to run one batch
forward, and which metric accumulator to use. Event emission, checkpointing, early
stopping, the stop signal and reporting are solved here once — which is what lets
the Training page work independently of the backend.
"""

from __future__ import annotations

import math
import random
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.events import E, EventWriter
from core.hardware import detect, telemetry
from core.schemas import Hyperparams, Layout, RunConfig, RunStatus, Task, metric_mode
from data.seeding import SeededDataset, SeededSampler

SYSTEM_EVENT_INTERVAL_S = 5.0
BATCH_EVENT_MIN_INTERVAL_S = 0.25


class StopRequested(Exception):
    """The graceful stop signal from the UI — a controlled exit, not an error."""


class PauseRequested(StopRequested):
    """The UI asked to stop at the end of the epoch; nothing is lost."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(requested: str) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device(detect().torch_device)


def build_optimizer(model: nn.Module, hp: Hyperparams) -> torch.optim.Optimizer:
    """Weight decay is applied to weights only; norm and bias layers are excluded
    (otherwise small models lose noticeable accuracy)."""
    if hp.layer_decay and hp.layer_decay < 1.0:
        groups = _layer_decay_groups(model, hp)
        if groups is not None:
            return _make_optimizer(groups, hp)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": hp.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return _make_optimizer(groups, hp)


def _make_optimizer(groups, hp: Hyperparams) -> torch.optim.Optimizer:
    if hp.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=hp.lr, momentum=hp.momentum,
                               nesterov=hp.nesterov and hp.momentum > 0)
    if hp.optimizer == "adam":
        return torch.optim.Adam(groups, lr=hp.lr, betas=(hp.beta1, hp.beta2))
    if hp.optimizer == "rmsprop":
        return torch.optim.RMSprop(groups, lr=hp.lr, momentum=hp.momentum)
    return torch.optim.AdamW(groups, lr=hp.lr, betas=(hp.beta1, hp.beta2))


def _layer_decay_groups(model: nn.Module, hp: Hyperparams):
    """timm's layer-wise decay helper — silently disabled when unavailable."""
    try:
        from timm.optim import param_groups_layer_decay

        return param_groups_layer_decay(
            model, weight_decay=hp.weight_decay, layer_decay=hp.layer_decay,
        )
    except Exception:
        return None


def build_scheduler(optimizer, hp: Hyperparams, steps_per_epoch: int):
    """A per-step schedule. Warmup is prepended to every schedule."""
    total_steps = max(1, steps_per_epoch * hp.epochs)
    # If warmup exceeds half of the whole run the learning rate never decays; that
    # happens easily when the epoch count is lowered by hand after a recommendation.
    warmup_steps = min(int(hp.warmup_epochs * steps_per_epoch), total_steps // 2)
    min_ratio = hp.min_lr / hp.lr if hp.lr > 0 else 0.0

    if hp.scheduler == "plateau":
        # Watches the validation metric; stepped per epoch, not per step
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=metric_mode(hp.monitor_metric),
            factor=hp.step_gamma, patience=max(2, hp.patience // 3),
            min_lr=hp.min_lr,
        ), "epoch"

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        if hp.scheduler == "cosine":
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        if hp.scheduler == "poly":
            return min_ratio + (1 - min_ratio) * (1 - progress) ** 0.9
        if hp.scheduler == "step":
            k = (step // max(1, hp.step_size * steps_per_epoch))
            return max(min_ratio, hp.step_gamma ** k)
        if hp.scheduler == "onecycle":
            # A simplified one-cycle: warmup is already handled above, the rest is cosine
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "step"


class EMA:
    """An exponential moving average of the weights — usually better at validation."""

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model: nn.Module) -> dict:
        """Write the EMA weights into the model and return the previous weights."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k in self.shadow}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))
        return backup

    def restore(self, model: nn.Module, backup: dict) -> None:
        sd = model.state_dict()
        for k, v in backup.items():
            sd[k].copy_(v)


# ─────────────────────────────────────────────────────────────────────────────
# The base trainer
# ─────────────────────────────────────────────────────────────────────────────


class BaseTrainer(ABC):
    task: Task

    def __init__(self, cfg: RunConfig, writer: EventWriter):
        self.cfg = cfg
        self.hp = cfg.hp
        self.ds = cfg.dataset
        self.w = writer
        self.device = resolve_device(cfg.device)
        self.model: nn.Module | None = None
        self.ema: EMA | None = None
        self.loaders: dict[str, DataLoader] = {}
        self.datasets: dict[str, Any] = {}
        self.criterion: nn.Module | None = None
        self.optimizer = None
        self.scheduler = None
        self.sched_interval = "step"
        self.scaler = None
        self.epoch = 0
        self.global_step = 0
        self.history: list[dict] = []
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.mode = metric_mode(self.hp.monitor_metric)
        self._last_system_event = 0.0
        self._last_batch_event = 0.0
        self._epochs_without_improvement = 0
        self.resume = False             # set by runner.py --resume
        self.start_epoch = 1
        self._elapsed_before = 0.0      # training time of the earlier sessions
        self._session_started = time.time()

    # ── the parts subclasses fill in ─────────────────────────────────────
    @abstractmethod
    def build_model(self) -> nn.Module: ...

    @abstractmethod
    def build_data(self) -> None:
        """Populates the `self.datasets` and `self.loaders` dictionaries."""

    @abstractmethod
    def new_metrics(self):
        """Returns a fresh metric accumulator."""

    def forward_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Default: (image, target) → (logits, target)."""
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        if self.hp.channels_last and x.ndim == 4:
            x = x.to(memory_format=torch.channels_last)
        return self.model(x), y

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, target)

    def save_preview(self, epoch: int, metrics) -> str | None:
        """Produce a preview image from validation samples (optional)."""
        return None

    @staticmethod
    def loader_source(split: str, dataset) -> dict:
        """The DataLoader arguments that choose the samples. The training split goes
        through a seeded sampler, so its order and augmentation follow `hp.seed` and
        survive a resume (data/seeding.py); the other splits are read in order."""
        if split != "train":
            return {"dataset": dataset, "shuffle": False}
        return {"dataset": SeededDataset(dataset), "sampler": SeededSampler(len(dataset))}

    def n_classes(self) -> int:
        return max(2, self.ds.num_classes)

    # ── setup ────────────────────────────────────────────────────────────
    def setup(self) -> None:
        set_seed(self.hp.seed, self.hp.deterministic)
        self.w.log(f"Device: {self.device} ({detect().label})")

        self.build_data()
        n_train = len(self.datasets.get("train", []))
        n_val = len(self.datasets.get("val", []))
        if n_train == 0:
            raise RuntimeError("The training split contains no samples.")
        self.w.log(f"Data: {n_train} training / {n_val} validation samples")

        self.model = self.build_model().to(self.device)
        if self.hp.channels_last and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        n_params = sum(p.numel() for p in self.model.parameters())
        n_train_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.w.log(f"Model: {n_params/1e6:.1f}M parameters ({n_train_params/1e6:.1f}M trainable)")

        if self.hp.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                self.w.log("torch.compile enabled")
            except Exception as exc:
                self.w.log(f"torch.compile failed, continuing without it: {exc}", "warning")

        self.criterion = self.build_criterion()
        self.optimizer = build_optimizer(self.model, self.hp)
        steps = max(1, math.ceil(len(self.loaders["train"]) / self.hp.accumulate_grad_batches))
        self.scheduler, self.sched_interval = build_scheduler(self.optimizer, self.hp, steps)

        if self.hp.amp and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
        if self.hp.ema:
            self.ema = EMA(self.model, self.hp.ema_decay)
            self.w.log(f"EMA enabled (decay={self.hp.ema_decay})")

        if self.resume:
            self._restore_resume()

        # The optimizer above already holds every parameter, so freezing after it
        # (or not freezing a resumed run past the frozen epochs) is safe.
        if self.hp.freeze_backbone_epochs >= self.start_epoch:
            self.set_backbone_frozen(True)
            self.w.log(f"Backbone frozen for the first {self.hp.freeze_backbone_epochs} epochs")

    def build_criterion(self) -> nn.Module:
        from trainers.losses import build_loss

        weight = self.class_weight_tensor()
        return build_loss(self.task, self.hp, self.n_classes(),
                          self.ds.ignore_index, weight)

    def class_weight_tensor(self) -> torch.Tensor | None:
        if self.hp.class_weights != "balanced":
            return None
        from data.datasets_2d import class_weights

        train = self.datasets.get("train")
        if self.task == Task.CLASSIFICATION and hasattr(train, "targets"):
            w = class_weights(train, self.n_classes())
            self.w.log(f"Balanced class weights: {[round(x,3) for x in w.tolist()]}")
            return w.to(self.device)
        return None

    def backbone_modules(self) -> list[nn.Module]:
        """The backbone to freeze — subclasses customise this when needed."""
        return []

    def set_backbone_frozen(self, frozen: bool) -> None:
        for mod in self.backbone_modules():
            for p in mod.parameters():
                p.requires_grad = not frozen

    # ── the main loop ────────────────────────────────────────────────────
    def fit(self) -> RunStatus:
        status = RunStatus.COMPLETED
        error: str | None = None
        started = self._session_started = time.time()

        try:
            self.setup()
            self.w.emit(E.RUN_START, run_name=self.cfg.run_name,
                        resumed_from=self.start_epoch - 1 if self.resume else None,
                        total_epochs=self.hp.epochs,
                        model=self.cfg.model.display_name or self.cfg.model.arch,
                        task=self.ds.task.value,
                        monitor_metric=self.hp.monitor_metric,
                        monitor_mode=self.mode,
                        n_train=len(self.datasets.get("train", [])),
                        n_val=len(self.datasets.get("val", [])),
                        device=str(self.device))
            self.w.update_state(status=RunStatus.RUNNING.value, pid=__import__("os").getpid(),
                                run_name=self.cfg.run_name, total_epochs=self.hp.epochs,
                                monitor_metric=self.hp.monitor_metric,
                                started_at=started, elapsed_before=self._elapsed_before,
                                epoch=self.start_epoch - 1)

            done = self.start_epoch > self.hp.epochs or self._should_early_stop()
            for epoch in range(self.start_epoch, self.hp.epochs + 1):
                if done:
                    break
                self.epoch = epoch
                self._maybe_unfreeze(epoch)
                t0 = time.time()
                self.w.emit(E.EPOCH_START, epoch=epoch, total_epochs=self.hp.epochs)

                train_metrics = self.train_one_epoch()
                val_metrics = {}
                if self.loaders.get("val") and epoch % self.hp.val_interval == 0:
                    val_metrics = self.validate()

                improved = self._update_best(val_metrics or train_metrics)
                lr_now = self.optimizer.param_groups[0]["lr"]
                epoch_time = time.time() - t0

                row = {"epoch": epoch, "lr": lr_now, "epoch_time": epoch_time,
                       "best": improved,
                       **{f"train_{k}": v for k, v in train_metrics.items()},
                       **{f"val_{k}": v for k, v in val_metrics.items()}}
                self.history.append(row)

                self.w.emit(E.EPOCH_END, epoch=epoch, train=train_metrics, val=val_metrics,
                            lr=lr_now, epoch_time=epoch_time, best=improved)
                self.w.update_state(epoch=epoch, best_value=self.best_value,
                                    best_epoch=self.best_epoch)
                self._log_epoch(epoch, train_metrics, val_metrics, lr_now, epoch_time, improved)

                self.save_checkpoint("last.pt")
                if improved:
                    self.save_checkpoint("best.pt")

                if self.hp.preview_every_n_epochs and \
                        epoch % self.hp.preview_every_n_epochs == 0:
                    self._emit_preview(epoch)

                if self.sched_interval == "epoch" and self.scheduler is not None:
                    monitored = (val_metrics or train_metrics).get(self.hp.monitor_metric)
                    if monitored is not None:
                        self.scheduler.step(monitored)

                self.write_metrics_table()
                self.save_resume_state()

                if self._should_early_stop():
                    self.w.log(
                        f"Early stopping: `{self.hp.monitor_metric}` did not improve for "
                        f"{self.hp.patience} epochs (best: epoch {self.best_epoch})."
                    )
                    break
                self.check_stop()
                if self.w.pause_requested() and epoch < self.hp.epochs:
                    raise PauseRequested

        except PauseRequested:
            status = RunStatus.STOPPED
            self.w.log(f"Paused after epoch {self.epoch}; resume it from the Training page.")
        except StopRequested:
            status = RunStatus.STOPPED
            saved = self.w.state.get("resume_epoch")
            self.w.log("Stopped at the user's request; the latest state was saved as `last.pt`."
                       + (f" Resuming continues after epoch {saved}." if saved else ""))
            try:
                self.save_checkpoint("last.pt")
            except Exception:
                pass
        except KeyboardInterrupt:
            status = RunStatus.STOPPED
            self.w.log("Interrupted (KeyboardInterrupt).", "warning")
        except Exception as exc:
            status = RunStatus.FAILED
            error = f"{type(exc).__name__}: {exc}"
            self.w.log(error, "error")
            self.w.log(traceback.format_exc(), "error")

        finally:
            self.w.clear_stop()
            duration = self._elapsed_before + time.time() - started
            if status == RunStatus.COMPLETED:
                # Only unfinished runs are resumed; the state is no use any more
                self.cfg.path(Layout.RESUME).unlink(missing_ok=True)
                self.w.update_state(resume_epoch=None)
            try:
                self.finalize(status)
            except Exception as exc:
                self.w.log(f"Error while producing the results: {exc}", "warning")
            self.w.emit(E.RUN_END, status=status.value, error=error,
                        best_value=self.best_value, best_epoch=self.best_epoch,
                        duration_s=duration, epochs_completed=self.epoch)
            self.w.update_state(status=status.value, error=error,
                                best_value=self.best_value, best_epoch=self.best_epoch,
                                duration_s=duration)
        return status

    def _maybe_unfreeze(self, epoch: int) -> None:
        if self.hp.freeze_backbone_epochs and epoch == self.hp.freeze_backbone_epochs + 1:
            self.set_backbone_frozen(False)
            self.w.log("Backbone unfrozen; all layers are training now.")

    # ── the training step ────────────────────────────────────────────────
    def train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        loader = self.loaders["train"]
        total_steps = len(loader)
        accum = self.hp.accumulate_grad_batches
        metrics = self.new_metrics()
        running_loss, n_batches, seen = 0.0, 0, 0
        t_start = time.time()

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            self.check_stop()

            use_amp = self.hp.amp and self.device.type == "cuda"
            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                logits, target = self.forward_batch(batch)
                loss = self.compute_loss(logits, target)

            if not torch.isfinite(loss):
                self.w.log(f"Step {step}: the loss is not finite, skipping this batch.", "warning")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            scaled = loss / accum
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

            is_last = step == total_steps - 1
            if (step + 1) % accum == 0 or is_last:
                if self.hp.grad_clip:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hp.grad_clip)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)
                if self.sched_interval == "step" and self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1

            running_loss += loss.item()
            n_batches += 1
            seen += target.shape[0]

            with torch.no_grad():
                metrics.update(logits.detach(), target)

            self._emit_batch(step, total_steps, loss.item(), seen, t_start)

        out = {"loss": running_loss / max(1, n_batches)}
        out.update(metrics.compute(full=False))
        return out

    # ── validation ───────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self, split: str = "val", full: bool = True) -> dict[str, float]:
        loader = self.loaders.get(split)
        if loader is None:
            return {}
        self.model.eval()

        backup = None
        if self.ema is not None:
            backup = self.ema.copy_to(self.model)

        metrics = self.new_metrics()
        total, n = 0.0, 0
        try:
            for batch in loader:
                self.check_stop()
                logits, target = self.forward_batch(batch)
                loss = self.compute_loss(logits, target)
                if torch.isfinite(loss):
                    total += loss.item()
                    n += 1
                metrics.update(logits, target)
        finally:
            if backup is not None:
                self.ema.restore(self.model, backup)

        out = {"loss": total / max(1, n)}
        out.update(metrics.compute(full=full))
        self._last_metrics = metrics
        return out

    # ── best tracking / early stopping ───────────────────────────────────
    def _update_best(self, metrics: dict[str, float]) -> bool:
        key = self.hp.monitor_metric
        value = metrics.get(key)
        if value is None and key.startswith("val_"):
            value = metrics.get(key[4:])
        if value is None:
            value = metrics.get("loss")
            key = "loss"
        if value is None:
            return False

        better = (self.best_value is None or
                  (value > self.best_value if metric_mode(key) == "max"
                   else value < self.best_value))
        if better:
            self.best_value, self.best_epoch = float(value), self.epoch
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1
        return better

    def _should_early_stop(self) -> bool:
        return (self.hp.early_stopping
                and self._epochs_without_improvement >= self.hp.patience)

    # ── the stop signal ──────────────────────────────────────────────────
    def check_stop(self) -> None:
        if self.w.stop_requested():
            raise StopRequested

    # ── event emission ───────────────────────────────────────────────────
    def _emit_batch(self, step: int, total: int, loss: float, seen: int, t_start: float) -> None:
        now = time.time()
        is_edge = step == 0 or step == total - 1
        if not is_edge:
            if (step + 1) % self.hp.log_every_n_steps != 0:
                return
            if now - self._last_batch_event < BATCH_EVENT_MIN_INTERVAL_S:
                return
        self._last_batch_event = now
        elapsed = max(1e-6, now - t_start)
        self.w.emit(E.BATCH, epoch=self.epoch, step=step + 1, total_steps=total,
                    loss=loss, lr=self.optimizer.param_groups[0]["lr"],
                    ips=seen / elapsed)

        if now - self._last_system_event >= SYSTEM_EVENT_INTERVAL_S:
            self._last_system_event = now
            self.w.emit(E.SYSTEM, **telemetry())

    def _log_epoch(self, epoch, train, val, lr, dt, improved) -> None:
        def fmt(d: dict) -> str:
            return " ".join(f"{k}={v:.4f}" for k, v in d.items()
                            if isinstance(v, (int, float)))

        star = " ★" if improved else ""
        self.w.log(f"epoch {epoch}/{self.hp.epochs} · {dt:.1f}s · lr={lr:.2e} · "
                   f"train[{fmt(train)}] val[{fmt(val)}]{star}")

    def _emit_preview(self, epoch: int) -> None:
        try:
            rel = self.save_preview(epoch, getattr(self, "_last_metrics", None))
            if rel:
                self.w.emit(E.ARTIFACT, kind="preview", path=rel, epoch=epoch)
        except Exception as exc:
            self.w.log(f"Could not produce a preview: {exc}", "warning")

    # ── persistence ──────────────────────────────────────────────────────
    def save_checkpoint(self, name: str) -> Path:
        path = self.cfg.path(Layout.CHECKPOINTS, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = self.model
        state = {
            "model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
            "epoch": self.epoch,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "monitor_metric": self.hp.monitor_metric,
            "config": self.cfg.model_dump(mode="json"),
            "classes": self.ds.classes,
        }
        if self.ema is not None:
            state["ema"] = self.ema.shadow
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)
        return path

    def save_resume_state(self) -> None:
        """Everything needed to continue after this epoch exactly as if the run had
        not been interrupted — written at every epoch boundary, apart from last.pt,
        which a stop in the middle of an epoch overwrites."""
        model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        state = {
            "format": 1,
            "epoch": self.epoch,
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "ema": self.ema.shadow if self.ema is not None else None,
            "global_step": self.global_step,
            "history": self.history,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self._epochs_without_improvement,
            "elapsed_s": self._elapsed_before + time.time() - self._session_started,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "arch": self.cfg.model.arch,
            "classes": self.ds.classes,
        }
        path = self.cfg.path(Layout.RESUME)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)
        self.w.update_state(resume_epoch=self.epoch)

    def _restore_resume(self) -> None:
        path = self.cfg.path(Layout.RESUME)
        if not path.is_file():
            raise RuntimeError("There is no resume checkpoint (checkpoints/resume.pt) to continue from.")
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("arch") != self.cfg.model.arch or state.get("classes") != self.ds.classes:
            raise RuntimeError("The resume checkpoint belongs to a different model or class list "
                               "than config.json; it cannot be continued.")
        model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler is not None and state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])
        if self.ema is not None and state.get("ema") is not None:
            self.ema.shadow = {k: v.to(self.device).float() for k, v in state["ema"].items()}
        self.epoch = state["epoch"]
        self.start_epoch = self.epoch + 1
        self.global_step = state.get("global_step", 0)
        self.history = list(state.get("history") or [])
        self.best_value, self.best_epoch = state.get("best_value"), state.get("best_epoch")
        self._epochs_without_improvement = state.get("epochs_without_improvement", 0)
        self._elapsed_before = float(state.get("elapsed_s") or 0.0)
        rng = state.get("rng") or {}
        try:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        except Exception as exc:
            self.w.log(f"Could not restore the random state ({exc}); the data order will differ "
                       "from an uninterrupted run.", "warning")
        best = (f" · best {self.hp.monitor_metric} {self.best_value:.4f} at epoch {self.best_epoch}"
                if self.best_value is not None else "")
        self.w.log(f"Resumed after epoch {self.epoch} of {self.hp.epochs}{best}")

    def write_metrics_table(self) -> None:
        """Rewrite metrics.csv at the end of every epoch — so a complete metric
        history survives even if training is cut short."""
        if not self.history:
            return
        import pandas as pd

        df = pd.DataFrame(self.history)
        path = self.cfg.path(Layout.METRICS_CSV)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def finalize(self, status: RunStatus) -> None:
        """Subclasses extend this to produce reports/plots/exports."""
        self.write_metrics_table()
