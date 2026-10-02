"""The Ultralytics (YOLO26 / YOLO11 / YOLOv8) adapter.

Ultralytics brings its own training loop, loss function and augmentation; rather
than forcing it into our loop, we hook into its callbacks and emit **the same
event stream**. That keeps the Training page from having to know which library
is doing the training.

The dataset conversion is handled by data/convert_yolo.py: the user still provides
a single canonical format.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from core.capabilities import log_applied, model_kwargs
from core.events import E, EventWriter
from core.hardware import detect, telemetry
from core.schemas import Layout, RunConfig, RunStatus, Task, metric_mode
from data import convert_yolo

# Ultralytics metric names → our vocabulary
METRIC_MAP = {
    "metrics/accuracy_top1": "accuracy",
    "metrics/accuracy_top5": "top5_accuracy",
    "metrics/precision(B)": "precision_box",
    "metrics/recall(B)": "recall_box",
    "metrics/mAP50(B)": "map50",
    "metrics/mAP50-95(B)": "map50_95",
    "metrics/precision(M)": "precision_macro",
    "metrics/recall(M)": "recall_macro",
    "metrics/mAP50(M)": "map50_mask",
    "metrics/mAP50-95(M)": "map50_95_mask",
    "metrics/mIoU": "iou_macro",
    "metrics/mDice": "dice_macro",
    "metrics/pixel_accuracy": "pixel_accuracy",
    "fitness": "fitness",
}


def _weights_path(weights: str) -> Path:
    """Where a bare release name such as `yolo26n-cls.pt` is downloaded to.

    Ultralytics downloads a bare name into the working directory — the project
    root, or the image's /app in Docker, where it is lost on every rebuild. A full
    path is downloaded to that path, so point it at the shared weight cache
    instead (XDG_CACHE_HOME is the mounted /cache volume in the container).
    """
    p = Path(weights)
    if p.parent != Path(".") or p.exists():
        return p            # a user-supplied path, or a file already in place
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    target = cache / "trainstudio" / "ultralytics" / p.name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


class UltralyticsTrainer:
    """Offers the same `fit()` contract as BaseTrainer, but owns its loop."""

    def __init__(self, cfg: RunConfig, writer: EventWriter):
        self.cfg = cfg
        self.hp = cfg.hp
        self.ds = cfg.dataset
        self.w = writer
        self.epoch = 0
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.history: list[dict] = []
        self.model = None
        self._yolo_trainer = None
        self._epoch_started = 0.0
        self._last_system = 0.0
        self._stopped = False
        self._monitor = cfg.hp.monitor_metric
        self._mode = metric_mode(self._monitor)

    # ── helpers ──────────────────────────────────────────────────────────
    def _device_arg(self) -> str | int:
        """Ultralytics wants the device in its own format: 0 / "mps" / "cpu"."""
        dev = self.cfg.device
        if dev == "auto":
            dev = detect().torch_device
        if dev.startswith("cuda"):
            return int(dev.split(":")[1]) if ":" in dev else 0
        return dev

    def _translate(self, raw: dict) -> dict[str, float]:
        """Translate the Ultralytics metric dictionary into our naming.

        Loss components (`val/box_loss`, `val/seg_loss`, …) are carried through
        individually and their sum is added as `loss` — so it means the same thing
        as the `loss` on the training side.
        """
        out: dict[str, float] = {}
        loss_parts: dict[str, float] = {}
        for key, value in (raw or {}).items():
            if not isinstance(value, (int, float)):
                continue
            name = METRIC_MAP.get(key)
            if name is None and key.startswith("val/"):
                short = key.split("/", 1)[1]
                if short.endswith("loss"):
                    loss_parts[short] = float(value)
                    continue
                name = short
            if name:
                out[name] = float(value)
        if loss_parts:
            out.update(loss_parts)
            out["loss"] = float(sum(loss_parts.values()))
        return out

    # ── callbacks ────────────────────────────────────────────────────────
    def _register(self, model) -> None:
        model.add_callback("on_train_start", self._on_train_start)
        model.add_callback("on_train_epoch_start", self._on_epoch_start)
        model.add_callback("on_train_batch_end", self._on_batch_end)
        model.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
        model.add_callback("on_train_end", self._on_train_end)

    def _on_train_start(self, trainer) -> None:
        self._yolo_trainer = trainer
        self.w.log(f"Ultralytics training started · output directory {trainer.save_dir}")

    def _on_epoch_start(self, trainer) -> None:
        self.epoch = int(getattr(trainer, "epoch", 0)) + 1
        self._epoch_started = time.time()
        self.w.emit(E.EPOCH_START, epoch=self.epoch, total_epochs=self.hp.epochs)

    def _on_batch_end(self, trainer) -> None:
        # We catch the stop signal here: Ultralytics reads its `stop` flag at the
        # end of an epoch, so it exits with at most one epoch of delay.
        if self.w.stop_requested() and not trainer.stop:
            trainer.stop = True
            self._stopped = True
            self.w.log("A stop was requested — Ultralytics will halt at the end of this epoch.")

        step = int(getattr(trainer, "_ts_step", 0)) + 1
        trainer._ts_step = step
        every = max(1, self.hp.log_every_n_steps)
        total = len(getattr(trainer, "train_loader", []) or [])
        if step % every and step != total:
            return

        loss = None
        tloss = getattr(trainer, "tloss", None)
        if tloss is not None:
            try:
                loss = float(tloss.sum() if hasattr(tloss, "sum") else tloss)
            except Exception:
                loss = None
        lrs = getattr(trainer, "lr", {}) or {}
        lr = float(next(iter(lrs.values()), 0.0)) if lrs else 0.0

        self.w.emit(E.BATCH, epoch=self.epoch, step=step, total_steps=total or step,
                    loss=loss, lr=lr)

        now = time.time()
        if now - self._last_system >= 5.0:
            self._last_system = now
            self.w.emit(E.SYSTEM, **telemetry())

    def _on_fit_epoch_end(self, trainer) -> None:
        trainer._ts_step = 0
        # When training finishes, Ultralytics validates once more with the best
        # weights and this callback fires a second time with the same epoch number.
        if any(row["epoch"] == self.epoch for row in self.history):
            return
        metrics = self._translate(getattr(trainer, "metrics", {}) or {})
        label = getattr(trainer, "label_loss_items", None)
        train_metrics: dict[str, float] = {}
        if callable(label) and getattr(trainer, "tloss", None) is not None:
            try:
                items = label(trainer.tloss, prefix="train")
                train_metrics = {k.split("/", 1)[-1]: float(v)
                                 for k, v in items.items() if isinstance(v, (int, float))}
                train_metrics["loss"] = float(sum(train_metrics.values()))
            except Exception:
                pass

        lrs = getattr(trainer, "lr", {}) or {}
        lr = float(next(iter(lrs.values()), 0.0)) if lrs else 0.0
        improved = self._update_best(metrics)
        dt = time.time() - self._epoch_started

        self.history.append({
            "epoch": self.epoch, "lr": lr, "epoch_time": dt, "best": improved,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in metrics.items()},
        })
        self.w.emit(E.EPOCH_END, epoch=self.epoch, train=train_metrics, val=metrics,
                    lr=lr, epoch_time=dt, best=improved)
        self.w.update_state(epoch=self.epoch, best_value=self.best_value,
                            best_epoch=self.best_epoch)

        fmt = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.w.log(f"epoch {self.epoch}/{self.hp.epochs} · {dt:.1f}s · lr={lr:.2e} · "
                   f"val[{fmt}]" + (" ★" if improved else ""))
        self._write_metrics()
        self._copy_artifacts(trainer)

    def _on_train_end(self, trainer) -> None:
        self._copy_artifacts(trainer, final=True)

    def _resolve_monitor(self, metrics: dict[str, float]) -> None:
        """Pin the monitored metric to one this backend actually produces.

        The recommendation engine picks a metric such as `dice_macro` based on the
        task, but Ultralytics reports mAP for instance segmentation. Once we see
        the real metric list in the first epoch, we correct the monitored metric
        and write it to state.json, so the label in the UI matches the value being
        tracked.
        """
        if self._monitor in metrics:
            return
        for candidate in ("map50_95_mask", "map50_95", "accuracy",
                          "dice_macro", "iou_macro", "fitness"):
            if candidate in metrics:
                self.w.log(
                    f"`{self._monitor}` is not produced by this backend; "
                    f"the monitored metric was changed to `{candidate}`."
                )
                self._monitor = candidate
                self._mode = metric_mode(candidate)
                self.w.update_state(monitor_metric=candidate)
                return

    def _update_best(self, metrics: dict[str, float]) -> bool:
        self._resolve_monitor(metrics)
        value = metrics.get(self._monitor)
        if value is None:
            return False
        mode = self._mode

        better = (self.best_value is None or
                  (value > self.best_value if mode == "max" else value < self.best_value))
        if better:
            self.best_value, self.best_epoch = float(value), self.epoch
        return better

    # ── moving the outputs ───────────────────────────────────────────────
    def _copy_artifacts(self, trainer, final: bool = False) -> None:
        """Ultralytics writes to its own `save_dir`; we move the results into our layout."""
        try:
            save_dir = Path(getattr(trainer, "save_dir", "")) if trainer else None
            if not save_dir or not save_dir.is_dir():
                return
            for name, target in (("best.pt", Layout.BEST), ("last.pt", Layout.LAST)):
                src = save_dir / "weights" / name
                if src.is_file():
                    dst = self.cfg.path(target)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            if final:
                plots = self.cfg.path(Layout.PLOTS)
                plots.mkdir(parents=True, exist_ok=True)
                for png in save_dir.glob("*.png"):
                    shutil.copy2(png, plots / png.name)
                for jpg in list(save_dir.glob("val_batch*.jpg"))[:3]:
                    prev = self.cfg.path(Layout.PREVIEWS)
                    prev.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(jpg, prev / jpg.name)
                    self.w.emit(E.ARTIFACT, kind="preview",
                                path=f"{Layout.PREVIEWS}/{jpg.name}", epoch=self.epoch)
        except Exception as exc:
            self.w.log(f"Could not move the Ultralytics outputs: {exc}", "warning")

    def _write_metrics(self) -> None:
        if not self.history:
            return
        import pandas as pd

        path = self.cfg.path(Layout.METRICS_CSV)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.history).to_csv(path, index=False)

    # ── main entry point ─────────────────────────────────────────────────
    def fit(self) -> RunStatus:
        import os

        from ultralytics import YOLO

        status = RunStatus.COMPLETED
        error: str | None = None
        started = time.time()

        try:
            self.w.update_state(status=RunStatus.RUNNING.value, pid=os.getpid(),
                                run_name=self.cfg.run_name, total_epochs=self.hp.epochs,
                                monitor_metric=self._monitor, started_at=started, epoch=0)

            data_arg = convert_yolo.prepare(self.ds, self.cfg.model.arch, self.w.log)
            if "-seg" in self.cfg.model.arch.lower() and self.ds.task == Task.SEGMENTATION:
                self.w.log(
                    "Instance segmentation was selected: the masks were converted to polygons. "
                    "Regions with holes are represented by their outer contour — if you need "
                    "exact pixel-level masks, prefer the `-sem` (semantic) variant or an smp "
                    "architecture such as U-Net."
                )

            weights = self.cfg.model.arch
            yaml_name = weights.replace(".pt", ".yaml")
            if not self.hp.pretrained:
                # A .pt always carries its pretrained weights, whatever train() is told
                self.w.log(f"Training from scratch: building `{yaml_name}`.")
                self.model = YOLO(yaml_name)
            else:
                try:
                    self.model = YOLO(str(_weights_path(weights)))
                except Exception as exc:
                    # If the pretrained weights cannot be downloaded, build from the architecture definition
                    self.w.log(f"Could not load `{weights}` ({exc}); starting from scratch "
                               f"with `{yaml_name}`.", "warning")
                    self.model = YOLO(yaml_name)

            self._register(self.model)
            self.w.emit(E.RUN_START, run_name=self.cfg.run_name,
                        total_epochs=self.hp.epochs,
                        model=self.cfg.model.display_name or self.cfg.model.arch,
                        task=self.ds.task.value, monitor_metric=self._monitor,
                        monitor_mode=self._mode, device=str(self._device_arg()))

            args = dict(
                data=str(data_arg),
                epochs=self.hp.epochs,
                imgsz=self.hp.img_size,
                batch=self.hp.batch_size,
                device=self._device_arg(),
                workers=self.hp.num_workers,
                seed=self.hp.seed,
                deterministic=self.hp.deterministic,
                optimizer={"adamw": "AdamW", "adam": "Adam",
                           "sgd": "SGD", "rmsprop": "RMSProp"}.get(self.hp.optimizer, "auto"),
                lr0=self.hp.lr,
                lrf=max(1e-4, self.hp.min_lr / self.hp.lr) if self.hp.lr else 0.01,
                weight_decay=self.hp.weight_decay,
                momentum=self.hp.momentum,
                warmup_epochs=float(self.hp.warmup_epochs),
                cos_lr=self.hp.scheduler == "cosine",
                patience=self.hp.patience if self.hp.early_stopping else 0,
                pretrained=self.hp.pretrained,
                amp=self.hp.amp,
                project=str(self.cfg.run_dir / "ultralytics"),
                name="train",
                exist_ok=True,
                plots=True,
                val=True,
                verbose=False,
                # Augmentation — map the user's settings onto their Ultralytics equivalents
                fliplr=self.cfg.aug.hflip,
                flipud=self.cfg.aug.vflip,
                degrees=float(self.cfg.aug.rotate_limit),
                translate=float(self.cfg.aug.shift_limit),
                scale=float(self.cfg.aug.scale_limit),
                mosaic=1.0 if self.cfg.aug.preset in ("medium", "heavy") else 0.0,
                mixup=float(self.cfg.aug.mixup),
                hsv_h=0.0 if self.ds.modality.is_medical else 0.015,
                hsv_s=0.0 if self.ds.modality.is_medical else 0.7,
                hsv_v=0.0 if self.ds.modality.is_medical else 0.4,
            )
            reg = model_kwargs(self.cfg)            # dropout, classification only
            args.update(reg)
            log_applied(self.cfg, self.w, reg)
            self.w.log(f"Ultralytics settings: epochs={args['epochs']} imgsz={args['imgsz']} "
                       f"batch={args['batch']} device={args['device']}")
            self.model.train(**args)

            if self._stopped or self.w.stop_requested():
                status = RunStatus.STOPPED

        except KeyboardInterrupt:
            status = RunStatus.STOPPED
        except Exception as exc:
            import traceback

            status = RunStatus.FAILED
            error = f"{type(exc).__name__}: {exc}"
            self.w.log(error, "error")
            self.w.log(traceback.format_exc(), "error")

        finally:
            self.w.clear_stop()
            duration = time.time() - started
            try:
                self._finalize(status)
            except Exception as exc:
                self.w.log(f"Error while producing the results: {exc}", "warning")
            self.w.emit(E.RUN_END, status=status.value, error=error,
                        best_value=self.best_value, best_epoch=self.best_epoch,
                        duration_s=duration, epochs_completed=self.epoch)
            self.w.update_state(status=status.value, error=error,
                                best_value=self.best_value, best_epoch=self.best_epoch,
                                duration_s=duration)
        return status

    def _finalize(self, status: RunStatus) -> None:
        import pandas as pd

        from metrics.report import plot_curves, write_report, write_summary, write_tables

        self._write_metrics()
        if not self.history:
            return

        df = pd.DataFrame(self.history)
        plots = self.cfg.path(Layout.PLOTS)
        plot_curves(df, plots / "loss_curve.png", ["train_loss", "val_loss"],
                    ["Training", "Validation"], "Loss", "Loss")
        metric_cols = [c for c in df.columns
                       if c.startswith("val_") and not c.endswith("loss")][:3]
        if metric_cols:
            plot_curves(df, plots / "metric_curve.png", metric_cols,
                        [c.replace("val_", "") for c in metric_cols],
                        "Validation metrics", "Value",
                        mark_best=(self.best_epoch, self.best_value)
                        if self.best_epoch else None)
        plot_curves(df, plots / "lr.png", ["lr"], ["Learning rate"],
                    "Learning rate schedule", "lr")

        write_tables(self)
        final = {k.replace("val_", ""): v for k, v in self.history[-1].items()
                 if k.startswith("val_") and isinstance(v, (int, float))}
        exports = self._export()
        summary = write_summary(self, status,
                                {"final_split": "val", "final_metrics": final,
                                 "exports": exports})
        write_report(self, summary, [], final, "val")

    def _export(self) -> list[str]:
        """Ultralytics uses its own exporter — it produces the ONNX file."""
        if not (self.cfg.export_onnx or self.cfg.export_torchscript):
            return []
        best = self.cfg.path(Layout.BEST)
        if not best.is_file():
            return []
        written: list[str] = []
        try:
            from ultralytics import YOLO

            model = YOLO(str(best))
            for want, fmt in ((self.cfg.export_onnx, "onnx"),
                              (self.cfg.export_torchscript, "torchscript")):
                if not want:
                    continue
                out = model.export(format=fmt, imgsz=self.hp.img_size)
                target = self.cfg.path(Layout.EXPORTS, Path(out).name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(out), target)
                written.append(str(target))
                self.w.log(f"{fmt.upper()} written: {target.name}")
        except Exception as exc:
            self.w.log(f"Export failed: {exc}", "warning")
        return written
