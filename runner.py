#!/usr/bin/env python3
"""The entry point of the training process.

    python runner.py --config <run_dir>/config.json

The UI starts this script as a separate process, but the script does not depend
on the UI: the same command can be run on a server, in CI or under `nohup` and
produce byte-for-byte the same output. That follows naturally from the
configuration being the single source of truth.

Exit codes: 0 completed · 1 error · 2 stopped by the user
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# Make the project importable even when the script is invoked from another directory
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.events import E, EventWriter                      # noqa: E402
from core.schemas import Backend, RunConfig, RunStatus, Task  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_STOPPED = 0, 1, 2


def build_trainer(cfg: RunConfig, writer: EventWriter):
    """Select the right trainer for the (task, backend) pair."""
    task, backend = cfg.dataset.task, cfg.model.backend

    if backend == Backend.ULTRALYTICS:
        from trainers.ultralytics_adapter import UltralyticsTrainer

        return UltralyticsTrainer(cfg, writer)

    if task == Task.CLASSIFICATION:
        if backend == Backend.TIMM:
            from trainers.cls_timm import TimmClassificationTrainer

            return TimmClassificationTrainer(cfg, writer)
        if backend == Backend.TORCHVISION:
            from trainers.cls_torchvision import TorchvisionClassificationTrainer

            return TorchvisionClassificationTrainer(cfg, writer)

    elif task == Task.SEGMENTATION:
        if backend == Backend.SMP:
            from trainers.seg_smp import SmpSegmentationTrainer

            return SmpSegmentationTrainer(cfg, writer)
        if backend == Backend.TORCHVISION:
            from trainers.seg_torchvision import TorchvisionSegmentationTrainer

            return TorchvisionSegmentationTrainer(cfg, writer)
        if backend == Backend.HF:
            from trainers.seg_hf import HfSegmentationTrainer

            return HfSegmentationTrainer(cfg, writer)

    elif task == Task.SEGMENTATION3D:
        if backend == Backend.MONAI:
            from trainers.seg3d_monai import MonaiSegmentation3DTrainer

            return MonaiSegmentation3DTrainer(cfg, writer)

    raise ValueError(
        f"Unsupported combination: task={task.value}, library={backend.value}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The TrainStudio training process")
    ap.add_argument("--config", required=True, help="path to config.json in the run directory")
    args = ap.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"config.json not found: {config_path}", file=sys.stderr)
        return EXIT_FAILED

    try:
        cfg = RunConfig.load(config_path)
    except Exception as exc:
        print(f"Could not read config.json: {exc}", file=sys.stderr)
        traceback.print_exc()
        return EXIT_FAILED

    # Take the run directory from wherever config.json lives: outputs land in the
    # right place even if the user has moved the folder.
    run_dir = config_path.parent
    writer = EventWriter(run_dir)
    writer.clear_stop()

    status = RunStatus.FAILED
    try:
        writer.update_state(pid=os.getpid(), run_name=cfg.run_name)
        writer.log(f"TrainStudio · run `{cfg.run_name}` · PID {os.getpid()}")
        writer.log(f"Model {cfg.model.display_name or cfg.model.arch} "
                   f"({cfg.model.backend.value}) · task {cfg.dataset.task.value}")

        trainer = build_trainer(cfg, writer)
        status = trainer.fit()

    except Exception as exc:
        # Errors that prevent fit() from being entered at all (imports, an
        # unsupported combination, …)
        detail = f"{type(exc).__name__}: {exc}"
        writer.log(detail, "error")
        writer.log(traceback.format_exc(), "error")
        writer.emit(E.RUN_END, status=RunStatus.FAILED.value, error=detail)
        writer.update_state(status=RunStatus.FAILED.value, error=detail)
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        writer.close()

    return {
        RunStatus.COMPLETED: EXIT_OK,
        RunStatus.STOPPED: EXIT_STOPPED,
    }.get(status, EXIT_FAILED)


if __name__ == "__main__":
    sys.exit(main())
