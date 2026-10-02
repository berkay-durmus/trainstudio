"""Starting, stopping and querying the training process.

Training **never** runs inside the Streamlit process. The reason is not only
performance: Streamlit re-runs the script on every interaction, so a long-running
loop inside the UI would either block it or be lost on the next re-run. A separate
process keeps training alive after the browser is closed and lets several tabs
watch the same run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from core.events import _atomic_write, read_state
from core.hardware import environment_snapshot
from core.runs import process_alive
from core.schemas import Layout, RunConfig, RunStatus

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER = PROJECT_ROOT / "runner.py"
STDOUT_FILE = "runner.out"
GRACEFUL_TIMEOUT_S = 30.0
OUTPUT_DIRS = (Layout.CHECKPOINTS, Layout.METRICS_DIR, Layout.PLOTS,
               Layout.PREVIEWS, Layout.PREDICTIONS, Layout.EXPORTS)


class LaunchError(RuntimeError):
    pass


@dataclass
class Launched:
    run_dir: Path
    pid: int
    command: list[str]


# ─────────────────────────────────────────────────────────────────────────────


def python_executable() -> str:
    """The interpreter that will run the training process — same environment as the UI."""
    return sys.executable or "python3"


def prepare_run_dir(cfg: RunConfig, overwrite: bool = False) -> Path:
    """Create the run directory and write config.json + env.json."""
    run_dir = cfg.run_dir
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise LaunchError(
            f"`{run_dir}` already exists and is not empty. Pick a different run name "
            "or confirm overwriting."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Leftovers from an earlier attempt must not pollute the new run: the logs are
    # appended to, and an old best.pt or report would pass for this run's output
    # if it failed early. Only the run layout is removed, nothing else in the folder.
    for stale in (Layout.STOP, Layout.EVENTS, Layout.STATE, Layout.LOG,
                  Layout.REPORT, STDOUT_FILE):
        (run_dir / stale).unlink(missing_ok=True)
    for sub in OUTPUT_DIRS:
        shutil.rmtree(run_dir / sub, ignore_errors=True)
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    cfg.save()
    (run_dir / Layout.ENV).write_text(
        json.dumps(environment_snapshot(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / Layout.STATE).write_text(
        json.dumps({
            "status": RunStatus.QUEUED.value,
            "run_name": cfg.run_name,
            "total_epochs": cfg.hp.epochs,
            "monitor_metric": cfg.hp.monitor_metric,
            "epoch": 0,
            "started_at": time.time(),
            "updated_at": time.time(),
        }, indent=2), encoding="utf-8"
    )
    return run_dir


def start_run(cfg: RunConfig, overwrite: bool = False,
              python_exe: str | None = None) -> Launched:
    """Prepare the run directory and start training in its own session."""
    if not RUNNER.is_file():
        raise LaunchError(f"runner.py not found: {RUNNER}")

    run_dir = prepare_run_dir(cfg, overwrite=overwrite)
    cmd = [python_exe or python_executable(), str(RUNNER),
           "--config", str(run_dir / Layout.CONFIG)]

    env = os.environ.copy()
    # Let the child process import the project
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["PYTHONUNBUFFERED"] = "1"
    # Keep BLAS threads from competing with the DataLoader workers
    env.setdefault("OMP_NUM_THREADS", "4")
    if cfg.device.startswith("mps") or cfg.device == "auto":
        # An unsupported MPS operator should fall back to CPU instead of failing
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    out_path = run_dir / STDOUT_FILE
    try:
        # The child holds its own copy of the descriptor; ours is closed on exit
        with out_path.open("ab") as out_fh:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=out_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                # Its own process session: a Ctrl-C aimed at Streamlit will not kill training
                start_new_session=True,
            )
    except Exception as exc:
        raise LaunchError(f"Could not start the process: {exc}") from exc

    # Record the PID straight away. A process that dies before the trainer writes
    # its first state (a bad config.json, an import error) would otherwise leave
    # the run 'queued' with nothing to tell that it is gone.
    _record_pid(run_dir, proc.pid)
    return Launched(run_dir=run_dir, pid=proc.pid, command=cmd)


def _record_pid(run_dir: Path, pid: int) -> None:
    st = read_state(run_dir)
    if st.get("status", RunStatus.QUEUED.value) != RunStatus.QUEUED.value or st.get("pid"):
        return      # the runner got there first
    st["pid"] = pid
    try:
        _atomic_write(run_dir / Layout.STATE, json.dumps(st, indent=2))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Stopping
# ─────────────────────────────────────────────────────────────────────────────


def request_stop(run_dir: str | Path) -> bool:
    """Graceful stop: a STOP file is written, the trainer notices it between
    batches, saves `last.pt` and shuts down cleanly."""
    p = Path(run_dir) / Layout.STOP
    try:
        p.write_text(str(time.time()), encoding="utf-8")
        return True
    except Exception:
        return False


def stop_requested(run_dir: str | Path) -> bool:
    return (Path(run_dir) / Layout.STOP).exists()


def clear_stop(run_dir: str | Path) -> None:
    (Path(run_dir) / Layout.STOP).unlink(missing_ok=True)


def force_kill(run_dir: str | Path) -> bool:
    """Terminate the process (and its children) when a graceful stop did not work."""
    st = read_state(run_dir)
    pid = st.get("pid")
    if not pid or not process_alive(pid):
        return False
    try:
        import psutil

        proc = psutil.Process(int(pid))
        children = proc.children(recursive=True)
        for c in children:
            c.terminate()
        proc.terminate()
        gone, alive = psutil.wait_procs([proc] + children, timeout=5)
        for p in alive:
            p.kill()
        return True
    except Exception:
        try:
            os.kill(int(pid), 15)
            return True
        except Exception:
            return False


def is_running(run_dir: str | Path) -> bool:
    st = read_state(run_dir)
    return st.get("status") == RunStatus.RUNNING.value and process_alive(st.get("pid"))


def runner_stdout(run_dir: str | Path, max_bytes: int = 20_000) -> str:
    """runner.out — crashes that happen before the trainer starts show up here.

    When training fails before entering the loop (an import error, say) events.jsonl
    stays empty; we read this file to be able to show the real cause.
    """
    p = Path(run_dir) / STDOUT_FILE
    if not p.is_file():
        return ""
    size = p.stat().st_size
    with p.open("rb") as fh:
        fh.seek(max(0, size - max_bytes))
        return fh.read().decode("utf-8", errors="replace")
