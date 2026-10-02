"""Discovering, summarising and comparing runs under an output root.

A run directory is any folder containing a `config.json`. The list view reads
only the small `state.json` + `config.json` files; the large `events.jsonl` is
parsed only when a run is actually opened.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.events import read_all_events, read_state
from core.schemas import Layout, RunConfig, RunStatus, metric_mode

MAX_SCAN_DEPTH = 2


@dataclass
class RunSummary:
    run_dir: str
    run_name: str
    status: RunStatus = RunStatus.QUEUED
    task: str = ""
    model: str = ""
    dataset: str = ""
    monitor_metric: str = ""
    best_value: float | None = None
    best_epoch: int | None = None
    epoch: int = 0
    total_epochs: int = 0
    started_at: float | None = None
    updated_at: float | None = None
    duration_s: float | None = None
    pid: int | None = None
    error: str | None = None
    resume_epoch: int | None = None     # the last epoch checkpoints/resume.pt holds
    config: RunConfig | None = field(default=None, repr=False)

    @property
    def path(self) -> Path:
        return Path(self.run_dir)

    @property
    def progress(self) -> float:
        return min(1.0, self.epoch / self.total_epochs) if self.total_epochs else 0.0

    @property
    def is_live(self) -> bool:
        return self.status == RunStatus.RUNNING and process_alive(self.pid)

    @property
    def age_s(self) -> float | None:
        return time.time() - self.updated_at if self.updated_at else None

    def has(self, rel: str) -> bool:
        return (self.path / rel).exists()


def process_alive(pid: int | None) -> bool:
    """Is the PID still running — state.json can stay 'running' after a crash."""
    if not pid:
        return False
    # A run launched from this UI is our child, and nothing else waits on it: once
    # it exits it stays a zombie, which pid_exists() still reports as alive — so a
    # crashed run would read 'running' forever. Reap it here if it has exited.
    try:
        if os.waitpid(int(pid), os.WNOHANG)[0] != 0:
            return False
    except (ChildProcessError, OSError, AttributeError):
        pass        # not our child (or no waitpid on this platform)
    try:
        import psutil
    except ImportError:
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ProcessLookupError, PermissionError):
            return False
    try:
        return psutil.Process(int(pid)).status() != psutil.STATUS_ZOMBIE
    except psutil.AccessDenied:
        return True         # it exists, it just isn't ours
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────


def is_run_dir(path: Path) -> bool:
    return (path / Layout.CONFIG).is_file()


def find_run_dirs(root: str | Path, max_depth: int = MAX_SCAN_DEPTH) -> list[Path]:
    """Find folders containing a config.json under `root` (bounded depth)."""
    root = Path(root).expanduser()
    try:
        if not root.is_dir():
            return []
    except OSError:
        # Unreadable output root (EACCES) — nothing to list, but never raise:
        # this runs on page load, where an exception blanks the whole view.
        return []
    found: list[Path] = []

    def walk(d: Path, depth: int) -> None:
        if is_run_dir(d):
            found.append(d)
            return                       # we do not expect nested runs
        if depth >= max_depth:
            return
        try:
            for child in sorted(d.iterdir()):
                if not child.name.startswith(".") and child.is_dir():
                    walk(child, depth + 1)
        except (PermissionError, OSError):
            return

    walk(root, 0)
    return found


def load_summary(run_dir: str | Path) -> RunSummary | None:
    run_dir = Path(run_dir)
    if not is_run_dir(run_dir):
        return None

    s = RunSummary(run_dir=str(run_dir), run_name=run_dir.name)
    try:
        cfg = RunConfig.load(run_dir / Layout.CONFIG)
        s.config = cfg
        s.run_name = cfg.run_name
        s.task = cfg.dataset.task.value
        s.model = cfg.model.display_name or cfg.model.arch
        s.dataset = Path(cfg.dataset.root).name
        s.monitor_metric = cfg.hp.monitor_metric
        s.total_epochs = cfg.hp.epochs
    except Exception as exc:
        s.error = f"Could not read config.json: {exc}"

    st = read_state(run_dir)
    if st:
        try:
            s.status = RunStatus(st.get("status", "queued"))
        except ValueError:
            s.status = RunStatus.QUEUED
        s.epoch = int(st.get("epoch") or 0)
        s.total_epochs = int(st.get("total_epochs") or s.total_epochs)
        s.best_value = st.get("best_value")
        s.best_epoch = st.get("best_epoch")
        s.monitor_metric = st.get("monitor_metric") or s.monitor_metric
        s.started_at = st.get("started_at")
        s.updated_at = st.get("updated_at")
        s.pid = st.get("pid")
        s.error = st.get("error") or s.error
        s.resume_epoch = st.get("resume_epoch")
        if s.started_at and s.updated_at:
            # A resumed run counts the sessions before this one, not the pause between
            s.duration_s = (float(st.get("elapsed_before") or 0.0)
                            + max(0.0, s.updated_at - s.started_at))

    # A 'running' label is misleading once the process is gone
    if s.status == RunStatus.RUNNING and not process_alive(s.pid):
        s.status = RunStatus.FAILED
        s.error = s.error or "The process ended unexpectedly (state.json was left at 'running')."
    # So is 'queued' once the launched process is gone: it died before training
    # began, and only runner.out says why. A queued run without a PID was never
    # launched (or predates the launcher recording one) and is left alone.
    elif s.status == RunStatus.QUEUED and s.pid and not process_alive(s.pid):
        s.status = RunStatus.FAILED
        s.error = s.error or ("The process exited before training began — "
                              "see the process output (runner.out).")

    return s


def list_runs(root: str | Path, limit: int | None = None) -> list[RunSummary]:
    """Most recently updated first."""
    summaries = [s for d in find_run_dirs(root) if (s := load_summary(d)) is not None]
    summaries.sort(key=lambda s: s.updated_at or s.started_at or 0, reverse=True)
    return summaries[:limit] if limit else summaries


def list_runs_multi(roots: list[str | Path]) -> list[RunSummary]:
    seen: set[str] = set()
    out: list[RunSummary] = []
    for root in roots:
        for s in list_runs(root):
            if s.run_dir not in seen:
                seen.add(s.run_dir)
                out.append(s)
    out.sort(key=lambda s: s.updated_at or 0, reverse=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Name generation
# ─────────────────────────────────────────────────────────────────────────────


def suggest_run_name(model_name: str, dataset_name: str, output_dir: str | Path) -> str:
    """`unet-r50__seg_shapes__0907-2214` — readable and collision-free."""
    def slug(s: str) -> str:
        keep = "".join(c if c.isalnum() or c in "-_" else "-" for c in s.lower())
        while "--" in keep:
            keep = keep.replace("--", "-")
        return keep.strip("-")[:28] or "run"

    base = f"{slug(model_name)}__{slug(dataset_name)}__{time.strftime('%m%d-%H%M')}"
    root = Path(output_dir)
    name, i = base, 2
    while (root / name).exists():
        name = f"{base}-{i}"
        i += 1
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Epoch table / comparison
# ─────────────────────────────────────────────────────────────────────────────


def epoch_table(run_dir: str | Path):
    """A run's epoch metrics as a pandas DataFrame.

    A ready-made `metrics.csv` is tried first (cheap once training has finished);
    otherwise the table is rebuilt from the event stream (while training runs).
    """
    import pandas as pd

    run_dir = Path(run_dir)
    csv = run_dir / Layout.METRICS_CSV
    if csv.is_file():
        try:
            return pd.read_csv(csv)
        except Exception:
            pass

    from core.events import MetricAccumulator

    acc = MetricAccumulator()
    acc.feed(read_all_events(run_dir / Layout.EVENTS))
    return pd.DataFrame(acc.epochs)


def best_of(run_dir: str | Path, metric: str | None = None) -> tuple[float | None, int | None]:
    """A run's best metric value and the epoch it happened."""
    s = load_summary(run_dir)
    if s is None:
        return None, None
    metric = metric or s.monitor_metric
    if s.best_value is not None and metric == s.monitor_metric:
        return s.best_value, s.best_epoch

    df = epoch_table(run_dir)
    col = metric if metric in df.columns else f"val_{metric}"
    if col not in df.columns or df.empty:
        return None, None
    series = df[col].dropna()
    if series.empty:
        return None, None
    idx = series.idxmax() if metric_mode(metric) == "max" else series.idxmin()
    epoch = int(df.loc[idx, "epoch"]) if "epoch" in df.columns else None
    return float(series.loc[idx]), epoch


def comparison_frame(run_dirs: list[str | Path], metric: str):
    """Collect the same metric across several runs into one long-format table."""
    import pandas as pd

    frames = []
    for d in run_dirs:
        df = epoch_table(d)
        if df.empty or "epoch" not in df.columns:
            continue
        col = metric if metric in df.columns else f"val_{metric}"
        if col not in df.columns:
            continue
        sub = df[["epoch", col]].dropna().rename(columns={col: "value"})
        sub["run"] = Path(d).name
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["epoch", "value", "run"]
    )


def available_metrics(run_dirs: list[str | Path]) -> list[str]:
    """The union of metric columns across the selected runs."""
    cols: list[str] = []
    for d in run_dirs:
        df = epoch_table(d)
        for c in df.columns:
            if c not in ("epoch", "best", "epoch_time") and c not in cols:
                cols.append(c)
    priority = ["val_loss", "train_loss"]
    cols.sort(key=lambda c: (c not in priority, not c.startswith("val_"), c))
    return cols


def is_deletable(s: RunSummary) -> bool:
    """A run may be deleted once nothing is writing to it any more.

    'queued' with a live process is a run that is still starting up — deleting
    it would pull the folder from under a process about to train into it.
    """
    if s.status in (RunStatus.RUNNING, RunStatus.QUEUED) and process_alive(s.pid):
        return False
    return True


def resume_info(s: RunSummary) -> tuple[bool, str, int | None]:
    """Can this run be continued → (ok, the reason when not, the next epoch).

    Only an unfinished run is resumed, with its own config.json unchanged, from the
    end of the last epoch that `checkpoints/resume.pt` holds.
    """
    if s.status in (RunStatus.RUNNING, RunStatus.QUEUED) and process_alive(s.pid):
        return False, "The run is still in progress.", None
    if s.status == RunStatus.COMPLETED:
        return False, "Completed runs cannot be resumed.", None
    if s.status not in (RunStatus.STOPPED, RunStatus.FAILED):
        return False, "The run has not started yet.", None
    if not s.has(Layout.RESUME):
        if s.resume_epoch is None and s.epoch:
            return False, "This run was started before resuming was supported.", None
        return False, "It stopped before its first epoch ended, so there is nothing to resume.", None
    # Past the last epoch when the process died while writing the results: a
    # resume then only finishes those.
    return True, "", (s.resume_epoch or 0) + 1


def delete_run(run_dir: str | Path) -> tuple[bool, str]:
    """Permanently delete a run directory → (deleted, reason when it was not).

    Only ever deletes a real run directory, and never one that is still being
    written to — checked here rather than trusted to the caller.
    """
    p = Path(run_dir)
    s = load_summary(p)
    if s is None:
        return False, "not a run directory"
    if not is_deletable(s):
        return False, "still running — stop it first"
    shutil.rmtree(p, ignore_errors=True)
    return (False, "could not remove every file") if p.exists() else (True, "")


def delete_runs(run_dirs) -> tuple[list[str], dict[str, str]]:
    """Delete several runs → (deleted, {run_dir: reason} for those left in place)."""
    deleted, kept = [], {}
    for d in run_dirs:
        ok, why = delete_run(d)
        if ok:
            deleted.append(str(d))
        else:
            kept[str(d)] = why
    return deleted, kept


def dir_size_mb(path: str | Path) -> float:
    total = 0
    try:
        for p in Path(path).rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except OSError:
        pass
    return total / 1024**2
