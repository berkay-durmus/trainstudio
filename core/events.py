"""The one-way event stream between the training process and the UI.

Trainer  → appends one JSON object per line to `events.jsonl` (append-only).
UI       → reads only the new lines, starting from the last byte offset it read.
Trainer  → atomically rewrites `state.json` at the end of every epoch; the run
           list reads that small file instead of parsing the large jsonl.

Append-only writes plus offset-based reads stay consistent even when the browser
is refreshed or several tabs watch the same run.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.schemas import Layout


# ─────────────────────────────────────────────────────────────────────────────
# Event types
# ─────────────────────────────────────────────────────────────────────────────


class E:
    RUN_START = "run_start"
    EPOCH_START = "epoch_start"
    BATCH = "batch"
    EPOCH_END = "epoch_end"
    ARTIFACT = "artifact"
    SYSTEM = "system"
    LOG = "log"
    RUN_END = "run_end"


def _atomic_write(path: Path, text: str) -> None:
    """Write to a temp file + rename — a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _jsonable(obj: Any) -> Any:
    """Reduce types such as numpy scalars / Path / Enum to plain JSON values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):      # numpy/torch scalar
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
            return None                                   # NaN/Inf → null
        return obj
    return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Writer — used inside the training process
# ─────────────────────────────────────────────────────────────────────────────


class EventWriter:
    """Appends events to events.jsonl, updates state.json, writes to train.log.

    Emitting an event must never bring training down: all I/O errors are swallowed.
    """

    def __init__(self, run_dir: str | Path, flush_every: int = 1):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / Layout.EVENTS
        self.state_path = self.run_dir / Layout.STATE
        self.log_path = self.run_dir / Layout.LOG
        self.stop_path = self.run_dir / Layout.STOP
        self._fh = self.events_path.open("a", encoding="utf-8", buffering=1)
        self._log_fh = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._n = 0
        self._flush_every = max(1, flush_every)
        self._state: dict[str, Any] = {}

    # ── core ─────────────────────────────────────────────────────────────
    def emit(self, type: str, **payload: Any) -> None:
        rec = {"t": round(time.time(), 3), "type": type}
        rec.update(_jsonable(payload))
        try:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._n += 1
            if self._n % self._flush_every == 0:
                self._fh.flush()
                os.fsync(self._fh.fileno())
        except Exception:
            pass

    def log(self, msg: str, level: str = "info") -> None:
        stamp = time.strftime("%H:%M:%S")
        try:
            self._log_fh.write(f"[{stamp}] {level.upper():5s} {msg}\n")
        except Exception:
            pass
        self.emit(E.LOG, level=level, msg=msg)

    # ── state.json ───────────────────────────────────────────────────────
    def update_state(self, **fields: Any) -> None:
        self._state.update(_jsonable(fields))
        self._state["updated_at"] = round(time.time(), 3)
        try:
            _atomic_write(
                self.state_path,
                json.dumps(self._state, indent=2, ensure_ascii=False),
            )
        except Exception:
            pass

    # ── stop signal ──────────────────────────────────────────────────────
    def stop_requested(self) -> bool:
        """The trainer calls this on every batch; the UI stops a run by creating
        this file."""
        return self.stop_path.exists()

    def clear_stop(self) -> None:
        try:
            self.stop_path.unlink(missing_ok=True)
        except Exception:
            pass

    def close(self) -> None:
        for fh in (self._fh, self._log_fh):
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Reader — used on the Streamlit side
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TailResult:
    events: list[dict] = field(default_factory=list)
    offset: int = 0
    truncated: bool = False      # file shrank (run restarted) → reset


def tail_events(path: str | Path, offset: int = 0, max_bytes: int = 8_000_000) -> TailResult:
    """Read whole lines starting at byte `offset` and return the new offset.

    A partially written final line is deliberately not consumed; the offset is
    left at the start of that line so the next call reads it once complete.
    """
    p = Path(path)
    if not p.exists():
        return TailResult(offset=0)

    size = p.stat().st_size
    if size < offset:                                  # file shrank → start over
        return TailResult(offset=0, truncated=True)
    if size == offset:
        return TailResult(offset=offset)

    read_from = offset
    truncated = False
    if size - offset > max_bytes:                      # too far behind → skip ahead
        read_from = size - max_bytes
        truncated = True

    with p.open("rb") as fh:
        fh.seek(read_from)
        blob = fh.read(size - read_from)

    # Everything up to the last newline is safe to parse
    cut = blob.rfind(b"\n")
    if cut == -1:
        return TailResult(offset=offset, truncated=truncated)
    consumed = blob[: cut + 1]
    new_offset = read_from + cut + 1

    events: list[dict] = []
    for line in consumed.split(b"\n"):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line.decode("utf-8")))
        except Exception:
            continue                                   # silently skip a corrupt line
    return TailResult(events=events, offset=new_offset, truncated=truncated)


def read_all_events(path: str | Path) -> list[dict]:
    return tail_events(path, 0, max_bytes=1 << 62).events


def read_state(run_dir: str | Path) -> dict:
    """Read state.json. Returns an empty dict if it cannot be parsed (the atomic
    rename makes this practically impossible, but we stay defensive)."""
    p = Path(run_dir) / Layout.STATE
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def tail_log(run_dir: str | Path, n_lines: int = 200, max_bytes: int = 400_000) -> str:
    """The last n lines of train.log."""
    p = Path(run_dir) / Layout.LOG
    if not p.exists():
        return ""
    size = p.stat().st_size
    with p.open("rb") as fh:
        fh.seek(max(0, size - max_bytes))
        blob = fh.read()
    lines = blob.decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


# ─────────────────────────────────────────────────────────────────────────────
# Accumulator that reduces the event stream to the tables the UI draws
# ─────────────────────────────────────────────────────────────────────────────


class MetricAccumulator:
    """Collects epoch metrics, batch progress and artifacts from the event stream.

    Kept in Streamlit's session_state and fed only the new events on each
    fragment tick (`feed`), so the jsonl is never re-parsed from scratch.
    """

    def __init__(self) -> None:
        self.offset: int = 0
        self.epochs: list[dict] = []          # epoch_end records (flattened)
        self.batches: list[dict] = []         # last N batches (progress + running loss)
        self.system: list[dict] = []
        self.artifacts: list[dict] = []
        self.run_start: dict | None = None
        self.run_end: dict | None = None
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.last_batch: dict | None = None
        self._max_batches = 2000
        self._max_system = 600

    # ── feeding ──────────────────────────────────────────────────────────
    def feed(self, events: list[dict]) -> None:
        for ev in events:
            t = ev.get("type")
            if t == E.EPOCH_END:
                self.epochs.append(self._flatten_epoch(ev))
            elif t == E.BATCH:
                self.last_batch = ev
                self.batches.append(ev)
                if len(self.batches) > self._max_batches:
                    del self.batches[: len(self.batches) - self._max_batches]
            elif t == E.EPOCH_START:
                self.current_epoch = ev.get("epoch", self.current_epoch)
                self.total_epochs = ev.get("total_epochs", self.total_epochs)
            elif t == E.SYSTEM:
                self.system.append(ev)
                if len(self.system) > self._max_system:
                    del self.system[: len(self.system) - self._max_system]
            elif t == E.ARTIFACT:
                self.artifacts.append(ev)
            elif t == E.RUN_START:
                self.run_start = ev
                self.total_epochs = ev.get("total_epochs", self.total_epochs)
            elif t == E.RUN_END:
                self.run_end = ev

    def reset(self) -> None:
        self.__init__()

    @staticmethod
    def _flatten_epoch(ev: dict) -> dict:
        """{"train":{"loss":..}, "val":{"dice":..}} → {"train_loss":.., "val_dice":..}"""
        row: dict[str, Any] = {
            "epoch": ev.get("epoch"),
            "epoch_time": ev.get("epoch_time"),
            "lr": ev.get("lr"),
            "best": bool(ev.get("best", False)),
        }
        for split in ("train", "val", "test"):
            for k, v in (ev.get(split) or {}).items():
                if isinstance(v, (int, float)) or v is None:
                    row[f"{split}_{k}"] = v
        return row

    # ── derived views ────────────────────────────────────────────────────
    def metric_keys(self) -> list[str]:
        keys: list[str] = []
        for row in self.epochs:
            for k in row:
                if k not in ("epoch", "best", "epoch_time") and k not in keys:
                    keys.append(k)
        return keys

    def series(self, key: str) -> tuple[list[int], list[float]]:
        xs, ys = [], []
        for row in self.epochs:
            v = row.get(key)
            if v is not None and row.get("epoch") is not None:
                xs.append(row["epoch"])
                ys.append(v)
        return xs, ys

    def best_row(self, metric: str, mode: str = "max") -> dict | None:
        rows = [r for r in self.epochs if r.get(metric) is not None]
        if not rows:
            return None
        return (max if mode == "max" else min)(rows, key=lambda r: r[metric])

    def eta_seconds(self) -> float | None:
        """Remaining-time estimate — from the median duration of the last 5 epochs."""
        times = [r["epoch_time"] for r in self.epochs if r.get("epoch_time")]
        if not times or not self.total_epochs:
            return None
        recent = sorted(times[-5:])
        median = recent[len(recent) // 2]
        remaining = self.total_epochs - (self.epochs[-1].get("epoch") or 0)
        return max(0.0, median * remaining) if remaining > 0 else 0.0

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.epochs)
