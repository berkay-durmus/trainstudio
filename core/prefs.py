"""Small user preferences that persist between sessions.

`~/.trainstudio/prefs.json` — convenience data such as recently used folders and
the default output directory. The application must stay fully functional if this
file disappears, so every I/O error is swallowed silently.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Overridable so a container can point it at a mounted volume.
PREFS_DIR = Path(os.environ.get("TRAINSTUDIO_HOME", Path.home() / ".trainstudio"))
PREFS_FILE = PREFS_DIR / "prefs.json"
MAX_RECENT = 8


def _load() -> dict[str, Any]:
    try:
        return json.loads(PREFS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict[str, Any]) -> None:
    try:
        PREFS_DIR.mkdir(parents=True, exist_ok=True)
        PREFS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def get(key: str, default: Any = None) -> Any:
    return _load().get(key, default)


def set(key: str, value: Any) -> None:      # noqa: A001 — no module-level name clash
    data = _load()
    data[key] = value
    _save(data)


def recent(kind: str) -> list[str]:
    """`kind`: "dataset" | "output" — paths that no longer exist are filtered out."""
    paths = _load().get(f"recent_{kind}", [])
    return [p for p in paths if isinstance(p, str) and Path(p).is_dir()]


def push_recent(kind: str, path: str | Path) -> None:
    p = str(Path(path).expanduser().resolve())
    data = _load()
    key = f"recent_{kind}"
    items = [x for x in data.get(key, []) if x != p]
    data[key] = [p] + items[: MAX_RECENT - 1]
    _save(data)
