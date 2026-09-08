"""Filesystem helpers shared by the UI, the scanner and the command line.

Two jobs live here, both of them about *not trusting the filesystem*:

* **Cleaning a path a human supplied.** A typed path is a guess, not a command.
  People paste out of a terminal, a file manager or a chat message, so paths
  arrive wrapped in quotes, with a trailing newline, as a percent-encoded
  `file://` URI, or relative to wherever they happened to be.
* **Asking questions that cannot fail.** `pathlib` lets `EACCES` through, so an
  unreadable parent turns "is there a train/ here?" into a crash. Browsing
  crosses mount points, other users' home directories and dead network shares,
  and every one of those answers with an errno rather than `False`.

Deliberately free of Streamlit and of any project imports, so the scanner and
`scripts/check_dataset.py` can use it without pulling in the UI.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse


def is_dir(path: str | Path | None) -> bool:
    """A permission-safe `Path.is_dir()`."""
    if path is None:
        return False
    try:
        return Path(path).is_dir()
    except OSError:
        return False


def is_file(path: str | Path | None) -> bool:
    """A permission-safe `Path.is_file()`."""
    if path is None:
        return False
    try:
        return Path(path).is_file()
    except OSError:
        return False


def can_list(path: str | Path) -> bool:
    """Whether the folder can actually be traversed, not merely stat'ed.

    Tells an empty folder apart from one the process is not allowed to read —
    the two are indistinguishable from an empty listing, and only one of them is
    something the user can fix.
    """
    try:
        next(iter(Path(path).iterdir()), None)
        return True
    except OSError:
        return False


def listdir(path: str | Path) -> list[Path]:
    """The entries of a directory, sorted by name, or `[]` if it cannot be read."""
    try:
        return sorted(Path(path).iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return []


def subdirs(path: str | Path) -> list[Path]:
    """Visible subdirectories, sorted by name; never raises."""
    return [d for d in listdir(path) if not d.name.startswith(".") and is_dir(d)]


def normalize_input(raw: str, base: str | Path | None = None) -> Path | None:
    """Turn whatever the user pasted into a usable absolute path.

    Handles, in order: surrounding whitespace and newlines; the single or double
    quotes a shell adds around a path with spaces; `file://` URIs (what a Linux
    file manager puts on the clipboard when you drag a folder into a text field,
    percent-encoded); shell escapes (`My\\ Data`); `~`; `$HOME`-style variables;
    and finally relative paths, which are resolved against `base` — the folder
    currently being shown — because that is what the user means when they type a
    bare folder name.

    Returns `None` for input that is empty once cleaned. Symlinks are *not*
    resolved: on Linux the dataset drive is often reached through one, and
    silently rewriting `/media/...` into its real device path makes the
    application look like it went somewhere else.
    """
    text = (raw or "").strip().strip("\r\n").strip()
    if not text:
        return None

    for quote in ('"', "'"):
        if len(text) >= 2 and text.startswith(quote) and text.endswith(quote):
            text = text[1:-1].strip()

    if text.lower().startswith("file://"):
        text = unquote(urlparse(text).path) or text
    elif "%" in text and "/" in text:
        # A path copied out of a browser's address bar keeps its escapes.
        text = unquote(text)

    text = text.replace("\\ ", " ").strip()
    if not text:
        return None

    text = os.path.expandvars(os.path.expanduser(text))

    p = Path(text)
    if not p.is_absolute():
        p = Path(base or Path.cwd()) / p

    # `abspath` normalises `.` and `..` textually — unlike `resolve()`, it keeps
    # the symlinked path the user actually typed.
    return Path(os.path.abspath(p))


def deepest_existing(path: str | Path) -> Path | None:
    """The nearest ancestor that exists — the useful half of "not found"."""
    p = Path(path)
    return next((a for a in [p, *p.parents] if is_dir(a)), None)
