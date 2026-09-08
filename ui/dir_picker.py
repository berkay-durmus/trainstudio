"""A server-side folder picker.

Because the application runs on the user's own machine, the dataset is chosen
*in place* rather than uploaded: instead of copying thousands of images we point
at the path. Until the selection is confirmed (`Select this folder`) the calling
page keeps seeing the previous value.

Two rules shape the whole module:

* **A typed path is a guess, not a command.** People paste paths out of a
  terminal, a file manager or a chat message, so they arrive wrapped in quotes,
  with a trailing newline, as a `file://` URI, or relative to wherever they
  happened to be. Normalising all of that before touching the filesystem is the
  difference between "it just works" and "Folder not found".
* **Nothing here may raise.** Browsing crosses mount points, other users' home
  directories and dead network shares; every one of those answers with `EACCES`
  or `ENOTCONN` rather than `False`. Any of them escaping would replace the page
  with a traceback.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from core import prefs
from core.paths import can_list as _can_list
from core.paths import deepest_existing, is_dir as _is_dir, is_file as _is_file
from core.paths import listdir as _listdir, normalize_input

MAX_LISTED = 400
BREADCRUMB_MAX = 5          # deepest crumbs shown; root is always kept as well


def _state_key(key: str, name: str) -> str:
    return f"_dp::{key}::{name}"


def _default_root() -> Path:
    """Where the picker starts when nothing usable was requested. $HOME can be
    unreadable (an unset HOST_HOME leaves it at /root) or simply not mounted, so
    fall through to the first path that is actually browsable."""
    for cand in (os.environ.get("HOME"), os.environ.get("TRAINSTUDIO_RUNS_DIR"),
                 "/home", "/mnt", "/media", "/"):
        if cand and _is_dir(cand):
            return Path(cand)
    return Path("/")


def _mount_dirs() -> list[Path]:
    """Removable and secondary drives, discovered rather than assumed.

    On Linux a dataset almost always lives on a second disk, mounted under
    `/media/<user>/<label>` or `/mnt/<name>`; on macOS under `/Volumes`. Those
    are exactly the paths `$HOME`-based shortcuts cannot reach, so they are
    listed one level deep and offered directly.
    """
    user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    roots = [Path("/media") / user, Path("/media"), Path("/mnt"), Path("/Volumes"),
             Path("/srv"), Path("/data")]

    out: list[Path] = []
    for root in roots:
        if not _is_dir(root):
            continue
        # `/media/<user>` is listed before `/media`; once it has contributed,
        # walking its parent would only offer the same drives one level up.
        if any(root == p or root in p.parents for p in out):
            continue
        children = [d for d in _listdir(root)
                    if not d.name.startswith(".") and _is_dir(d)]
        # A mount root is only interesting for what is mounted *inside* it.
        out.extend(children[:8] or [root])
    return out


def _quick_links() -> list[tuple[str, Path]]:
    """Shortcut buttons. Convenience only — every one of them is just a jump to a
    directory, and the picker can still walk anywhere from there."""
    home = Path(os.environ.get("HOME") or Path.home())
    if not _is_dir(home):
        home = _default_root()
    candidates: list[tuple[str, Path]] = [
        ("🏠 Home", home),
        ("🖥️ Desktop", home / "Desktop"),
        ("📥 Downloads", home / "Downloads"),
    ]
    candidates += [(f"💾 {p.name}", p) for p in _mount_dirs()]

    # Pinned roots, e.g. TRAINSTUDIO_QUICK_DIRS=/mnt/nas/datasets:/srv/data
    for raw in (os.environ.get("TRAINSTUDIO_QUICK_DIRS") or "").split(":"):
        raw = raw.strip()
        if raw:
            candidates.insert(0, (f"📌 {Path(raw).name or raw}", Path(raw)))
    runs = os.environ.get("TRAINSTUDIO_RUNS_DIR")
    if runs:
        candidates.append(("📊 Runs", Path(runs)))

    seen: set[str] = set()
    out: list[tuple[str, Path]] = []
    for label, p in candidates:
        if _is_dir(p) and str(p) not in seen:
            seen.add(str(p))
            out.append((label, p))
    return out[:6]          # a single row of buttons; the tree does the rest


def _subdirs(path: Path) -> list[Path]:
    return [d for d in _listdir(path)
            if not d.name.startswith(".") and _is_dir(d)][:MAX_LISTED]


def _breadcrumbs(cwd: Path) -> list[Path]:
    """Root, an optional gap, then the deepest few ancestors.

    Slicing the chain naively drops the *root* end, which is the one crumb you
    always need on a long path like `/media/<user>/<disk>/…` — without it there
    is no way back up other than clicking ⬆️ ten times.
    """
    chain = list(reversed([cwd, *cwd.parents]))          # root … cwd
    if len(chain) <= BREADCRUMB_MAX + 1:
        return chain
    return [chain[0], *chain[-BREADCRUMB_MAX:]]


def directory_picker(
    key: str,
    label: str = "Folder",
    initial: str | Path | None = None,
    recent_kind: str | None = None,
    allow_create: bool = False,
    help_text: str = "",
) -> str | None:
    """Draw the folder picker and return the **confirmed** selection (None if there is none).

    With `allow_create=True` a path that does not exist yet is also accepted
    (needed when picking an output directory — the folder is created when
    training starts).
    """
    cwd_key = _state_key(key, "cwd")
    sel_key = _state_key(key, "selected")
    typed_key = _state_key(key, "typed")
    pending_key = _state_key(key, "pending")

    if cwd_key not in st.session_state:
        start = normalize_input(str(initial)) if initial else None
        start = start or _default_root()
        while not _is_dir(start) and start.parent != start:
            start = start.parent
        st.session_state[cwd_key] = str(start if _is_dir(start) else _default_root())

    # The text field is keyed, so Streamlit keeps showing whatever was last typed
    # and ignores any `value=`. Navigation therefore hands the new path over in
    # `pending`, which is written into the widget's state here — before the
    # widget is instantiated, the only point at which Streamlit allows it.
    if pending_key in st.session_state:
        st.session_state[typed_key] = st.session_state.pop(pending_key)

    cwd = Path(st.session_state[cwd_key])

    def navigate(target: str | Path) -> None:
        st.session_state[cwd_key] = str(target)
        st.session_state[pending_key] = str(target)
        st.rerun()

    def confirm(target: str | Path) -> None:
        st.session_state[sel_key] = str(target)
        if recent_kind:
            prefs.push_recent(recent_kind, target)
        st.rerun()

    st.markdown(f"**{label}**")
    if help_text:
        st.caption(help_text)

    # ── Direct path entry ────────────────────────────────────────────────
    # Enter in the text box submits: Streamlit reruns on change, and the value
    # differing from the folder on screen *is* the signal that the user typed
    # something new. Without this the path has to be re-confirmed with a click,
    # which reads as "typing the path does nothing".
    st.session_state.setdefault(typed_key, str(cwd))
    col_path, col_go = st.columns([6, 1])
    typed = col_path.text_input(
        "Path", key=typed_key,
        label_visibility="collapsed", placeholder="/media/<user>/<disk>/datasets/my_set",
    )
    target = normalize_input(typed, base=cwd)
    submitted = col_go.button("Go", key=_state_key(key, "go"), width="stretch")
    entered = target is not None and str(target) != str(cwd)

    if submitted or entered:
        if target is None:
            st.warning("Type a path first.")
        elif _is_dir(target):
            navigate(target)                    # unreadable is still a folder;
                                                # the listing below says so
        elif allow_create and _is_dir(target.parent):
            confirm(target)
        elif _is_file(target):
            st.error(f"`{target}` is a file, not a folder. Select the folder that "
                     "contains your dataset.")
        elif _is_dir(target.parent):
            st.error(f"There is no `{target.name}` folder in `{target.parent}`.")
        else:
            near = deepest_existing(target.parent)
            st.error(f"Folder not found: `{target}`"
                     + (f" — the deepest existing part is `{near}`." if near else ""))
            if submitted and near:
                navigate(near)

    # ── Quick access ─────────────────────────────────────────────────────
    quick = _quick_links()
    recents = prefs.recent(recent_kind) if recent_kind else []
    if quick or recents:
        cols = st.columns(max(1, len(quick) + (1 if recents else 0)))
        for col, (lbl, p) in zip(cols, quick):
            if col.button(lbl, key=_state_key(key, f"q{p}"), width="stretch"):
                navigate(p)
        if recents:
            with cols[-1]:
                choice = st.selectbox(
                    "Recently used", ["🕘 Recently used"] + recents,
                    key=_state_key(key, "recent"), label_visibility="collapsed",
                )
                if choice != "🕘 Recently used":
                    navigate(choice)

    # ── Breadcrumb ───────────────────────────────────────────────────────
    parts = _breadcrumbs(cwd)
    bc_cols = st.columns(len(parts) + 1)
    for col, p in zip(bc_cols, parts):
        name = p.name or "/"
        if col.button(name, key=_state_key(key, f"bc{p}"), width="stretch",
                      help=str(p)):
            navigate(p)
    if cwd.parent != cwd:
        if bc_cols[-1].button("⬆️ Up", key=_state_key(key, "up"), width="stretch"):
            navigate(cwd.parent)

    # ── Subfolders ───────────────────────────────────────────────────────
    subs = _subdirs(cwd)
    if subs:
        st.markdown("<div class='ts-scroll' style='max-height:260px'>", unsafe_allow_html=True)
        n_cols = 3
        for row_start in range(0, len(subs), n_cols):
            row = subs[row_start: row_start + n_cols]
            cols = st.columns(n_cols)
            for col, d in zip(cols, row):
                if col.button(f"📁 {d.name}", key=_state_key(key, f"d{d}"),
                              width="stretch"):
                    navigate(d)
        st.markdown("</div>", unsafe_allow_html=True)
        if len(subs) == MAX_LISTED:
            st.caption(f"Showing the first {MAX_LISTED} folders — you can type the path directly.")
    elif not _can_list(cwd):
        st.error(f"This folder cannot be read (permission denied). Grant access with, "
                 f"for example: `sudo chmod -R a+rX {cwd}`")
    else:
        st.caption("This folder has no subfolders.")

    # ── Confirmation ─────────────────────────────────────────────────────
    # Confirm what is on screen, and say which folder that is: the button used
    # to commit `cwd` silently, so a path that failed to resolve confirmed
    # whatever the picker happened to be showing instead.
    st.caption(f"Selecting: `{cwd}`")
    if st.button("✓ Select this folder", key=_state_key(key, "pick"),
                 type="primary", width="stretch"):
        confirm(cwd)

    return st.session_state.get(sel_key)


def clear_selection(key: str) -> None:
    st.session_state.pop(_state_key(key, "selected"), None)


def set_selection(key: str, path: str | Path) -> None:
    """Confirm a path from outside the picker.

    Pages call this after the picker has already drawn, so the text field is
    updated through `pending` (applied on the next run) rather than written
    directly — Streamlit rejects writes to an instantiated widget's state.
    """
    p = str(normalize_input(str(path)) or path)
    st.session_state[_state_key(key, "selected")] = p
    st.session_state[_state_key(key, "cwd")] = p
    st.session_state[_state_key(key, "pending")] = p
