"""Reusable interface pieces.

Where Streamlit's own components fall short we produce small fragments of HTML;
all of them use the classes defined in ui/theme.py, with no inline styling.
"""

from __future__ import annotations

import html
from typing import Iterable, Sequence

import streamlit as st

from ui.theme import PALETTE, STATUS_COLORS


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page skeleton
# ─────────────────────────────────────────────────────────────────────────────

STEPS: list[tuple[str, str]] = [
    ("Dataset", "pages/1_Dataset.py"),
    ("Model", "pages/2_Model_Selection.py"),
    ("Settings", "pages/3_Settings.py"),
    ("Training", "pages/4_Training.py"),
    ("Results", "pages/5_Results.py"),
]


def page_header(title: str, subtitle: str = "", active_step: int | None = None,
                done_steps: Iterable[int] = ()) -> None:
    """Title plus the step indicator. `active_step` starts at 1."""
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(f"<p class='ts-dim' style='margin-top:-.6rem'>{_esc(subtitle)}</p>",
                    unsafe_allow_html=True)
    if active_step is not None:
        done = set(done_steps)
        parts = []
        for i, (name, _) in enumerate(STEPS, start=1):
            if i == active_step:
                cls = "ts-step ts-step-active"
            elif i in done:
                cls = "ts-step ts-step-done"
            else:
                cls = "ts-step"
            mark = "✓" if (i in done and i != active_step) else str(i)
            parts.append(f"<div class='{cls}'><span class='ts-step-idx'>{mark}</span>{_esc(name)}</div>")
        sep = "<span class='ts-step-sep'>›</span>"
        st.markdown(f"<div class='ts-steps'>{sep.join(parts)}</div>", unsafe_allow_html=True)


def brand_sidebar() -> None:
    st.sidebar.markdown(
        "<div class='ts-brand'>"
        "<div class='ts-brand-mark'>TS</div>"
        "<div><div class='ts-brand-name'>TrainStudio</div>"
        "<div class='ts-brand-sub'>Model Training Platform</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Small pieces
# ─────────────────────────────────────────────────────────────────────────────


def badge(text: str, kind: str = "") -> str:
    """`kind`: "" | "accent" | "new" """
    extra = f" ts-badge-{kind}" if kind else ""
    return f"<span class='ts-badge{extra}'>{_esc(text)}</span>"


def badges(items: Sequence[tuple[str, str] | str]) -> str:
    out = []
    for it in items:
        if isinstance(it, tuple):
            out.append(badge(it[0], it[1]))
        else:
            out.append(badge(it))
    return f"<div class='ts-badges'>{''.join(out)}</div>"


def status_pill(status: str, label: str | None = None, pulse: bool | None = None) -> str:
    color = STATUS_COLORS.get(status, PALETTE["text_dim"])
    if pulse is None:
        pulse = status == "running"
    dot_cls = "ts-dot ts-dot-pulse" if pulse else "ts-dot"
    text = label or status
    return (
        f"<span class='ts-pill' style='background:{color}1F;color:{color}'>"
        f"<span class='{dot_cls}' style='background:{color}'></span>{_esc(text)}</span>"
    )


def kv_rows(pairs: Sequence[tuple[str, object]]) -> str:
    rows = "".join(
        f"<div class='ts-kv'><span class='ts-kv-k'>{_esc(k)}</span>"
        f"<span class='ts-kv-v'>{_esc(v)}</span></div>"
        for k, v in pairs
    )
    return f"<div>{rows}</div>"


def card(body_html: str, tight: bool = False) -> str:
    cls = "ts-card ts-card-tight" if tight else "ts-card"
    return f"<div class='{cls}'>{body_html}</div>"


def dim(text: str) -> None:
    st.markdown(f"<p class='ts-dim'>{_esc(text)}</p>", unsafe_allow_html=True)


def faint(text: str) -> None:
    st.markdown(f"<p class='ts-faint'>{_esc(text)}</p>", unsafe_allow_html=True)


def hint(text: str) -> None:
    """Used to show the recommendation engine's rationale underneath a field."""
    if text:
        st.markdown(f"<div class='ts-faint' style='margin-top:-.7rem;margin-bottom:.6rem'>"
                    f"{_esc(text)}</div>", unsafe_allow_html=True)


def empty_state(icon: str, title: str, detail: str = "", cta: str = "") -> None:
    st.markdown(
        f"<div class='ts-card' style='text-align:center;padding:2.6rem 1rem'>"
        f"<div style='font-size:2.2rem;line-height:1'>{icon}</div>"
        f"<div style='font-weight:650;margin-top:.7rem'>{_esc(title)}</div>"
        f"<div class='ts-dim' style='margin-top:.3rem'>{_esc(detail)}</div>"
        + (f"<div class='ts-faint' style='margin-top:.6rem'>{_esc(cta)}</div>" if cta else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def issue_list(issues, limit: int = 30) -> None:
    """Render a list of data.scan.Issue objects by severity."""
    for iss in issues[:limit]:
        body = iss.detail
        if iss.items:
            shown = ", ".join(str(x) for x in iss.items[:6])
            more = f" … (+{len(iss.items) - 6})" if len(iss.items) > 6 else ""
            body = f"{body}\n\n`{shown}{more}`" if body else f"`{shown}{more}`"
        fn = {"error": st.error, "warning": st.warning, "info": st.info}[iss.level]
        fn(f"**{iss.title}**" + (f"\n\n{body}" if body else ""))


def metric_row(items: Sequence[tuple[str, object, object]]) -> None:
    """A KPI strip from (label, value, delta) triples. delta may be None."""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col:
            st.metric(label, value, delta)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def fmt_count(n: int | None) -> str:
    return "—" if n is None else f"{n:,}"


def fmt_metric(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1000 or (v != 0 and abs(v) < 1e-3):
        return f"{v:.3e}"
    return f"{v:.{digits}f}"
