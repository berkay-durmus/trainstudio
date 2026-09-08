"""The application-wide visual identity: palette, CSS injection, Plotly template."""

from __future__ import annotations

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# The palette — single source of truth. The charts (ui/charts.py) read from it too.
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":            "#0E1117",
    "surface":       "#171D26",
    "surface_hi":    "#1E2632",
    "border":        "#2A333F",
    "border_hi":     "#3A4654",
    "text":          "#E6EDF3",
    "text_dim":      "#8B98A5",
    "text_faint":    "#5C6773",
    "accent":        "#2DD4BF",
    "accent_dim":    "#1A9E90",
    "running":       "#F59E0B",
    "completed":     "#22C55E",
    "failed":        "#EF4444",
    "stopped":       "#6B7280",
    "queued":        "#60A5FA",
}

# Series colours, cycled through in the run comparison charts.
SERIES_COLORS = [
    "#2DD4BF", "#60A5FA", "#F472B6", "#FBBF24", "#A78BFA",
    "#34D399", "#FB923C", "#F87171", "#22D3EE", "#C084FC",
]

STATUS_COLORS = {
    "queued":    PALETTE["queued"],
    "running":   PALETTE["running"],
    "completed": PALETTE["completed"],
    "failed":    PALETTE["failed"],
    "stopped":   PALETTE["stopped"],
}

_CSS = """
<style>
/* ── Simplify the Streamlit chrome ─────────────────────────────────── */
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; height: 0; }

.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1500px; }

/* ── Typography ───────────────────────────────────────────────────── */
h1, h2, h3, h4 { letter-spacing: -0.015em; font-weight: 650; }
h1 { font-size: 1.85rem !important; }
h2 { font-size: 1.30rem !important; margin-top: 0.4rem !important; }
h3 { font-size: 1.05rem !important; }

/* ── Sidebar ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: {surface};
  border-right: 1px solid {border};
}
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

.ts-brand {
  display: flex; align-items: center; gap: .6rem;
  padding: .25rem .25rem 1rem .25rem; margin-bottom: .5rem;
  border-bottom: 1px solid {border};
}
.ts-brand-mark {
  width: 32px; height: 32px; flex: 0 0 32px; border-radius: 9px;
  background: linear-gradient(135deg, {accent} 0%, {accent_dim} 100%);
  display: grid; place-items: center;
  font-weight: 800; font-size: .95rem; color: #06231F;
}
.ts-brand-name { font-weight: 700; font-size: 1.02rem; line-height: 1.1; }
.ts-brand-sub  { font-size: .70rem; color: {text_faint}; letter-spacing: .06em; text-transform: uppercase; }

/* ── Card ─────────────────────────────────────────────────────────── */
.ts-card {
  background: {surface}; border: 1px solid {border}; border-radius: 12px;
  padding: 1rem 1.1rem; margin-bottom: .85rem;
}
.ts-card-tight { padding: .75rem .9rem; }
.ts-card:hover { border-color: {border_hi}; }
.ts-card-title { font-weight: 650; font-size: .98rem; margin: 0 0 .15rem 0; }
.ts-card-sub   { font-size: .78rem; color: {text_dim}; margin: 0; }

/* ── Badges / pills ───────────────────────────────────────────────── */
.ts-badges { display: flex; flex-wrap: wrap; gap: .32rem; margin: .55rem 0 .1rem 0; }
.ts-badge {
  font-size: .70rem; font-weight: 600; letter-spacing: .01em;
  padding: .16rem .48rem; border-radius: 999px;
  background: {surface_hi}; color: {text_dim}; border: 1px solid {border};
  white-space: nowrap;
}
.ts-badge-accent { background: rgba(45,212,191,.12); color: {accent}; border-color: rgba(45,212,191,.35); }
.ts-badge-new    { background: rgba(45,212,191,.18); color: {accent}; border-color: {accent}; }

.ts-pill {
  display: inline-flex; align-items: center; gap: .38rem;
  font-size: .76rem; font-weight: 650;
  padding: .2rem .6rem; border-radius: 999px;
}
.ts-dot { width: 7px; height: 7px; border-radius: 50%; flex: 0 0 7px; }
.ts-dot-pulse { animation: ts-pulse 1.4s ease-in-out infinite; }
@keyframes ts-pulse { 0%,100% { opacity: 1; } 50% { opacity: .28; } }

/* ── Breadcrumb / step indicator ──────────────────────────────────── */
.ts-steps {
  display: flex; align-items: center; gap: .1rem; flex-wrap: wrap;
  margin: -.5rem 0 1.4rem 0; font-size: .78rem;
}
.ts-step { display: flex; align-items: center; gap: .4rem; padding: .2rem .55rem; border-radius: 7px; color: {text_faint}; }
.ts-step-done { color: {text_dim}; }
.ts-step-active { color: {accent}; background: rgba(45,212,191,.10); font-weight: 650; }
.ts-step-idx {
  width: 17px; height: 17px; border-radius: 50%; flex: 0 0 17px;
  display: grid; place-items: center; font-size: .62rem; font-weight: 700;
  border: 1px solid currentColor;
}
.ts-step-active .ts-step-idx { background: {accent}; color: #06231F; border-color: {accent}; }
.ts-step-sep { color: {border_hi}; padding: 0 .05rem; }

/* ── Metric ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: {surface}; border: 1px solid {border};
  border-radius: 11px; padding: .7rem .9rem;
}
[data-testid="stMetricLabel"] p {
  font-size: .70rem !important; color: {text_dim} !important;
  text-transform: uppercase; letter-spacing: .06em; font-weight: 600;
}
[data-testid="stMetricValue"] { font-size: 1.42rem !important; font-weight: 680; }

/* ── Buttons ──────────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button {
  border-radius: 9px; font-weight: 600; border: 1px solid {border_hi};
  transition: border-color .12s, background .12s;
}
.stButton > button:hover { border-color: {accent}; color: {accent}; }
.stButton > button[kind="primary"] {
  background: {accent}; color: #06231F; border-color: {accent};
}
.stButton > button[kind="primary"]:hover { background: #5DE4D3; color: #06231F; }

/* ── Inputs ───────────────────────────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div {
  border-radius: 8px !important; font-size: .88rem;
}
code, .stCode { font-size: .80rem !important; }

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: .1rem; border-bottom: 1px solid {border}; }
.stTabs [data-baseweb="tab"] { font-size: .88rem; font-weight: 600; padding: .5rem .95rem; }

/* ── Utilities ────────────────────────────────────────────────────── */
.ts-mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .78rem; }
.ts-dim  { color: {text_dim}; font-size: .82rem; }
.ts-faint{ color: {text_faint}; font-size: .76rem; }
.ts-kv   { display: flex; justify-content: space-between; gap: 1rem; padding: .22rem 0;
           border-bottom: 1px dashed {border}; font-size: .82rem; }
.ts-kv:last-child { border-bottom: none; }
.ts-kv-k { color: {text_dim}; }
.ts-kv-v { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.ts-scroll { max-height: 620px; overflow-y: auto; padding-right: .45rem; }
.ts-scroll::-webkit-scrollbar { width: 8px; }
.ts-scroll::-webkit-scrollbar-thumb { background: {border_hi}; border-radius: 4px; }
hr { margin: 1rem 0; border-color: {border}; }
</style>
"""


def inject_theme() -> None:
    """Called once at the top of every page (idempotent)."""
    css = _CSS
    for key, value in PALETTE.items():
        css = css.replace("{" + key + "}", value)
    st.markdown(css, unsafe_allow_html=True)


def plotly_layout(**overrides) -> dict:
    """The Plotly layout base shared by every chart."""
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], size=12),
        margin=dict(l=52, r=18, t=34, b=42),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        xaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
        yaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
    )
    layout.update(overrides)
    return layout
