"""Plotly charts — all of them use the ui/theme.py palette and the shared layout.

On the live training page the charts are redrawn several times per second, so
`uirevision` is kept constant: the user's zoom and pan are preserved.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from ui.theme import PALETTE, SERIES_COLORS, plotly_layout


def _fig(title: str = "", height: int = 320, uirev: str = "keep", **layout) -> go.Figure:
    fig = go.Figure()
    # The key is omitted rather than set to None when there is no title:
    # `title=None` reaches Plotly.js as a null layout.title and is drawn as the
    # literal text "undefined" above the chart.
    if title:
        layout["title"] = dict(text=title,
                               font=dict(size=13, color=PALETTE["text_dim"]))
    fig.update_layout(**plotly_layout(height=height, uirevision=uirev, **layout))
    return fig


def color_for(i: int) -> str:
    return SERIES_COLORS[i % len(SERIES_COLORS)]


# ─────────────────────────────────────────────────────────────────────────────
# Curves
# ─────────────────────────────────────────────────────────────────────────────


def curves(
    series: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    title: str = "",
    y_title: str = "",
    x_title: str = "Epoch",
    height: int = 320,
    log_y: bool = False,
    markers: bool = True,
    uirev: str = "curves",
    highlight: tuple[float, float] | None = None,   # (x, y) — the best point
) -> go.Figure:
    """Draw several series on one axis. `series` = {name: (x, y)}"""
    fig = _fig(title, height, uirev, xaxis_title=x_title, yaxis_title=y_title)
    for i, (name, (xs, ys)) in enumerate(series.items()):
        if not len(xs):
            continue
        fig.add_trace(go.Scatter(
            x=list(xs), y=list(ys), name=name,
            mode="lines+markers" if markers and len(xs) < 120 else "lines",
            line=dict(color=color_for(i), width=2.2),
            marker=dict(size=5),
            hovertemplate=f"<b>{name}</b>: %{{y:.5f}}<extra></extra>",
        ))
    if highlight is not None:
        fig.add_trace(go.Scatter(
            x=[highlight[0]], y=[highlight[1]], mode="markers", name="Best",
            marker=dict(size=13, color=PALETTE["completed"], symbol="star",
                        line=dict(width=1, color="#0E1117")),
            hovertemplate="Best: %{y:.5f} (epoch %{x})<extra></extra>",
        ))
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def multi_run_curves(df, metric_label: str = "", height: int = 400) -> go.Figure:
    """Overlay the output of core.runs.comparison_frame (epoch, value, run)."""
    fig = _fig("", height, "compare", xaxis_title="Epoch", yaxis_title=metric_label)
    for i, run in enumerate(sorted(df["run"].unique())):
        sub = df[df["run"] == run].sort_values("epoch")
        fig.add_trace(go.Scatter(
            x=sub["epoch"], y=sub["value"], name=run, mode="lines",
            line=dict(color=color_for(i), width=2.2),
            hovertemplate=f"<b>{run}</b><br>epoch %{{x}}: %{{y:.5f}}<extra></extra>",
        ))
    return fig


def sparkline(values: Sequence[float], color: str | None = None, height: int = 60,
              y_range: tuple[float, float] | None = None, fill: bool = True) -> go.Figure:
    fig = _fig("", height, "spark", margin=dict(l=0, r=0, t=4, b=0), hovermode="closest",
               showlegend=False)
    c = color or PALETTE["accent"]
    fig.add_trace(go.Scatter(
        y=list(values), mode="lines", line=dict(color=c, width=1.8),
        fill="tozeroy" if fill else None, fillcolor=f"{c}22",
        hovertemplate="%{y:.1f}<extra></extra>",
    ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=y_range)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Distributions
# ─────────────────────────────────────────────────────────────────────────────


def class_distribution(
    counts_by_split: Mapping[str, Mapping[str, int]],
    classes: Sequence[str],
    height: int = 300,
) -> go.Figure:
    """The class distribution stacked by split."""
    fig = _fig("", height, "dist", barmode="stack", xaxis_title="",
               yaxis_title="Number of samples")
    split_colors = {"train": PALETTE["accent"], "val": "#60A5FA", "test": "#A78BFA"}
    for split, counts in counts_by_split.items():
        fig.add_trace(go.Bar(
            x=list(classes), y=[counts.get(c, 0) for c in classes], name=split,
            marker_color=split_colors.get(split, PALETTE["text_dim"]),
            hovertemplate=f"<b>{split}</b> %{{x}}: %{{y}}<extra></extra>",
        ))
    return fig


def horizontal_bars(labels: Sequence[str], values: Sequence[float], title: str = "",
                    x_title: str = "", height: int | None = None,
                    color: str | None = None, x_range: tuple[float, float] | None = None) -> go.Figure:
    h = height or max(180, 34 * len(labels) + 70)
    fig = _fig(title, h, "hbar", xaxis_title=x_title, showlegend=False, hovermode="closest")
    fig.add_trace(go.Bar(
        x=list(values), y=list(labels), orientation="h",
        marker_color=color or PALETTE["accent"],
        text=[f"{v:.3f}" if isinstance(v, float) else str(v) for v in values],
        textposition="auto",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    if x_range:
        fig.update_xaxes(range=list(x_range))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Classification metric visuals
# ─────────────────────────────────────────────────────────────────────────────


def confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str],
    normalize: bool = True,
    height: int = 420,
) -> go.Figure:
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        z = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        fmt, zmax = ".2f", 1.0
    else:
        z, fmt, zmax = cm, ".0f", float(cm.max() or 1)

    text = [[f"{v:{fmt}}" for v in row] for row in z]
    fig = _fig("", height, "cm", xaxis_title="Predicted", yaxis_title="Actual",
               hovermode="closest")
    fig.add_trace(go.Heatmap(
        z=z, x=list(labels), y=list(labels), text=text, texttemplate="%{text}",
        colorscale=[[0, PALETTE["surface"]], [0.5, "#1A9E90"], [1, PALETTE["accent"]]],
        zmin=0, zmax=zmax, showscale=False,
        hovertemplate="Actual %{y} → Predicted %{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return fig


def roc_curves(curves_by_class: Mapping[str, tuple[Sequence[float], Sequence[float], float]],
               height: int = 380) -> go.Figure:
    """{class: (fpr, tpr, auc)}"""
    fig = _fig("", height, "roc", xaxis_title="False positive rate",
               yaxis_title="True positive rate", hovermode="closest")
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(color=PALETTE["border_hi"], dash="dash", width=1.2),
                             hoverinfo="skip"))
    for i, (name, (fpr, tpr, auc)) in enumerate(curves_by_class.items()):
        fig.add_trace(go.Scatter(x=list(fpr), y=list(tpr), mode="lines",
                                 name=f"{name} (AUC {auc:.3f})",
                                 line=dict(color=color_for(i), width=2)))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def pr_curves(curves_by_class: Mapping[str, tuple[Sequence[float], Sequence[float], float]],
              height: int = 380) -> go.Figure:
    """{class: (recall, precision, ap)}"""
    fig = _fig("", height, "pr", xaxis_title="Recall",
               yaxis_title="Precision", hovermode="closest")
    for i, (name, (rec, prec, ap)) in enumerate(curves_by_class.items()):
        fig.add_trace(go.Scatter(x=list(rec), y=list(prec), mode="lines",
                                 name=f"{name} (AP {ap:.3f})",
                                 line=dict(color=color_for(i), width=2)))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def reliability_diagram(bin_centers, accuracies, counts, ece: float | None = None,
                        height: int = 340) -> go.Figure:
    """The calibration curve — does the model's confidence match its real accuracy."""
    title = f"Calibration (ECE = {ece:.4f})" if ece is not None else "Calibration"
    fig = _fig(title, height, "cal", xaxis_title="Mean confidence",
               yaxis_title="Actual accuracy", hovermode="closest")
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                             line=dict(color=PALETTE["border_hi"], dash="dash", width=1.2)))
    fig.add_trace(go.Bar(x=list(bin_centers), y=list(accuracies), name="Observed",
                         marker_color=PALETTE["accent"], opacity=0.85,
                         customdata=list(counts),
                         hovertemplate="confidence %{x:.2f} → accuracy %{y:.3f}"
                                       "<br>%{customdata} samples<extra></extra>"))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def gpu_gauge(used_gb: float, total_gb: float, height: int = 150) -> go.Figure:
    pct = 100 * used_gb / total_gb if total_gb else 0
    color = PALETTE["completed"] if pct < 70 else (
        PALETTE["running"] if pct < 90 else PALETTE["failed"])
    fig = _fig("", height, "gauge", margin=dict(l=10, r=10, t=10, b=4), showlegend=False)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=used_gb,
        number=dict(suffix=" GB", font=dict(size=20)),
        gauge=dict(
            axis=dict(range=[0, total_gb], tickwidth=1, tickcolor=PALETTE["border"]),
            bar=dict(color=color, thickness=0.7),
            bgcolor=PALETTE["surface"], borderwidth=0,
        ),
    ))
    return fig
