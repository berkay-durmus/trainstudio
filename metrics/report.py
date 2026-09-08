"""Post-training outputs: plots, tables, an HTML report and model export.

The plots are produced **on a white background**: the in-app view already uses
dark-themed Plotly, and these PNGs are meant for papers and presentations. The
same data is also written as `metrics.csv` and `metrics.xlsx`, so external
analysis is possible.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from core.schemas import Layout, RunStatus

# A plain, high-contrast palette for publication
COLORS = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
          "#0891B2", "#DB2777", "#65A30D"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.frameon": False, "figure.dpi": 130,
    })
    return plt


# ─────────────────────────────────────────────────────────────────────────────
# Shared plots
# ─────────────────────────────────────────────────────────────────────────────


def plot_curves(df, out: Path, columns: list[str], labels: list[str],
                title: str, ylabel: str, mark_best: tuple[int, float] | None = None) -> Path | None:
    plt = _plt()
    present = [(c, l) for c, l in zip(columns, labels) if c in df.columns and df[c].notna().any()]
    if not present:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for i, (col, label) in enumerate(present):
        sub = df[["epoch", col]].dropna()
        ax.plot(sub["epoch"], sub[col], label=label, color=COLORS[i % len(COLORS)],
                linewidth=1.9, marker="o" if len(sub) <= 40 else None, markersize=3.4)
    if mark_best:
        ax.scatter([mark_best[0]], [mark_best[1]], s=110, marker="*", zorder=5,
                   color="#16A34A", label=f"Best (epoch {mark_best[0]})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion(cm: np.ndarray, labels: list[str], out: Path,
                   normalize: bool = True) -> Path:
    plt = _plt()
    m = cm.astype(float)
    if normalize:
        rs = m.sum(axis=1, keepdims=True)
        m = np.divide(m, rs, out=np.zeros_like(m), where=rs > 0)

    size = max(4.6, 0.62 * len(labels) + 2.6)
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if normalize else m.max())
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix" + (" (row normalised)" if normalize else ""))
    ax.grid(False)

    thresh = m.max() / 2 if m.max() else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{m[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if m[i, j] > thresh else "#111827")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_curve_family(data: dict, out: Path, xlabel: str, ylabel: str, title: str,
                      diagonal: bool = False) -> Path | None:
    """ROC / PR curves — {name: (x, y, score)}"""
    if not data:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    if diagonal:
        ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1.1, label="Random")
    for i, (name, (xs, ys, score)) in enumerate(data.items()):
        ax.plot(xs, ys, color=COLORS[i % len(COLORS)], linewidth=1.9,
                label=f"{name} ({score:.3f})")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel), ax.set_ylabel(ylabel), ax.set_title(title)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_bars(labels: list[str], values: list[float], out: Path, title: str,
              xlabel: str, xlim: tuple[float, float] | None = (0, 1)) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.8, max(2.6, 0.42 * len(labels) + 1.4)))
    y = np.arange(len(labels))
    ax.barh(y, values, color=COLORS[0], height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel), ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    for i, v in enumerate(values):
        ax.text(min(v + 0.015, 0.98) if xlim else v, i, f"{v:.3f}",
                va="center", fontsize=8.6)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_histogram(values: list[float], out: Path, title: str, xlabel: str) -> Path | None:
    if not values:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.hist(values, bins=min(30, max(6, len(values) // 3)), color=COLORS[0],
            edgecolor="white", alpha=0.88)
    mean = float(np.mean(values))
    ax.axvline(mean, color="#DC2626", linestyle="--", linewidth=1.6,
               label=f"Mean {mean:.3f}")
    ax.set_xlabel(xlabel), ax.set_ylabel("Number of samples"), ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_calibration(centers, accs, counts, ece: float, out: Path) -> Path | None:
    if not centers:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1.1, label="Perfect calibration")
    ax.bar(centers, accs, width=0.07, color=COLORS[0], alpha=0.85, label="Observed")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1.02)
    ax.set_xlabel("Mean confidence"), ax.set_ylabel("Actual accuracy")
    ax.set_title(f"Calibration (ECE = {ece:.4f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────


def write_tables(trainer, per_class_rows: list[dict] | None = None) -> None:
    import pandas as pd

    cfg = trainer.cfg
    df = pd.DataFrame(trainer.history)
    if df.empty:
        return
    df.to_csv(cfg.path(Layout.METRICS_CSV), index=False)

    if per_class_rows:
        pd.DataFrame(per_class_rows).to_csv(cfg.path(Layout.PER_CLASS), index=False)

    try:
        with pd.ExcelWriter(cfg.path(Layout.METRICS_XLSX), engine="openpyxl") as xl:
            df.to_excel(xl, sheet_name="epochs", index=False)
            if per_class_rows:
                pd.DataFrame(per_class_rows).to_excel(xl, sheet_name="per_class", index=False)
            pd.DataFrame([_flat_config(cfg)]).T.rename(columns={0: "value"}).to_excel(
                xl, sheet_name="config")
    except Exception as exc:
        trainer.w.log(f"Could not write the Excel file: {exc}", "warning")


def _flat_config(cfg) -> dict:
    d = cfg.model_dump(mode="json")
    out: dict[str, object] = {}

    def walk(prefix: str, value) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(value, list):
            out[prefix] = ", ".join(map(str, value))
        else:
            out[prefix] = value

    walk("", d)
    return out


def write_summary(trainer, status: RunStatus, extra: dict | None = None) -> dict:
    cfg = trainer.cfg
    summary = {
        "run_name": cfg.run_name,
        "status": status.value,
        "task": cfg.dataset.task.value,
        "model": cfg.model.display_name or cfg.model.arch,
        "arch": cfg.model.arch,
        "encoder": cfg.model.encoder,
        "dataset": cfg.dataset.root,
        "classes": cfg.dataset.classes,
        "epochs_completed": trainer.epoch,
        "epochs_planned": cfg.hp.epochs,
        "monitor_metric": cfg.hp.monitor_metric,
        "best_value": trainer.best_value,
        "best_epoch": trainer.best_epoch,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **(extra or {}),
    }
    cfg.path(Layout.SUMMARY).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────


def export_model(trainer) -> list[str]:
    """Export the best checkpoint as ONNX / TorchScript."""
    import torch

    cfg = trainer.cfg
    if not (cfg.export_onnx or cfg.export_torchscript):
        return []

    written: list[str] = []
    best = cfg.path(Layout.BEST)
    model = trainer.model
    try:
        if best.is_file():
            state = torch.load(best, map_location="cpu", weights_only=False)
            target = model._orig_mod if hasattr(model, "_orig_mod") else model
            target.load_state_dict(state["model"])
    except Exception as exc:
        trainer.w.log(f"Could not load the best weights, exporting the current ones instead: {exc}",
                      "warning")

    model = model.eval().to("cpu")
    c = cfg.dataset.channels if cfg.dataset.channels in (1, 3) else 3
    dummy = torch.randn(1, c, cfg.hp.img_size, cfg.hp.img_size)

    if cfg.export_onnx:
        try:
            path = cfg.path(Layout.EXPORTS, "model.onnx")
            torch.onnx.export(
                model, dummy, str(path), input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=17,
            )
            written.append(str(path))
            trainer.w.log(f"ONNX written: {path.name}")
        except Exception as exc:
            trainer.w.log(f"The ONNX export failed: {exc}", "warning")

    if cfg.export_torchscript:
        try:
            path = cfg.path(Layout.EXPORTS, "model.torchscript")
            torch.jit.save(torch.jit.trace(model, dummy), str(path))
            written.append(str(path))
            trainer.w.log(f"TorchScript written: {path.name}")
        except Exception as exc:
            trainer.w.log(f"The TorchScript export failed: {exc}", "warning")

    model.to(trainer.device)
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Task-specific finalisation
# ─────────────────────────────────────────────────────────────────────────────


def _load_best_weights(trainer) -> bool:
    import torch

    best = trainer.cfg.path(Layout.BEST)
    if not best.is_file():
        return False
    try:
        state = torch.load(best, map_location=trainer.device, weights_only=False)
        target = trainer.model._orig_mod if hasattr(trainer.model, "_orig_mod") else trainer.model
        target.load_state_dict(state["model"])
        trainer.w.log(f"Best weights loaded (epoch {state.get('epoch')}).")
        return True
    except Exception as exc:
        trainer.w.log(f"Could not load the best weights: {exc}", "warning")
        return False


def finalize_classification(trainer, status: RunStatus) -> None:
    import pandas as pd

    cfg = trainer.cfg
    df = pd.DataFrame(trainer.history)
    if df.empty:
        return
    plots = cfg.path(Layout.PLOTS)

    plot_curves(df, plots / "loss_curve.png", ["train_loss", "val_loss"],
                ["Training", "Validation"], "Loss", "Loss")
    plot_curves(df, plots / "accuracy_curve.png",
                ["train_accuracy", "val_accuracy", "val_balanced_accuracy"],
                ["Training accuracy", "Validation accuracy", "Balanced accuracy"],
                "Accuracy", "Accuracy",
                mark_best=(trainer.best_epoch, trainer.best_value)
                if trainer.best_epoch and cfg.hp.monitor_metric.endswith("accuracy") else None)
    plot_curves(df, plots / "f1_curve.png", ["val_f1_macro", "val_auroc"],
                ["Macro F1", "AUROC"], "F1 and AUROC", "Value")
    plot_curves(df, plots / "lr.png", ["lr"], ["Learning rate"],
                "Learning rate schedule", "lr")

    # Final evaluation — with the best weights, on the test set where available
    _load_best_weights(trainer)
    split = "test" if trainer.loaders.get("test") else "val"
    final = trainer.validate(split=split, full=True)
    metrics = getattr(trainer, "_last_metrics", None)

    per_class_rows: list[dict] = []
    extra: dict = {"final_split": split, "final_metrics": final}

    if metrics is not None and not metrics.empty:
        cm = metrics.confusion_matrix()
        plot_confusion(cm, cfg.dataset.classes, plots / "confusion_matrix.png")
        plot_confusion(cm, cfg.dataset.classes, plots / "confusion_matrix_counts.png",
                       normalize=False)
        plot_curve_family(metrics.roc_data(), plots / "roc.png",
                          "False positive rate", "True positive rate",
                          "ROC curves (one versus rest)", diagonal=True)
        plot_curve_family(metrics.pr_data(), plots / "pr.png",
                          "Recall", "Precision", "Precision–recall curves")
        centers, accs, counts = metrics.calibration_data()
        plot_calibration(centers, accs, counts, final.get("ece", 0.0),
                         plots / "calibration.png")

        per_class_rows = metrics.per_class()
        plot_bars([r["class"] for r in per_class_rows],
                  [r["f1"] for r in per_class_rows],
                  plots / "per_class_f1.png", "F1 per class", "F1")

        try:
            pd.DataFrame(metrics.predictions_table()).to_csv(
                cfg.path(Layout.PREDICTIONS, f"{split}_predictions.csv"), index=False)
        except Exception as exc:
            trainer.w.log(f"Could not write the prediction table: {exc}", "warning")

        # Publication facing: a 95% confidence interval on the test set
        if split == "test":
            try:
                from sklearn.metrics import balanced_accuracy_score

                from metrics.classification import bootstrap_ci

                point, lo, hi = bootstrap_ci(
                    metrics.probs, metrics.targets,
                    lambda p, y: balanced_accuracy_score(y, p.argmax(1)), n_boot=1000,
                )
                extra["balanced_accuracy_ci95"] = [point, lo, hi]
                trainer.w.log(f"Test balanced accuracy: {point:.4f} "
                              f"[95% CI: {lo:.4f}–{hi:.4f}]")
            except Exception:
                pass

    write_tables(trainer, per_class_rows)
    extra["exports"] = export_model(trainer)
    summary = write_summary(trainer, status, extra)
    write_report(trainer, summary, per_class_rows, final, split)


def finalize_segmentation(trainer, status: RunStatus) -> None:
    import pandas as pd
    import torch

    cfg = trainer.cfg
    df = pd.DataFrame(trainer.history)
    if df.empty:
        return
    plots = cfg.path(Layout.PLOTS)

    plot_curves(df, plots / "loss_curve.png", ["train_loss", "val_loss"],
                ["Training", "Validation"], "Loss", "Loss")
    plot_curves(df, plots / "dice_curve.png",
                ["train_dice_macro", "val_dice_macro", "val_iou_macro"],
                ["Training Dice", "Validation Dice", "Validation IoU"],
                "Dice and IoU", "Value",
                mark_best=(trainer.best_epoch, trainer.best_value)
                if trainer.best_epoch else None)
    plot_curves(df, plots / "lr.png", ["lr"], ["Learning rate"],
                "Learning rate schedule", "lr")

    _load_best_weights(trainer)
    split = "test" if trainer.loaders.get("test") else "val"

    # A separate pass for per-sample Dice and the distance metrics
    metrics = trainer.new_metrics()
    dist_rows: list[dict] = []
    trainer.model.eval()
    with torch.no_grad():
        for batch in trainer.loaders[split]:
            logits, target = trainer.forward_batch(batch)
            pred = logits.argmax(dim=1)
            metrics.update(logits, target)
            p_np, t_np = pred.cpu().numpy(), target.cpu().numpy()
            for i in range(p_np.shape[0]):
                metrics.add_sample_dice(p_np[i], t_np[i])
                if len(dist_rows) < 200:      # distance metrics are expensive
                    from metrics.segmentation import distance_metrics_for_case

                    dist_rows.append(distance_metrics_for_case(
                        p_np[i], t_np[i], trainer.n_classes()))

    final = metrics.compute(full=True)
    for key in ("hd95", "assd", "boundary_f1", "volume_similarity"):
        vals = [r[key] for r in dist_rows if key in r]
        if vals:
            final[key] = float(np.mean(vals))

    per_class_rows = metrics.per_class()
    plot_bars([r["class"] for r in per_class_rows], [r["dice"] for r in per_class_rows],
              plots / "per_class_dice.png", "Dice per class", "Dice")
    plot_bars([r["class"] for r in per_class_rows], [r["iou"] for r in per_class_rows],
              plots / "per_class_iou.png", "IoU per class", "IoU")
    plot_histogram(metrics.sample_dices, plots / "dice_distribution.png",
                   f"Per-sample Dice distribution ({split})", "Dice")
    plot_confusion(metrics.confusion_matrix(), cfg.dataset.classes,
                   plots / "confusion_matrix.png")

    try:
        pd.DataFrame({"dice": metrics.sample_dices}).to_csv(
            cfg.path(Layout.PREDICTIONS, f"{split}_per_sample_dice.csv"), index=False)
    except Exception:
        pass

    write_tables(trainer, per_class_rows)
    extra = {"final_split": split, "final_metrics": final,
             "exports": export_model(trainer)}
    summary = write_summary(trainer, status, extra)
    write_report(trainer, summary, per_class_rows, final, split)


# ─────────────────────────────────────────────────────────────────────────────
# The HTML report
# ─────────────────────────────────────────────────────────────────────────────

_REPORT_CSS = """
:root{--bg:#fff;--fg:#111827;--dim:#6B7280;--line:#E5E7EB;--accent:#0F766E}
*{box-sizing:border-box}
body{margin:0;padding:2.5rem 1.5rem;background:var(--bg);color:var(--fg);
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 .2rem} h2{font-size:1.15rem;margin:2.2rem 0 .8rem;
 padding-bottom:.4rem;border-bottom:1px solid var(--line)}
.sub{color:var(--dim);margin:0 0 1.6rem}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.8rem;margin:1.2rem 0}
.kpi{border:1px solid var(--line);border-radius:10px;padding:.85rem 1rem}
.kpi .k{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--dim)}
.kpi .v{font-size:1.35rem;font-weight:650;margin-top:.15rem}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.6rem 0}
th,td{text-align:left;padding:.45rem .7rem;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:600;font-size:.76rem;text-transform:uppercase;letter-spacing:.04em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
img{max-width:100%;border:1px solid var(--line);border-radius:8px;margin:.5rem 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1rem}
.badge{display:inline-block;background:#F0FDFA;color:var(--accent);border:1px solid #99F6E4;
 border-radius:999px;padding:.15rem .6rem;font-size:.75rem;font-weight:600}
code{background:#F3F4F6;padding:.1rem .35rem;border-radius:4px;font-size:.85em}
footer{margin-top:3rem;color:var(--dim);font-size:.8rem;border-top:1px solid var(--line);padding-top:1rem}
"""


def write_report(trainer, summary: dict, per_class_rows: list[dict],
                 final: dict, split: str) -> Path:
    """A single-file, self-contained HTML report (with relative links to the plots)."""
    cfg = trainer.cfg
    plots_dir = cfg.path(Layout.PLOTS)
    images = sorted(p.name for p in plots_dir.glob("*.png")) if plots_dir.is_dir() else []

    def esc(v) -> str:
        import html

        return html.escape(str(v))

    def kpi(label: str, value) -> str:
        return f"<div class='kpi'><div class='k'>{esc(label)}</div><div class='v'>{esc(value)}</div></div>"

    def table(rows: list[dict]) -> str:
        if not rows:
            return "<p class='sub'>No data.</p>"
        cols = list(rows[0])
        head = "".join(
            f"<th class='{'num' if isinstance(rows[0][c], (int, float)) else ''}'>{esc(c)}</th>"
            for c in cols)
        body = ""
        for r in rows:
            cells = "".join(
                f"<td class='num'>{v:.4f}</td>" if isinstance(v, float)
                else f"<td class='{'num' if isinstance(v, int) else ''}'>{esc(v)}</td>"
                for v in (r[c] for c in cols))
            body += f"<tr>{cells}</tr>"
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    metric_rows = [{"metric": k, "value": v} for k, v in sorted(final.items())
                   if isinstance(v, (int, float))]
    ci = summary.get("balanced_accuracy_ci95")
    ci_html = (f"<p class='sub'>Balanced accuracy 95% confidence interval (bootstrap, 1000 resamples): "
               f"<b>{ci[0]:.4f}</b> [{ci[1]:.4f} – {ci[2]:.4f}]</p>") if ci else ""

    cfg_rows = [{"setting": k, "value": v} for k, v in _flat_config(cfg).items()]

    html_out = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(cfg.run_name)} — TrainStudio report</title><style>{_REPORT_CSS}</style></head>
<body><div class="wrap">
<h1>{esc(cfg.run_name)}</h1>
<p class="sub"><span class="badge">{esc(summary.get('status'))}</span>
 {esc(summary.get('model'))} · {esc(cfg.dataset.task.value)} ·
 dataset <code>{esc(Path(cfg.dataset.root).name)}</code> ·
 {esc(summary.get('finished_at'))}</p>
{f"<p class='sub'>{esc(cfg.notes)}</p>" if cfg.notes else ""}

<div class="kpis">
{kpi('Monitored metric', cfg.hp.monitor_metric)}
{kpi('Best value', f"{trainer.best_value:.4f}" if trainer.best_value is not None else "—")}
{kpi('Best epoch', trainer.best_epoch or "—")}
{kpi('Epochs completed', f"{trainer.epoch}/{cfg.hp.epochs}")}
{kpi('Number of classes', cfg.dataset.num_classes)}
{kpi('Training samples', f"{cfg.dataset.n_train:,}")}
</div>

<h2>Final evaluation — the <code>{esc(split)}</code> split</h2>
{ci_html}
{table(metric_rows)}

<h2>Per-class results</h2>
{table(per_class_rows)}

<h2>Plots</h2>
<div class="grid">
{''.join(f'<div><img src="{Layout.PLOTS}/{esc(n)}" alt="{esc(n)}"><div class="sub">{esc(n)}</div></div>' for n in images)}
</div>

<h2>Configuration</h2>
{table(cfg_rows)}

<footer>Generated by TrainStudio · raw data in <code>{Layout.METRICS_CSV}</code>
 and <code>{Layout.METRICS_XLSX}</code> · configuration in
 <code>{Layout.CONFIG}</code>, environment in <code>{Layout.ENV}</code></footer>
</div></body></html>"""

    path = cfg.path(Layout.REPORT)
    path.write_text(html_out, encoding="utf-8")
    trainer.w.log(f"Report written: {path.name}")
    return path
