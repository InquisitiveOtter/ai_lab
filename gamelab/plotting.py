"""Plots of a training run's `metrics.jsonl`.

Uses matplotlib's object API (no pyplot), so it works headless and in worker
processes; in a notebook, a returned Figure displays itself.
"""

import json
from pathlib import Path

from matplotlib.figure import Figure

# Categorical slots in fixed order, plus text and surface colors.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def load_metrics(path: str | Path) -> list[dict]:
    """Read a metrics.jsonl file, or the one inside a run directory."""
    path = Path(path)
    if path.is_dir():
        path = path / "metrics.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _style(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    ax.set_xlabel("iteration", color=TEXT_MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)


def _series(metrics: list[dict], key: str) -> tuple[list, list]:
    rows = [m for m in metrics if key in m]
    return [m["iteration"] for m in rows], [m[key] for m in rows]


def plot_series(ax, metrics: list[dict], keys: list[str], labels: list[str] | None = None) -> None:
    """Line plot of one or more metric keys against iteration, one color per key."""
    labels = labels or keys
    for i, (key, label) in enumerate(zip(keys, labels)):
        x, y = _series(metrics, key)
        ax.plot(x, y, color=SERIES[i % len(SERIES)], linewidth=2, label=label,
                marker="o" if len(x) < 30 else None, markersize=4)
    if len(keys) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)


def plot_training(source: str | Path | list[dict], save_to: str | Path | None = None) -> Figure:
    """A 2x2 overview: policy loss, value loss, evaluation scores, game length.

    `source` is a run directory, a metrics.jsonl path, or loaded metrics.
    """
    metrics = source if isinstance(source, list) else load_metrics(source)
    fig = Figure(figsize=(11, 7), layout="constrained", facecolor=SURFACE)
    (a, b), (c, d) = fig.subplots(2, 2)

    plot_series(a, metrics, ["train/policy_loss"])
    _style(a, "Policy loss (cross-entropy)")
    plot_series(b, metrics, ["train/value_loss"])
    _style(b, "Value loss (MSE)")

    score_keys = sorted({k for m in metrics for k in m if k.startswith("eval/") and k.endswith("/score")})
    plot_series(c, metrics, score_keys, [k.split("/")[1] for k in score_keys])
    c.set_ylim(-0.02, 1.02)
    _style(c, "Score vs baselines", "(wins + draws/2) / games")
    if not score_keys:
        c.text(0.5, 0.5, "no evaluations yet", ha="center", va="center", color=TEXT_MUTED, transform=c.transAxes)

    plot_series(d, metrics, ["selfplay/mean_length"])
    _style(d, "Self-play game length", "moves")

    if save_to is not None:
        fig.savefig(save_to, dpi=120)
    return fig


def compare_runs(run_dirs: dict[str, str | Path], key: str, save_to: str | Path | None = None) -> Figure:
    """One metric across several runs, e.g. `compare_runs({"small": ..., "big": ...}, "eval/random/score")`."""
    fig = Figure(figsize=(7, 4), layout="constrained", facecolor=SURFACE)
    ax = fig.subplots()
    for i, (label, run_dir) in enumerate(run_dirs.items()):
        x, y = _series(load_metrics(run_dir), key)
        ax.plot(x, y, color=SERIES[i % len(SERIES)], linewidth=2, label=label)
    if len(run_dirs) > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    _style(ax, key)
    if save_to is not None:
        fig.savefig(save_to, dpi=120)
    return fig
