"""Publication-quality Pareto-front figures (v0.2.0).

Renders one or more fronts (e.g. NSGA-II vs NRGA) overlaid, adapting to the
objective dimensionality:

    2 objectives -> a single Z1-Z2 scatter.
    3 objectives -> a 2x2 panel: the three pairwise projections (Z1-Z2, Z1-Z3,
                    Z2-Z3) plus a 3-D scatter.

matplotlib is an optional dependency (the ``viz`` extra). Import errors are
surfaced with an actionable message by the CLI. The Agg backend is selected so
figures render headlessly (CI, servers) and are written to PNG/SVG by extension.

Public API:
    plot_fronts(fronts, output_path, title=None) -> Path
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402

from presidio_vol_assign.models import ParetoFront  # noqa: E402

# Distinct, colour-blind-friendly styles cycled per front.
_STYLES = [
    {"color": "#0072B2", "marker": "o"},
    {"color": "#D55E00", "marker": "s"},
    {"color": "#009E73", "marker": "^"},
    {"color": "#CC79A7", "marker": "D"},
]


def _points(front: ParetoFront) -> list[tuple[float, ...]]:
    return [s.objectives for s in front.solutions]


def plot_fronts(
    fronts: list[ParetoFront],
    output_path: Path,
    title: str | None = None,
) -> Path:
    """Plot one or more Pareto fronts and save to *output_path*.

    Raises:
        ValueError: If *fronts* is empty, any front is empty, or the fronts have
            inconsistent objective dimensions.
    """
    if not fronts:
        raise ValueError("no fronts to plot")

    dims = {len(s.objectives) for f in fronts for s in f.solutions}
    if not dims:
        raise ValueError("fronts contain no solutions to plot")
    if len(dims) > 1:
        raise ValueError(f"inconsistent objective dimensions across fronts: {sorted(dims)}")
    dim = dims.pop()

    if dim == 2:
        fig = _plot_2d(fronts)
    elif dim == 3:
        fig = _plot_3d(fronts)
    else:
        raise ValueError(f"plotting is supported for 2 or 3 objectives, not {dim}")

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _label(front: ParetoFront) -> str:
    return front.solver.value.upper()


def _plot_2d(fronts: list[ParetoFront]):
    fig, ax = plt.subplots(figsize=(6, 5))
    for front, style in zip(fronts, _cycle(len(fronts))):
        pts = _points(front)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, label=_label(front), alpha=0.8, edgecolors="none", **style)
    ax.set_xlabel("Z1")
    ax.set_ylabel("Z2")
    ax.set_title("Pareto front")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def _plot_3d(fronts: list[ParetoFront]):
    fig = plt.figure(figsize=(11, 9))
    pairs = [(0, 1, "Z1", "Z2"), (0, 2, "Z1", "Z3"), (1, 2, "Z2", "Z3")]
    styles = list(_cycle(len(fronts)))

    for idx, (a, b, xlabel, ylabel) in enumerate(pairs):
        ax = fig.add_subplot(2, 2, idx + 1)
        for front, style in zip(fronts, styles):
            pts = _points(front)
            ax.scatter(
                [p[a] for p in pts],
                [p[b] for p in pts],
                label=_label(front),
                alpha=0.8,
                edgecolors="none",
                **style,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel} vs {ylabel}")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    ax3d = fig.add_subplot(2, 2, 4, projection="3d")
    for front, style in zip(fronts, styles):
        pts = _points(front)
        ax3d.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            [p[2] for p in pts],
            label=_label(front),
            alpha=0.8,
            **style,
        )
    ax3d.set_xlabel("Z1")
    ax3d.set_ylabel("Z2")
    ax3d.set_zlabel("Z3")
    ax3d.set_title("Z1 vs Z2 vs Z3")
    return fig


def _cycle(n: int) -> list[dict]:
    return [_STYLES[i % len(_STYLES)] for i in range(n)]
