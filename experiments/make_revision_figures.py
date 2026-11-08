"""Figures of the Paper B revision (R1): multiplicity, turbulence, envelope.

Same palette as ``make_audit_figures`` with a larger base font (R3: "Figures 4 and 5 would
benefit from larger labels"). Writes PDFs into ``--out``.

Usage::

    python -m experiments.make_revision_figures --out <dir> [multiplicity turbulence envelope]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
RESULTS = Path("experiments/results")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
    }
)

MODELS = {
    "published": ("published", BLUE),
    "soft_k2": ("soft, $\\kappa=2$", AQUA),
    "hard_kf1.5": ("hard, $\\kappa_f=1.5$", ORANGE),
}
# Seed floor of the evolutionary system at 150 generations (paper Section 5.2): churn, Jaccard.
SEED_FLOOR = {"small": (0.834, 0.31), "large": (0.939, 0.25)}
MOEA_DEFICIT = {"small": 17.8, "large": 33.3}


def fig_multiplicity(out: Path) -> None:
    rows = list(csv.DictReader((RESULTS / "multiplicity" / "summary.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for size, style in (("small", "-"), ("large", "--")):
        for model, (label, colour) in MODELS.items():
            pairs = sorted(
                (100 * float(r["epsilon_rel"]), float(r["max_churn_vs_optimum"]))
                for r in rows
                if r["size"] == size and r["model"] == model and r["mode"] == "pairs"
            )
            jacc = sorted(
                (100 * float(r["epsilon_rel"]), float(r["min_jaccard_vs_optimum"]))
                for r in rows
                if r["size"] == size and r["model"] == model and r["mode"] == "directed"
            )
            name = f"{label}, {size}"
            axes[0].plot(
                *zip(*pairs, strict=True), style, color=colour, marker="o", ms=3, label=name
            )
            axes[1].plot(*zip(*jacc, strict=True), style, color=colour, marker="o", ms=3)
        z = next(
            float(r["z_star"]) for r in rows if r["size"] == size and r["model"] == "published"
        )
        x = 100 * MOEA_DEFICIT[size] / z
        churn, jaccard = SEED_FLOOR[size]
        marker = "*" if size == "small" else "X"
        axes[0].plot(x, churn, marker, color=INK, ms=9, label=f"seed floor, {size}")
        axes[1].plot(x, jaccard, marker, color=INK, ms=9)
    axes[0].set(
        xlabel="$\\varepsilon$ (% of optimal value)",
        ylabel="largest churn vs. optimum",
        ylim=(-0.03, 1.05),
    )
    axes[1].set(
        xlabel="$\\varepsilon$ (% of optimal value)",
        ylabel="smallest Jaccard vs. optimum",
        ylim=(-0.03, 1.05),
    )
    for ax in axes:
        ax.set_xscale("symlog", linthresh=0.1, linscale=0.4)
        ax.set_xticks([0, 0.1, 0.5, 2, 10, 40], ["0", "0.1", "0.5", "2", "10", "40"])
        ax.set_xlim(-0.01, 48)
    axes[0].legend(loc="upper left", fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(out / "fig_multiplicity.pdf")
    plt.close(fig)


_FIELD = {
    "infrastructure_damage_level": "damage",
    "resource_time_remaining": "resource time",
    "center_occupancy_rate": "occupancy",
    "travel_duration": "travel time",
    "road_condition": "road",
    "possible_hazard": "hazard",
}
_LABELS = {
    f"{k}_{m}": f"{v}, {t}"
    for k, v in _FIELD.items()
    for m, t in (
        ("noise", "noise"),
        ("missingness", "missing"),
        ("flip", "flip"),
        ("bias_up", "bias +"),
        ("bias_down", "bias \u2212"),
    )
}
_LABELS.update(
    {
        "joint_all": "joint: all eight",
        "joint_continuous_noise": "joint: continuous noise",
        "joint_transport": "joint: transport",
    }
)


def fig_turbulence(out: Path) -> None:
    """Regret of the exact decider across instances and conditions, per cell, at each level."""
    data: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for size in ("small", "large"):
        for path in sorted((RESULTS / "turbulence_extended" / size).glob("manifest_*.csv")):
            per_condition: dict[tuple, list[float]] = defaultdict(list)
            for r in csv.DictReader(path.open()):
                key = (r["instance"], r["demand"], r["occupancy"], r["cell"], float(r["level"]))
                per_condition[key].append(float(r["regret"]))
            for (_, _, _, cell, level), vals in per_condition.items():
                data[(size, cell)][level].append(float(np.mean(vals)))
    cells = sorted({c for _, c in data})
    # one row order for both panels: by the small instance's median regret at level 0.4
    order = sorted(cells, key=lambda c: -np.median(data[("small", c)].get(0.4, [0])))
    pos = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.0), sharey=True)
    for ax, size in zip(axes, ("small", "large"), strict=True):
        for k, (level, colour) in enumerate(((0.05, AQUA), (0.1, BLUE), (0.4, ORANGE))):
            vals = [data[(size, c)].get(level, [np.nan]) for c in order]
            med = np.array([np.median(v) for v in vals])
            lo = np.array([np.percentile(v, 10) for v in vals])
            hi = np.array([np.percentile(v, 90) for v in vals])
            ax.errorbar(
                med,
                pos + (k - 1) * 0.25,
                xerr=[med - lo, hi - med],
                fmt="o",
                color=colour,
                ms=3.5,
                lw=1,
                label=f"level {level:g}",
            )
        ax.set(xlabel="regret (objective units)", title=f"{size} instances")
    axes[0].set_yticks(pos, [_LABELS.get(c, c) for c in order], fontsize=8.5)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig4_turbulence_regret.pdf")
    plt.close(fig)


def fig_envelope(out: Path) -> None:
    rows = []
    for path in sorted((RESULTS / "envelope").glob("envelope_t*_s*.csv")):
        rows += list(csv.DictReader(path.open()))
    methods = (
        ("mip_soft", "MIP, soft", BLUE, "o"),
        ("ls_soft", "local search, soft", AQUA, "s"),
        ("nsga2_seeded_best_point", "NSGA-II seeded", MUTED, "^"),
        ("nsga2_best_point", "NSGA-II", ORANGE, "v"),
        ("fallback", "fallback", INK, "x"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for limit, ax in zip(("60", "300"), axes, strict=True):
        sub = [r for r in rows if r["limit"].rstrip("0").rstrip(".") == limit]
        sizes = sorted({(int(r["people"]), int(r["centres"])) for r in sub})
        x = np.arange(len(sizes))
        for method, label, colour, marker in methods:
            means, unsolved = [], []
            for size in sizes:
                cell = [
                    r
                    for r in sub
                    if (int(r["people"]), int(r["centres"])) == size and r["method"] == method
                ]
                solved = [float(r["quality_soft"]) for r in cell if r["solved"] == "True"]
                means.append(np.mean(solved) if solved else np.nan)
                unsolved.append(len(cell) - len(solved))
            ax.plot(x, means, marker=marker, color=colour, label=label, ms=4)
            for xi, (m, u) in enumerate(zip(means, unsolved, strict=True)):
                if u:
                    ax.annotate(
                        f"{u} unsolved",
                        (xi, m if np.isfinite(m) else 100),
                        fontsize=7,
                        color=colour,
                        xytext=(3, 6),
                        textcoords="offset points",
                    )
        ax.set_xticks(x, [f"{p}×{c}" for p, c in sizes], rotation=30)
        ax.set(
            xlabel="people × centres",
            ylabel="quality, soft repaired model",
            title=f"{limit} s limit",
        )
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig5_scaling.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("figures", nargs="*", default=["multiplicity", "turbulence", "envelope"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    makers = {
        "multiplicity": fig_multiplicity,
        "turbulence": fig_turbulence,
        "envelope": fig_envelope,
    }
    for name in args.figures:
        makers[name](args.out)
        print(f"wrote {name}", flush=True)


if __name__ == "__main__":
    main()
