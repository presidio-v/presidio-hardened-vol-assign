"""Figures for the Paper B audit (RQ1-RQ3). Writes PDFs to pubs/systems-turbulence/figures/audit/.

Fig. 2  fronts vs exact supported front (small instance, two 2-D projections)
Fig. 3  budget curve: committed quality and seed-floor churn vs generations (small, large)
Fig. 4  regret dose-response under input turbulence: exact decider vs crisp rule

Colours: validated categorical slots (blue/orange/aqua) with a distinct marker and line style
per series, so every figure also reads in grayscale print.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.analyze_audit import exact_supported_front, nondominated  # noqa: E402
from experiments.generate_instances import SIZES  # noqa: E402
from experiments.run_turbulence import OBJECTIVES, _config  # noqa: E402
from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs  # noqa: E402
from presidio_vol_assign.allocation.fast_fis import precompute_fis_cache_fast  # noqa: E402
from presidio_vol_assign.allocation.solvers import evaluate_pairs  # noqa: E402
from presidio_vol_assign.allocation.validation import load_allocation_problem  # noqa: E402

OUT = Path("pubs/systems-turbulence/figures/audit")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
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
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
    }
)


def _problem(size: str):  # noqa: ANN202
    base = Path("experiments/instances") / size
    return load_allocation_problem(
        base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=SIZES[size].n_dir
    )


def _front(run: str, seed_index: int) -> np.ndarray:
    lines = Path(f"experiments/results/seed_floor/{run}/fronts.jsonl").read_text().splitlines()
    return nondominated(
        np.array([s["fitness"] for s in json.loads(lines[seed_index])["solutions"]])
    )


def fig_fronts() -> None:
    problem = _problem("small")
    cache = precompute_fis_cache_fast(problem, _config(100, 150))
    exact = exact_supported_front(cache, problem.n_dir)
    opt = np.array(
        evaluate_pairs(exact_weighted_sum_pairs(cache, problem.n_dir, OBJECTIVES), cache, 3)
    )
    g150, g2400 = _front("small_gen150", 0), _front("small_gen2400", 0)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), constrained_layout=True)
    for ax, (j, label) in zip(axes, [(1, "TIL"), (2, "CAIL")], strict=True):
        ax.scatter(
            exact[:, 0], exact[:, j], s=10, marker="o", color=BLUE, label="Exact supported front"
        )
        ax.scatter(
            g2400[:, 0],
            g2400[:, j],
            s=12,
            marker="^",
            facecolor="none",
            edgecolor=AQUA,
            linewidth=0.8,
            label="NSGA-II, 2400 generations",
        )
        ax.scatter(
            g150[:, 0],
            g150[:, j],
            s=12,
            marker="s",
            facecolor="none",
            edgecolor=ORANGE,
            linewidth=0.8,
            label="NSGA-II, 150 generations",
        )
        ax.scatter(
            [opt[0]],
            [opt[j]],
            s=70,
            marker="*",
            color=INK,
            zorder=5,
            label="Exact equal-weight optimum",
        )
        ax.set_xlabel("ULPP (lower is better)")
        ax.set_ylabel(f"{label} (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7)
    fig.savefig(OUT / "fig2_fronts_vs_exact.pdf")
    plt.close(fig)


def fig_budget() -> None:
    rows = list(csv.DictReader(Path("experiments/results/audit/audit_summary.csv").open()))
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.2), constrained_layout=True, sharex=True)
    for col, size in enumerate(("small", "large")):
        pts = sorted(
            (int(r["generations"]), r)
            for r in rows
            if r["size"] == size
            and r["solver"] == "nsga2"
            and r["pop_size"] == "100"
            and r["instance"] == "reference"
        )
        gens = np.array([g for g, _ in pts])
        q = np.array([float(r["committed_q_mean"]) for _, r in pts])
        qsd = np.array([float(r["committed_q_sd"]) for _, r in pts])
        churn = np.array([float(r["churn_floor"]) for _, r in pts])
        exact = float(pts[0][1]["exact_sum"])
        ax = axes[0, col]
        ax.errorbar(
            gens,
            q,
            yerr=qsd,
            color=ORANGE,
            marker="s",
            markersize=4,
            capsize=2,
            label="NSGA-II committed decision (mean ± s.d.)",
        )
        ax.axhline(exact, color=BLUE, linestyle="--", linewidth=1.2, label="Exact optimum")
        ax.set_title(f"{size.capitalize()} instance", fontsize=8, color=INK)
        ax.set_ylabel("Clean quality, sum of objectives")
        ax2 = axes[1, col]
        ax2.plot(
            gens, churn, color=ORANGE, marker="s", markersize=4, label="NSGA-II, different seeds"
        )
        ax2.axhline(0.0, color=BLUE, linestyle="--", linewidth=1.2, label="Exact decider")
        ax2.set_ylim(-0.03, 1.0)
        ax2.set_ylabel("Churn between seeds")
        ax2.set_xscale("log")
        ax2.set_xlim(gens.min() / 1.3, gens.max() * 1.3)
        ax2.set_xticks(gens)
        ax2.set_xticklabels([str(g) for g in gens], rotation=45)
        ax2.minorticks_off()
        ax2.set_xlabel("Generations (population 100)")
    axes[0, 1].legend(fontsize=7, loc="center right")
    axes[1, 1].legend(fontsize=7, loc="center right")
    fig.savefig(OUT / "fig3_budget_curve.pdf")
    plt.close(fig)


CELLS = [
    ("travel_duration_noise", "Travel duration, noise"),
    ("resource_time_remaining_noise", "Resource time, noise"),
    ("infrastructure_damage_level_noise", "Damage level, noise"),
    ("center_occupancy_rate_noise", "Centre occupancy, noise"),
    ("road_condition_flip", "Road condition, flip"),
    ("possible_hazard_flip", "Possible hazard, flip"),
    ("infrastructure_damage_level_missingness", "Damage level, missing"),
    ("center_occupancy_rate_missingness", "Centre occupancy, missing"),
]


def fig_turbulence() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.6), constrained_layout=True, sharex=True)
    styles = {
        ("exact", "small"): (BLUE, "-", "o"),
        ("exact", "large"): (BLUE, "--", "o"),
        ("crisp", "small"): (ORANGE, "-", "s"),
        ("crisp", "large"): (ORANGE, "--", "s"),
    }
    for ax, (cell, title) in zip(axes.flat, CELLS, strict=True):
        for size in ("small", "large"):
            path = Path(
                f"experiments/results/turbulence_exact/{size}/{cell}/turbulence_manifest.csv"
            )
            vals: dict = defaultdict(list)
            for r in csv.DictReader(path.open()):
                vals[(r["system"], float(r["level"]))].append(float(r["quality_loss"]))
            for system in ("exact", "crisp"):
                levels = sorted(lv for (s, lv) in vals if s == system)
                mean = np.array([np.mean(vals[(system, lv)]) for lv in levels])
                se = np.array(
                    [
                        np.std(vals[(system, lv)], ddof=1) / np.sqrt(len(vals[(system, lv)]))
                        for lv in levels
                    ]
                )
                color, ls, mk = styles[(system, size)]
                name = "Exact decider" if system == "exact" else "Crisp rule"
                ax.errorbar(
                    levels,
                    mean,
                    yerr=1.96 * se,
                    color=color,
                    linestyle=ls,
                    marker=mk,
                    markersize=3,
                    capsize=1.5,
                    linewidth=1.1,
                    label=f"{name}, {size}",
                )
        ax.axhline(0.0, color=MUTED, linewidth=0.6)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_xticks([0, 0.1, 0.2, 0.4])
    for ax in axes[1]:
        ax.set_xlabel("Turbulence level")
    for ax in axes[:, 0]:
        ax.set_ylabel("Clean-quality loss")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7)
    fig.savefig(OUT / "fig4_turbulence_regret.pdf")
    plt.close(fig)


def _greedy_fallback(people: int, centres: int, seed: int) -> float:
    """Exact greedy of the published model, scored in the soft repaired model (kappa = 2)."""
    from presidio_vol_assign.allocation.load_coupling import precompute_load_coupled_cache

    inst = Path(f"experiments/results/scaling/instances/s{seed}/p{people}_c{centres}")
    n_dir = people // 3
    problem = load_allocation_problem(
        inst / "people.csv", inst / "centers.csv", inst / "travel.csv", n_dir=n_dir
    )
    static = precompute_fis_cache_fast(problem, _config(100, 20))
    soft = precompute_load_coupled_cache(problem, _config(100, 20), 2.0, static=static)
    pairs = exact_weighted_sum_pairs(static, n_dir, OBJECTIVES)
    return float(np.sum(evaluate_pairs(pairs, soft, OBJECTIVES)))


def fig_scaling() -> None:
    rows = []
    for name in ("scaling_manifest.csv", "scaling_manifest_3000_t60.csv"):
        path = Path("experiments/results/scaling") / name
        if path.exists():
            rows += [r for r in csv.DictReader(path.open()) if r["method"] != "greedy_static"]
    sizes = sorted({(int(r["people"]), int(r["centres"])) for r in rows})
    seeds = sorted({int(r["instance_seed"]) for r in rows})
    series = {
        "mip_soft": ("MIP, soft capacity (60 s limit)", BLUE, "-", "o"),
        "mip_hard": ("MIP, hard capacity (60 s limit)", BLUE, "--", "D"),
        "moea_committed": ("NSGA-II, same wall-clock", ORANGE, "-", "s"),
        "greedy": ("Published-model greedy (deterministic)", AQUA, ":", "^"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8), constrained_layout=True)
    x = np.array([p for p, _ in sizes])
    for key, (label, color, ls, mk) in series.items():
        if key == "mip_hard":
            continue  # scored under a different capacity model; timing panel only
        means, lows, highs = [], [], []
        for people, centres in sizes:
            if key == "greedy":
                vals = [_greedy_fallback(people, centres, s) for s in seeds]
            else:
                vals = [
                    float(r["quality"])
                    for r in rows
                    if (int(r["people"]), int(r["centres"])) == (people, centres)
                    and r["method"] == key
                    and r["quality"]
                ]
            means.append(np.mean(vals) if vals else np.nan)
            lows.append(np.min(vals) if vals else np.nan)
            highs.append(np.max(vals) if vals else np.nan)
        means, lows, highs = map(np.array, (means, lows, highs))
        axes[0].errorbar(
            x,
            means,
            yerr=[means - lows, highs - means],
            color=color,
            linestyle=ls,
            marker=mk,
            markersize=4,
            capsize=2,
            label=label,
        )
    axes[0].set_xscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{p}\n×{c}" for p, c in sizes], fontsize=6.5)
    axes[0].minorticks_off()
    axes[0].set_xlabel("People × centres")
    axes[0].set_ylabel("Quality, soft model")
    for key, (label, color, ls, mk) in list(series.items())[:2]:
        t = []
        for people, centres in sizes:
            sel = [
                r
                for r in rows
                if (int(r["people"]), int(r["centres"])) == (people, centres) and r["method"] == key
            ]
            t.append(np.mean([float(r["seconds"]) for r in sel]) if sel else np.nan)
        proven = [
            all(
                r["optimal"] == "True"
                for r in rows
                if (int(r["people"]), int(r["centres"])) == sz and r["method"] == key
            )
            for sz in sizes
        ]
        axes[1].plot(x, t, color=color, linestyle=ls, marker=mk, markersize=4, label=label)
        unproven = [i for i, ok in enumerate(proven) if not ok]
        axes[1].scatter(
            x[unproven],
            np.array(t)[unproven],
            s=60,
            facecolor="none",
            edgecolor=INK,
            linewidth=0.8,
            zorder=5,
        )
    axes[1].axhline(60, color=MUTED, linewidth=0.8, linestyle=":")
    axes[1].text(x[0], 63, "time limit", fontsize=6.5, color=MUTED)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{p}\n×{c}" for p, c in sizes], fontsize=6.5)
    axes[1].minorticks_off()
    axes[1].set_xlabel("People × centres")
    axes[1].set_ylabel("MIP wall-clock (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    for h, lab in zip(h1, l1, strict=True):
        if lab not in labels:
            handles.append(h)
            labels.append(lab)
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7)
    fig.savefig(OUT / "fig5_scaling.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import sys

    wanted = set(sys.argv[1:]) or {"fronts", "budget", "turbulence", "scaling"}
    for name, fn in (
        ("fronts", fig_fronts),
        ("budget", fig_budget),
        ("turbulence", fig_turbulence),
        ("scaling", fig_scaling),
    ):
        if name in wanted:
            fn()


if __name__ == "__main__":
    main()
