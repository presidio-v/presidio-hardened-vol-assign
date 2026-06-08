"""Manuscript figures from the v0.2.0 experimental data.

Produces five PDF figures into ``pubs/v0.2.0-mdpi/figures/``:

    fig5_pareto_fronts.pdf       — 4-obj Pareto fronts, parallel coords per size
    fig6_h1_spearman.pdf         — H1 Spearman ρ + 3-obj-projection dominance
    fig7_h2_algorithm_compare.pdf — H2: HV + CPU per (size, algorithm)
    fig8_h3a_rule_sensitivity.pdf — H3a: ΔHV distribution per FIS
    fig9_h3b_weight_sensitivity.pdf — H3b: per-objective stability under LHS

Captions are intentionally minimal here; the manuscript adds prose
captions per the figure-budget table in book-implementation-plan.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
    }
)

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "pubs" / "v0.2.0-mdpi" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

ALGO_COLOR = {
    "nsga2": "#1f77b4",
    "nrga": "#ff7f0e",
    "nsga3": "#2ca02c",
}
ALGO_LABEL = {"nsga2": "NSGA-II", "nrga": "NRGA", "nsga3": "NSGA-III"}
SIZE_ORDER = ["small", "medium", "large"]


# ---------------------------------------------------------------------------
# Fig 5: Pareto fronts per size, 4-obj, parallel coordinates
# ---------------------------------------------------------------------------


def fig5_pareto_fronts() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    obj_cols = ["mn_ulpp", "mn_trd", "mn_rpd", "mn_cail"]
    obj_labels = [
        r"$\overline{\mathrm{ULPP}}$",
        r"$\overline{\mathrm{TRD}}$",
        r"$\overline{\mathrm{RPD}}$",
        r"$\overline{\mathrm{CAIL}}$",
    ]

    manifest = pd.read_csv(RESULTS / "h1_h2_h4" / "manifest.csv")
    manifest = manifest[(manifest["objectives"] == 4) & (manifest["rep"] == 0)]

    for ax, size in zip(axes, SIZE_ORDER):
        for _, row in manifest[manifest["size"] == size].iterrows():
            run_dir = RESULTS / "h1_h2_h4" / row["run_id"]
            csv_path = next(run_dir.glob("pareto_alloc_*.csv"))
            df = pd.read_csv(csv_path)
            xs = list(range(len(obj_cols)))
            for _, sol in df.iterrows():
                ax.plot(
                    xs,
                    [sol[c] for c in obj_cols],
                    color=ALGO_COLOR[row["algorithm"]],
                    alpha=0.10,
                    linewidth=0.7,
                )
        ax.set_xticks(range(len(obj_cols)))
        ax.set_xticklabels(obj_labels)
        ax.set_title(f"{size.capitalize()} (rep 0)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Objective value (lower is better)")
    handles = [
        plt.Line2D([0], [0], color=c, lw=2, label=ALGO_LABEL[a]) for a, c in ALGO_COLOR.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES / "fig5_pareto_fronts.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 6: H1 Spearman + projection dominance
# ---------------------------------------------------------------------------


def fig6_h1_spearman() -> None:
    df = pd.read_csv(RESULTS / "h1_h2_h4" / "h1_analysis.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))

    # Left: histogram of Spearman ρ
    ax = axes[0]
    for algo in ["nsga2", "nrga", "nsga3"]:
        sub = df[df["algorithm"] == algo]["spearman_rho"].dropna()
        ax.hist(
            sub,
            bins=20,
            alpha=0.55,
            label=ALGO_LABEL[algo],
            color=ALGO_COLOR[algo],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.axvline(-0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.text(
        -0.5,
        ax.get_ylim()[1] * 0.95,
        " H1 threshold ",
        ha="right",
        va="top",
        fontsize=7,
        color="grey",
    )
    ax.set_xlabel(r"Spearman $\rho(\mathrm{TRD},\mathrm{RPD})$")
    ax.set_ylabel("Number of runs")
    ax.set_title(r"(a) Spearman $\rho$ across 4-obj fronts")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Right: fraction dominated when projected to 3-obj
    ax = axes[1]
    pivot = (
        df.groupby(["size", "algorithm"])["fraction_dominated"]
        .mean()
        .unstack("algorithm")
        .reindex(SIZE_ORDER)
    )
    x = np.arange(len(SIZE_ORDER))
    width = 0.27
    for i, algo in enumerate(["nsga2", "nrga", "nsga3"]):
        ax.bar(
            x + (i - 1) * width,
            pivot[algo],
            width=width,
            label=ALGO_LABEL[algo],
            color=ALGO_COLOR[algo],
        )
    ax.axhline(0.20, color="grey", linestyle="--", linewidth=0.8, label="H1 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SIZE_ORDER])
    ax.set_ylabel("Fraction dominated in 3-obj projection")
    ax.set_title("(b) Mean per cell")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_h1_spearman.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 7: H2 algorithm comparison (HV + CPU) on 4-obj
# ---------------------------------------------------------------------------


def fig7_h2_algorithm_compare() -> None:
    df = pd.read_csv(RESULTS / "h1_h2_h4" / "manifest.csv")
    df = df[df["objectives"] == 4]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))

    # HV
    ax = axes[0]
    for i, algo in enumerate(["nsga2", "nrga", "nsga3"]):
        sub = df[df["algorithm"] == algo]
        means = [sub[sub["size"] == s]["hv"].mean() / 1e6 for s in SIZE_ORDER]
        stds = [sub[sub["size"] == s]["hv"].std() / 1e6 for s in SIZE_ORDER]
        x = np.arange(len(SIZE_ORDER)) + (i - 1) * 0.25
        ax.bar(
            x,
            means,
            width=0.23,
            yerr=stds,
            label=ALGO_LABEL[algo],
            color=ALGO_COLOR[algo],
            capsize=2,
        )
    ax.set_xticks(np.arange(len(SIZE_ORDER)))
    ax.set_xticklabels([s.capitalize() for s in SIZE_ORDER])
    ax.set_ylabel("Hypervolume (millions)")
    ax.set_title("(a) HV — higher is better")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # CPU
    ax = axes[1]
    for i, algo in enumerate(["nsga2", "nrga", "nsga3"]):
        sub = df[df["algorithm"] == algo]
        means = [sub[sub["size"] == s]["cpu_time_sec"].mean() for s in SIZE_ORDER]
        stds = [sub[sub["size"] == s]["cpu_time_sec"].std() for s in SIZE_ORDER]
        x = np.arange(len(SIZE_ORDER)) + (i - 1) * 0.25
        ax.bar(
            x,
            means,
            width=0.23,
            yerr=stds,
            label=ALGO_LABEL[algo],
            color=ALGO_COLOR[algo],
            capsize=2,
        )
    ax.set_xticks(np.arange(len(SIZE_ORDER)))
    ax.set_xticklabels([s.capitalize() for s in SIZE_ORDER])
    ax.set_ylabel("CPU time (seconds)")
    ax.set_title("(b) CPU — lower is better")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_h2_algorithm_compare.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 8: H3a rule-base ΔHV per FIS
# ---------------------------------------------------------------------------


def fig8_h3a_rule_sensitivity() -> None:
    df = pd.read_csv(RESULTS / "h3a" / "h3a_manifest.csv")
    df = df[df["fis_name"] != "baseline"]
    fis_order = ["fis1", "fis2a_trd", "fis2b_rpd", "fis3"]
    fis_label = {
        "fis1": "FIS₁ ULPP\n(27 rules)",
        "fis2a_trd": "FIS₂ₐ TRD\n(9 rules)",
        "fis2b_rpd": "FIS₂ᵦ RPD\n(3 rules)",
        "fis3": "FIS₃ CAIL\n(27 rules)",
    }

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    data = [df[df["fis_name"] == name]["delta_hv_pct"].values for name in fis_order]
    parts = ax.violinplot(
        data, positions=range(len(fis_order)), showmeans=True, showmedians=True, widths=0.7
    )
    for pc in parts["bodies"]:
        pc.set_alpha(0.55)
        pc.set_edgecolor("black")
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color("black")
        parts[key].set_linewidth(0.8)

    ax.axhline(0, color="grey", linestyle="-", linewidth=0.5)
    ax.axhline(5, color="red", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.axhline(
        -5,
        color="red",
        linestyle="--",
        linewidth=0.7,
        alpha=0.6,
        label="±5% (H3a confirmation band)",
    )
    ax.set_xticks(range(len(fis_order)))
    ax.set_xticklabels([fis_label[n] for n in fis_order])
    ax.set_ylabel(r"$\Delta$HV (%) vs. baseline")
    ax.set_title("Single-rule deletion: ΔHV distribution per FIS (medium, NSGA-II, 30 reps)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_h3a_rule_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 9: H3b weight LHS — per-objective stability
# ---------------------------------------------------------------------------


def fig9_h3b_weight_sensitivity() -> None:
    df = pd.read_csv(RESULTS / "h3b" / "h3b_manifest.csv")

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), sharex=False)
    objs = [
        ("mn_ulpp_avg", r"$\overline{\mathrm{ULPP}}$"),
        ("mn_trd_avg", r"$\overline{\mathrm{TRD}}$"),
        ("mn_rpd_avg", r"$\overline{\mathrm{RPD}}$"),
        ("mn_cail_avg", r"$\overline{\mathrm{CAIL}}$"),
    ]
    for ax, (col, label) in zip(axes, objs):
        vals = df[col]
        cv = 100 * vals.std() / vals.mean()
        ax.hist(vals, bins=18, alpha=0.85, color="#4c72b0", edgecolor="white", linewidth=0.4)
        ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=0.7)
        ax.set_xlabel(label)
        ax.set_title(f"CV = {cv:.2f}%", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    axes[0].set_ylabel("Number of LHS samples")
    fig.suptitle(
        "Mean objective values across 100 LHS samples of weights ∈ baseline·(1±20%)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig9_h3b_weight_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"Writing figures to {FIGURES}")
    for fn in (
        fig5_pareto_fronts,
        fig6_h1_spearman,
        fig7_h2_algorithm_compare,
        fig8_h3a_rule_sensitivity,
        fig9_h3b_weight_sensitivity,
    ):
        fn()
        print(f"  ✓ {fn.__name__}")


if __name__ == "__main__":
    main()
