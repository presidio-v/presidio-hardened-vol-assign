"""Figures for the RQ1 turbulence study (Paper B): fuzzy-vs-crisp fragility.

Reads every ``*/turbulence_manifest.csv`` under a results directory and renders:
  - one degradation figure per (field, mode) cell — objective drift and allocation
    churn vs turbulence level, fuzzy-MOEA vs crisp, mean with ±1 s.d. band;
  - one summary bar chart of allocation churn (fuzzy vs crisp) at a chosen level.

Output PDFs land in the figures directory (git-ignored, per Paper A convention).
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

_SYSTEMS = ("fuzzy", "crisp")
_COLORS = {"fuzzy": "#c1272d", "crisp": "#0000a7"}


def _agg(
    rows: list[dict], system: str, metric: str
) -> tuple[list[float], list[float], list[float]]:
    """Return (levels, mean, std) for one system/metric, sorted by level."""
    by_level: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if r["system"] == system:
            by_level[float(r["level"])].append(float(r[metric]))
    levels = sorted(by_level)
    mean = [float(np.mean(by_level[x])) for x in levels]
    std = [float(np.std(by_level[x])) for x in levels]
    return levels, mean, std


def _degradation_figure(rows: list[dict], field: str, mode: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, metric, title in (
        (axes[0], "objective_drift", "Realised-objective drift"),
        (axes[1], "allocation_churn", "Allocation churn"),
    ):
        for system in _SYSTEMS:
            levels, mean, std = _agg(rows, system, metric)
            mean_a, std_a = np.array(mean), np.array(std)
            ax.plot(levels, mean, marker="o", label=system, color=_COLORS[system])
            ax.fill_between(
                levels, mean_a - std_a, mean_a + std_a, alpha=0.15, color=_COLORS[system]
            )
        ax.set_xlabel("turbulence level")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("distance on clean objectives")
    axes[1].set_ylabel("fraction re-routed")
    axes[1].legend()
    fig.suptitle(f"{field} / {mode}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _churn_summary(cells: dict[tuple[str, str], list[dict]], level: float, out: Path) -> None:
    labels, fuzzy_v, crisp_v = [], [], []
    for (field, mode), rows in sorted(cells.items()):
        labels.append(f"{field[:12]}\n{mode}")
        for system, bucket in (("fuzzy", fuzzy_v), ("crisp", crisp_v)):
            vals = [
                float(r["allocation_churn"])
                for r in rows
                if r["system"] == system and float(r["level"]) == level
            ]
            bucket.append(float(np.mean(vals)) if vals else 0.0)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 3.8))
    ax.bar(x - 0.2, fuzzy_v, 0.4, label="fuzzy", color=_COLORS["fuzzy"])
    ax.bar(x + 0.2, crisp_v, 0.4, label="crisp", color=_COLORS["crisp"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("allocation churn")
    ax.set_title(f"Allocation churn at turbulence level {level} (fuzzy vs crisp)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--figures", type=Path, default=Path("pubs/systems-turbulence/figures"))
    parser.add_argument("--churn-level", type=float, default=0.2)
    args = parser.parse_args()

    manifests = sorted(args.results_dir.glob("*/turbulence_manifest.csv"))
    if not manifests:
        raise SystemExit(f"no turbulence_manifest.csv under {args.results_dir}")

    args.figures.mkdir(parents=True, exist_ok=True)
    cells: dict[tuple[str, str], list[dict]] = {}
    for manifest in manifests:
        rows = list(csv.DictReader(manifest.open()))
        field, mode = rows[0]["field"], rows[0]["mode"]
        cells[(field, mode)] = rows
        out = args.figures / f"fig_degradation_{field}_{mode}.pdf"
        _degradation_figure(rows, field, mode, out)
        print(f"wrote {out}")

    summary = args.figures / "fig_churn_summary.pdf"
    _churn_summary(cells, args.churn_level, summary)
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
