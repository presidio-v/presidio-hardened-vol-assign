"""Analyse the RQ1 turbulence matrix (Paper B): fuzzy-vs-crisp fragility.

Reads every ``*/turbulence_manifest.csv`` under a results directory and, per
(field, mode) cell, computes each system's per-realisation degradation *slope*
(metric regressed on turbulence level, fuzzy averaged over solver reps) and a
paired Wilcoxon test of fuzzy-vs-crisp slopes. Writes a summary CSV and prints a
table. The claim (H-B1 reframed): the fuzzy-MOEA slope exceeds the crisp slope.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

_METRICS = ("objective_drift", "allocation_churn")


def _slopes_by_realization(rows: list[dict], system: str, metric: str) -> list[float]:
    """One degradation slope per realisation: mean-over-reps metric regressed on level."""
    # (realization) -> level -> [values over reps]
    by_real: dict[int, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["system"] != system:
            continue
        by_real[int(r["realization"])][float(r["level"])].append(float(r[metric]))
    slopes: list[float] = []
    for levels in by_real.values():
        xs = sorted(levels)
        ys = [float(np.mean(levels[x])) for x in xs]
        if len(xs) >= 2:
            slopes.append(float(np.polyfit(xs, ys, 1)[0]))
    return slopes


def _paired_wilcoxon_p(fuzzy: list[float], crisp: list[float]) -> float:
    n = min(len(fuzzy), len(crisp))
    if n < 1:
        return float("nan")
    diffs = np.array(fuzzy[:n]) - np.array(crisp[:n])
    if np.allclose(diffs, 0.0):
        return float("nan")  # no difference to test
    try:
        return float(wilcoxon(fuzzy[:n], crisp[:n]).pvalue)
    except ValueError:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifests = sorted(args.results_dir.glob("*/turbulence_manifest.csv"))
    if not manifests:
        raise SystemExit(f"no turbulence_manifest.csv under {args.results_dir}")

    summary: list[dict] = []
    for manifest in manifests:
        rows = list(csv.DictReader(manifest.open()))
        field = rows[0]["field"]
        mode = rows[0]["mode"]
        for metric in _METRICS:
            fuzzy = _slopes_by_realization(rows, "fuzzy", metric)
            crisp = _slopes_by_realization(rows, "crisp", metric)
            summary.append(
                {
                    "field": field,
                    "mode": mode,
                    "metric": metric,
                    "fuzzy_slope_median": round(float(np.median(fuzzy)), 5) if fuzzy else "",
                    "crisp_slope_median": round(float(np.median(crisp)), 5) if crisp else "",
                    "fuzzy_gt_crisp": bool(np.median(fuzzy) > np.median(crisp))
                    if fuzzy and crisp
                    else "",
                    "wilcoxon_p": round(_paired_wilcoxon_p(fuzzy, crisp), 5),
                    "n_realizations": len(fuzzy),
                }
            )

    cols = list(summary[0])
    print("  ".join(f"{c:>18}" for c in cols))
    for row in summary:
        print("  ".join(f"{str(row[c]):>18}" for c in cols))

    out = args.out or (args.results_dir / "turbulence_summary.csv")
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
