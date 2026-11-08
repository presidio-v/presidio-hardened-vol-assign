"""Summary of the extended exact-decider turbulence study (Paper B revision, R2.6 / R3).

A *condition* is one (instance, demand, occupancy) triple: six instances x three demand x two
occupancy conditions = 36 per size. For every condition and cell this computes the mean regret
per level and a one-sided Spearman trend test (levels 0.05-0.4, realisations as units; bias has
one realisation per level and is tested across levels only). Holm correction runs across all
(condition, cell) tests of a size. For the joint cells it compares the regret with the sum of the
regrets of their single-field parts in the same condition, to show whether errors interact.

Writes ``experiments/results/turbulence_extended/summary.csv`` (per cell and level, across
conditions) and ``trends.csv`` (per condition and cell).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.analyze_trend_exact import holm
from experiments.run_turbulence_extended import JOINT

ROOT = Path("experiments/results/turbulence_extended")


def _load(size: str) -> dict[tuple, list[tuple[float, float, float]]]:
    """(instance, demand, occupancy, cell) -> [(level, regret, churn), ...]."""
    data: dict[tuple, list[tuple[float, float, float]]] = defaultdict(list)
    for path in sorted((ROOT / size).glob("manifest_*.csv")):
        for r in csv.DictReader(path.open()):
            key = (r["instance"], r["demand"], r["occupancy"], r["cell"])
            data[key].append((float(r["level"]), float(r["regret"]), float(r["churn"])))
    return data


def main() -> None:
    summary, trends = [], []
    for size in ("small", "large"):
        data = _load(size)
        if not data:
            continue
        conditions = sorted({k[:3] for k in data})
        size_trends = []
        for key, obs in data.items():
            levels = [o[0] for o in obs]
            regret = [o[1] for o in obs]
            if len(set(regret)) < 2:
                rho, p = 0.0, 1.0
            else:
                rho, p_two = spearmanr(levels, regret)
                p = p_two / 2 if rho > 0 else 1 - p_two / 2
            size_trends.append(
                {
                    "size": size,
                    "instance": key[0],
                    "demand": key[1],
                    "occupancy": key[2],
                    "cell": key[3],
                    "n": len(obs),
                    "rho": float(rho),
                    "p_one_sided": float(p),
                }
            )
        for row, adj in zip(
            size_trends, holm(np.array([t["p_one_sided"] for t in size_trends])), strict=True
        ):
            row["p_holm"] = float(adj)
        trends += size_trends

        cells = sorted({k[3] for k in data})
        for cell in cells:
            for level in sorted({o[0] for k, v in data.items() if k[3] == cell for o in v}):
                per_cond = [
                    np.mean([o[1] for o in data[c + (cell,)] if o[0] == level])
                    for c in conditions
                    if c + (cell,) in data
                ]
                churn = [
                    np.mean([o[2] for o in data[c + (cell,)] if o[0] == level])
                    for c in conditions
                    if c + (cell,) in data
                ]
                row = {
                    "size": size,
                    "cell": cell,
                    "level": level,
                    "conditions": len(per_cond),
                    "regret_median": float(np.median(per_cond)),
                    "regret_min": float(np.min(per_cond)),
                    "regret_max": float(np.max(per_cond)),
                    "churn_median": float(np.median(churn)),
                }
                if cell in JOINT:
                    parts = [f"{f}_{m}" for f, m in JOINT[cell]]
                    ratios = []
                    for c in conditions:
                        if c + (cell,) not in data or any(c + (p,) not in data for p in parts):
                            continue
                        joint = np.mean([o[1] for o in data[c + (cell,)] if o[0] == level])
                        singles = sum(
                            np.mean([o[1] for o in data[c + (p,)] if o[0] == level]) for p in parts
                        )
                        if singles > 0:
                            ratios.append(joint / singles)
                    row["joint_over_sum_median"] = (
                        float(np.median(ratios)) if ratios else float("nan")
                    )
                ct = [t for t in size_trends if t["cell"] == cell]
                row["share_significant"] = sum(t["p_holm"] < 0.05 for t in ct) / len(ct)
                summary.append(row)
        sig = sum(t["p_holm"] < 0.05 for t in size_trends)
        print(
            f"{size}: {len(conditions)} conditions, {len(size_trends)} trend tests, "
            f"{sig} significant after Holm",
            flush=True,
        )
    for name, rows in (("summary", summary), ("trends", trends)):
        fields = sorted(
            {k for r in rows for k in r},
            key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99,
        )
        with (ROOT / f"{name}.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
