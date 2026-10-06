"""RQ4 repair sweep (Paper B): the load-coupled model solved exactly across capacity settings.

For each reference instance and capacity variant (soft: size factor kappa, inf = published
model; hard: free-capacity factor kappa_f), solves the equal-weight MIP and re-scores the
published model's exact optimum in the repaired model. Writes
``experiments/results/repair/kappa_sweep.csv``.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.generate_instances import SIZES
from experiments.run_turbulence import OBJECTIVES, _config
from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs
from presidio_vol_assign.allocation.exact_mip import solve_weighted_mip
from presidio_vol_assign.allocation.fast_fis import precompute_fis_cache_fast
from presidio_vol_assign.allocation.load_coupling import (
    capacities_from_free_factor,
    hard_load_limits,
    overload,
    precompute_load_coupled_cache,
)
from presidio_vol_assign.allocation.solvers import evaluate_pairs
from presidio_vol_assign.allocation.validation import load_allocation_problem

OUT = Path("experiments/results/repair")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in ("small", "large"):
        base = Path("experiments/instances") / size
        problem = load_allocation_problem(
            base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=SIZES[size].n_dir
        )
        cfg = _config(100, 150)
        static = precompute_fis_cache_fast(problem, cfg)
        published = exact_weighted_sum_pairs(static, problem.n_dir, OBJECTIVES)
        variants = [("soft", k, k, None) for k in (math.inf, 1.0, 2.0, 3.0)]
        for kf in (1.2, 1.5, 2.0):
            cap = capacities_from_free_factor(problem, kf)
            variants.append(("hard", kf, cap, hard_load_limits(problem, cap)))
        for kind, factor, capacity, limits in variants:
            cache = precompute_load_coupled_cache(problem, cfg, capacity, static=static)
            r = solve_weighted_mip(
                cache, problem.n_dir, OBJECTIVES, time_limit=120, max_load=limits
            )
            loads = Counter(c for _, c in r.pairs)
            row = {
                "size": size,
                "capacity": kind,
                "factor": factor,
                "mip_quality": float(np.sum(evaluate_pairs(r.pairs, cache, OBJECTIVES))),
                "mip_max_load": max(loads.values()),
                "mip_overload": overload(r.pairs, problem, capacity),
                "mip_seconds": r.seconds,
                "mip_optimal": r.optimal,
                "mip_gap": r.gap,
                "published_quality": float(np.sum(evaluate_pairs(published, cache, OBJECTIVES))),
                "published_overload": overload(published, problem, capacity),
                "overlap": len(set(r.pairs) & set(published)) / problem.n_dir,
            }
            rows.append(row)
            print(row, flush=True)
    with (OUT / "kappa_sweep.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
