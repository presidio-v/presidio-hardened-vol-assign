"""Sensitivity of the repair's form (Paper B revision, R2.7).

The repair of the paper adds a centre's load to its occupancy rate as a share of its
capacity, COR'_i = min(100, COR_i + 100 l / cap_i), which is the definition of an
occupancy rate. The finding that CAIL has too little dynamic range to spread load could
depend on that form, so the RQ4 capacity grid is re-solved exactly under

* ``linear``: the paper's form (must reproduce ``repair/kappa_sweep.csv``);
* ``amplified``: alpha = 2, the load counts double;
* ``convex``: gamma = 2, light loads matter little, heavy loads a lot;
* ``concave``: gamma = 0.5, the first arrivals already raise occupancy sharply;
* ``stretched``: the linear form with the rule base's response to load tripled,
  CAIL(l) = CAIL(0) + 3 (CAIL(l) - CAIL(0)), clipped to [0, 100]. This changes the rule
  base, not the repair, and tests the dynamic-range explanation directly.

Writes ``experiments/results/repair/repair_forms.csv``.
"""

from __future__ import annotations

import csv
import dataclasses
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
FORMS = {
    "linear": (1.0, 1.0, 1.0),
    "amplified": (2.0, 1.0, 1.0),
    "convex": (1.0, 2.0, 1.0),
    "concave": (1.0, 0.5, 1.0),
    "stretched": (1.0, 1.0, 3.0),
}  # name -> (alpha, gamma, rule-base stretch)


def _stretch(cache, factor: float):  # noqa: ANN001, ANN202
    if factor == 1.0:
        return cache
    base = cache.cail_load[:, :, :1]
    table = np.clip(base + factor * (cache.cail_load - base), 0.0, 100.0)
    return dataclasses.replace(cache, cail_load=table)


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
        variants = [("soft", k, k, None) for k in (1.0, 2.0, 3.0)]
        for kf in (1.2, 1.5, 2.0):
            cap = capacities_from_free_factor(problem, kf)
            variants.append(("hard", kf, cap, hard_load_limits(problem, cap)))
        for form, (alpha, gamma, stretch) in FORMS.items():
            for kind, factor, capacity, limits in variants:
                cache = precompute_load_coupled_cache(
                    problem, cfg, capacity, static=static, alpha=alpha, gamma=gamma
                )
                cache = _stretch(cache, stretch)
                r = solve_weighted_mip(
                    cache, problem.n_dir, OBJECTIVES, time_limit=120, max_load=limits
                )
                loads = Counter(c for _, c in r.pairs)
                cail_range = float(np.mean(cache.cail_load[:, :, -1] - cache.cail_load[:, :, 0]))
                row = {
                    "size": size,
                    "form": form,
                    "capacity": kind,
                    "factor": factor,
                    "mip_quality": float(np.sum(evaluate_pairs(r.pairs, cache, OBJECTIVES))),
                    "mip_max_load": max(loads.values()),
                    "centres_used": len(loads),
                    "mip_overload": overload(r.pairs, problem, capacity),
                    "mip_seconds": r.seconds,
                    "mip_optimal": r.optimal,
                    "mip_gap": r.gap,
                    "published_quality": float(
                        np.sum(evaluate_pairs(published, cache, OBJECTIVES))
                    ),
                    "published_overload": overload(published, problem, capacity),
                    "overlap": len(set(r.pairs) & set(published)) / problem.n_dir,
                    "mean_cail_load_range": cail_range,
                }
                rows.append(row)
                print(
                    {
                        k: row[k]
                        for k in (
                            "size",
                            "form",
                            "capacity",
                            "factor",
                            "mip_quality",
                            "mip_max_load",
                            "mip_overload",
                            "published_quality",
                        )
                    },
                    flush=True,
                )
    with (OUT / "repair_forms.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
