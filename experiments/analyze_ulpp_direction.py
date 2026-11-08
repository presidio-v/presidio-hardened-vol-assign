"""Which people does the published model prefer to direct? (Paper B revision, S1 finding.)

FIS1 assigns high ULPP to people who are highly vulnerable, live where damage is severe and
have little resource time left; the published objective minimises the mean ULPP over the
directed people. The model therefore rewards directing the people with the lowest ULPP,
although its documentation reads low ULPP as "directing the most vulnerable". This script
measures the effect on the reference instances: how ULPP relates to the inputs and to the
crisp priority score, and who the exact optimum directs compared with everyone and with the
crisp priority rule. It also solves the reading the documentation states, with ULPP
replaced by 100 - ULPP, to show how far the two readings differ.

Writes ``experiments/results/audit/ulpp_direction.csv``.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.generate_instances import SIZES
from experiments.run_turbulence import OBJECTIVES, _config
from presidio_vol_assign.allocation.baselines import (
    _crisp_priority,
    crisp_greedy_pairs,
    exact_weighted_sum_pairs,
)
from presidio_vol_assign.allocation.fast_fis import precompute_fis_cache_fast
from presidio_vol_assign.allocation.solvers import evaluate_pairs
from presidio_vol_assign.allocation.validation import load_allocation_problem


def main() -> None:
    rows = []
    for size in ("small", "large"):
        base = Path("experiments/instances") / size
        problem = load_allocation_problem(
            base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=SIZES[size].n_dir
        )
        cfg = _config(100, 150)
        cache = precompute_fis_cache_fast(problem, cfg)
        priority = np.array([_crisp_priority(p) for p in problem.people])
        damage = np.array([p.infrastructure_damage_level for p in problem.people])
        time_left = np.array([p.resource_time_remaining for p in problem.people])
        exact = exact_weighted_sum_pairs(cache, problem.n_dir, OBJECTIVES)
        crisp = crisp_greedy_pairs(problem, cfg)
        flipped_cache = dataclasses.replace(cache, ulpp=100.0 - cache.ulpp)
        flipped = exact_weighted_sum_pairs(flipped_cache, problem.n_dir, OBJECTIVES)

        def directed(pairs):  # noqa: ANN001, ANN202
            return np.array(sorted(j for j, _ in pairs))

        top = set(np.argsort(-priority, kind="stable")[: problem.n_dir])
        for name, pairs in (
            ("exact", exact),
            ("crisp_priority", crisp),
            ("exact_documented_reading", flipped),
        ):
            d = directed(pairs)
            rows.append(
                {
                    "size": size,
                    "decider": name,
                    "rho_ulpp_priority": float(spearmanr(cache.ulpp, priority)[0]),
                    "mean_priority_directed": float(priority[d].mean()),
                    "mean_priority_all": float(priority.mean()),
                    "share_of_top_priority_directed": len(top & set(d.tolist())) / problem.n_dir,
                    "mean_ulpp_directed": float(cache.ulpp[d].mean()),
                    "mean_damage_directed": float(damage[d].mean()),
                    "mean_damage_all": float(damage.mean()),
                    "mean_time_left_directed": float(time_left[d].mean()),
                    "mean_time_left_all": float(time_left.mean()),
                    "objectives_published": ";".join(
                        f"{v:.2f}" for v in evaluate_pairs(pairs, cache, OBJECTIVES)
                    ),
                    "overlap_with_exact": len(set(d.tolist()) & set(directed(exact).tolist()))
                    / problem.n_dir,
                }
            )
            print(rows[-1], flush=True)
    out = Path("experiments/results/audit")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ulpp_direction.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
