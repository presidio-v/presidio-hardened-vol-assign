"""RQ1 turbulence with the exact deterministic decider (Paper B audit).

The static three-objective model is separable, so ``exact_weighted_sum_pairs`` returns the
exact equal-weight optimum with no solver stochasticity. Running it through the same
perturbations as ``run_turbulence.py`` (realisation r uses ``default_rng(BASE_SEED + r)``,
so the first realisations are identical to the MOEA runs) isolates the *pure input
sensitivity* of the objective landscape: any churn or quality loss here is caused by the
degraded inputs alone. The crisp baseline is re-recorded alongside for a like-for-like table.

Writes ``<out>/<field>_<mode>/turbulence_manifest.csv`` in the same schema as the MOEA driver
(``rep`` is -1 for both deterministic systems).

Usage::

    python -m experiments.run_turbulence_exact --size small --realizations 50
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.generate_instances import SIZES
from experiments.run_h1_h2_h4 import BASE_SEED
from experiments.run_turbulence import OBJECTIVES, _config
from presidio_vol_assign.allocation.baselines import crisp_greedy_pairs, exact_weighted_sum_pairs
from presidio_vol_assign.allocation.decisions import decision_stability
from presidio_vol_assign.allocation.fast_fis import precompute_fis_cache_fast
from presidio_vol_assign.allocation.turbulence import (
    PerturbationSpec,
    TurbulenceMode,
    apply_turbulence,
)
from presidio_vol_assign.allocation.validation import load_allocation_problem

CELLS = (
    ("infrastructure_damage_level", "noise"),
    ("resource_time_remaining", "noise"),
    ("center_occupancy_rate", "noise"),
    ("travel_duration", "noise"),
    ("infrastructure_damage_level", "missingness"),
    ("center_occupancy_rate", "missingness"),
    ("road_condition", "flip"),
    ("possible_hazard", "flip"),
)
_FIELDS = ["objective_drift", "quality_loss", "allocation_churn", "load_rank_stability"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="small", choices=sorted(SIZES.keys()))
    parser.add_argument("--levels", default="0.0,0.05,0.1,0.2,0.4")
    parser.add_argument("--realizations", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cell", default=None, help="one cell only, e.g. travel_duration_noise")
    args = parser.parse_args()
    cells = [c for c in CELLS if args.cell in (None, f"{c[0]}_{c[1]}")]
    if not cells:
        parser.error(f"unknown --cell {args.cell!r}")

    levels = [float(x) for x in args.levels.split(",")]
    out = args.out or Path("experiments/results/turbulence_exact") / args.size
    base = Path("experiments/instances") / args.size
    problem = load_allocation_problem(
        base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=SIZES[args.size].n_dir
    )
    cfg = _config(100, 150)  # solver budget is irrelevant here; only FIS settings are used
    clean_cache = precompute_fis_cache_fast(problem, cfg)
    n_dir, n_centers = problem.n_dir, problem.n_centers
    clean = {
        "exact": exact_weighted_sum_pairs(clean_cache, n_dir, OBJECTIVES),
        "crisp": crisp_greedy_pairs(problem, cfg),
    }

    header = ["field", "mode", "level", "realization", "system", "rep", *_FIELDS]
    for field, mode in cells:
        cell_dir = out / f"{field}_{mode}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        with (cell_dir / "turbulence_manifest.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for level in levels:
                for r in range(args.realizations):
                    rng = np.random.default_rng(BASE_SEED + r)
                    spec = PerturbationSpec(field, TurbulenceMode(mode), level)
                    perturbed = apply_turbulence(problem, spec, rng)
                    # level 0 is the identity check: the clean cache is exact for it
                    pert_cache = (
                        clean_cache if level == 0.0 else precompute_fis_cache_fast(perturbed, cfg)
                    )
                    decided = {
                        "exact": exact_weighted_sum_pairs(pert_cache, n_dir, OBJECTIVES),
                        "crisp": crisp_greedy_pairs(perturbed, cfg),
                    }
                    for system, pairs in decided.items():
                        m = decision_stability(
                            clean[system], pairs, clean_cache, OBJECTIVES, n_centers
                        )
                        writer.writerow(
                            [field, mode, level, r, system, -1, *[m[k] for k in _FIELDS]]
                        )
        print(f"  {args.size} {field}/{mode}: done", flush=True)


if __name__ == "__main__":
    main()
