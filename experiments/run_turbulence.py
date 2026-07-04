"""RQ1 input-turbulence sweep: fuzzy-MOEA vs crisp decision stability (Paper B).

For each (field, mode, level) and each turbulence realisation, perturb the clean
instance, re-decide with both the fuzzy-MOEA solver and the crisp greedy baseline,
score each decision on the CLEAN ground truth, and record how far it drifted from
the clean-input decision. Writes a manifest CSV for the degradation analysis.

The 3-objective relief core is used throughout (Paper B), distinct from Paper A's
4-objective / NSGA-III study.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.generate_instances import SIZES
from experiments.run_h1_h2_h4 import BASE_SEED, SEED_STEP
from presidio_vol_assign.allocation.baselines import crisp_greedy_pairs
from presidio_vol_assign.allocation.decisions import (
    canonical_decision,
    decision_stability,
    pairs_of,
)
from presidio_vol_assign.allocation.models import (
    AllocationConfig,
    AllocationSolverType,
    Weights,
)
from presidio_vol_assign.allocation.solvers import precompute_fis_cache, solve
from presidio_vol_assign.allocation.turbulence import (
    PerturbationSpec,
    TurbulenceMode,
    apply_turbulence,
)
from presidio_vol_assign.allocation.validation import load_allocation_problem

OBJECTIVES = 3  # Paper B: the 3-objective relief core

_FIELDS = (
    "objective_drift",
    "quality_loss",
    "allocation_churn",
    "load_rank_stability",
)


def _config(pop_size: int, generations: int, seed: int | None = None) -> AllocationConfig:
    return AllocationConfig(
        solver=AllocationSolverType.NSGA2,
        objectives=OBJECTIVES,
        weights=Weights(),
        pop_size=pop_size,
        generations=generations,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="small", choices=sorted(SIZES.keys()))
    parser.add_argument("--field", default="infrastructure_damage_level")
    parser.add_argument("--mode", default="noise", choices=[m.value for m in TurbulenceMode])
    parser.add_argument("--levels", default="0.0,0.1,0.2,0.4")
    parser.add_argument("--reps", type=int, default=5, help="fuzzy solver seeds")
    parser.add_argument("--realizations", type=int, default=10, help="turbulence draws per level")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("experiments/results/turbulence"))
    args = parser.parse_args()

    levels = [float(x) for x in args.levels.split(",")]
    spec_size = SIZES[args.size]
    base = Path("experiments/instances") / args.size
    problem = load_allocation_problem(
        base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=spec_size.n_dir
    )
    n_centers = problem.n_centers

    base_cfg = _config(args.pop_size, args.generations)
    clean_cache = precompute_fis_cache(problem, base_cfg)

    # Clean reference decisions (fuzzy: one per solver seed; crisp: deterministic).
    clean_fuzzy: dict[int, list[tuple[int, int]]] = {}
    for rep in range(args.reps):
        cfg = _config(args.pop_size, args.generations, seed=BASE_SEED + rep * SEED_STEP)
        clean_fuzzy[rep] = pairs_of(
            canonical_decision(solve(problem, cfg, cache=clean_cache)), problem
        )
    clean_crisp = crisp_greedy_pairs(problem, base_cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "turbulence_manifest.csv"
    header = ["field", "mode", "level", "realization", "system", "rep", *_FIELDS]

    print(
        f"RQ1 turbulence: {args.field}/{args.mode} levels={levels} × "
        f"{args.realizations} realisations, fuzzy {args.reps} reps + crisp, "
        f"{args.size} 3-obj NSGA-II",
        flush=True,
    )

    with manifest.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for level in levels:
            for r in range(args.realizations):
                rng = np.random.default_rng(BASE_SEED + r)
                spec = PerturbationSpec(args.field, TurbulenceMode(args.mode), level)
                perturbed = apply_turbulence(problem, spec, rng)
                pert_cache = precompute_fis_cache(perturbed, base_cfg)

                crisp_pert = crisp_greedy_pairs(perturbed, base_cfg)
                m = decision_stability(clean_crisp, crisp_pert, clean_cache, OBJECTIVES, n_centers)
                writer.writerow(
                    [args.field, args.mode, level, r, "crisp", -1, *[m[k] for k in _FIELDS]]
                )

                for rep in range(args.reps):
                    cfg = _config(args.pop_size, args.generations, seed=BASE_SEED + rep * SEED_STEP)
                    fuzzy_pert = pairs_of(
                        canonical_decision(solve(perturbed, cfg, cache=pert_cache)), perturbed
                    )
                    m = decision_stability(
                        clean_fuzzy[rep], fuzzy_pert, clean_cache, OBJECTIVES, n_centers
                    )
                    writer.writerow(
                        [args.field, args.mode, level, r, "fuzzy", rep, *[m[k] for k in _FIELDS]]
                    )
            print(f"  level {level}: done", flush=True)

    print(f"wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
