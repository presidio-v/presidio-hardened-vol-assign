"""Deployment envelope (Paper B): exact MIP vs time-matched MOEA as instances grow.

For each (people, centres) size and instance seed, on the load-coupled model:

* ``mip_soft`` — HiGHS under a wall-clock limit, capacity soft (faithful to the published
  semantics); reports objective, gap, and whether optimality was proven;
* ``mip_hard`` — the same with hard capacity from a free-capacity factor;
* ``moea`` — NSGA-II given (approximately) the same wall-clock: generations are calibrated
  from a short timed run. Both its committed (canonical) decision and its best equal-weight
  front point are scored, so the comparison cannot be lost to the committal rule.

All decisions are scored by the one shared evaluator; overload (persons beyond free capacity)
is reported for each. Writes ``<out>/scaling_manifest.csv``.

Usage::

    python -m experiments.run_scaling --time-limit 60
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from experiments.generate_instances import SizeSpec, generate_instance
from experiments.run_h1_h2_h4 import BASE_SEED
from experiments.run_turbulence import OBJECTIVES, _config
from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs
from presidio_vol_assign.allocation.decisions import canonical_decision, pairs_of
from presidio_vol_assign.allocation.exact_mip import solve_weighted_mip
from presidio_vol_assign.allocation.fast_fis import precompute_fis_cache_fast
from presidio_vol_assign.allocation.load_coupling import (
    capacities_from_free_factor,
    hard_load_limits,
    overload,
    precompute_load_coupled_cache,
)
from presidio_vol_assign.allocation.solvers import evaluate_pairs, solve
from presidio_vol_assign.allocation.validation import load_allocation_problem

# (people, centres); n_dir = people / 3 and 10 directed per centre at the reference sizes
SIZES = ((150, 5), (300, 10), (600, 20), (1000, 25), (1500, 40), (3000, 50))
KAPPA_SOFT = 2.0
KAPPA_FREE_HARD = 1.5
CALIBRATION_GENERATIONS = 20

FIELDS = [
    "people",
    "centres",
    "n_dir",
    "instance_seed",
    "method",
    "quality",
    "gap",
    "optimal",
    "seconds",
    "generations",
    "overload",
]


def _quality(pairs: list[tuple[int, int]], cache) -> float:  # noqa: ANN001
    return float(np.sum(evaluate_pairs(pairs, cache, OBJECTIVES)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-people", type=int, default=3000)
    parser.add_argument("--min-people", type=int, default=0)
    parser.add_argument("--manifest", default="scaling_manifest.csv")
    parser.add_argument("--out", type=Path, default=Path("experiments/results/scaling"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / args.manifest

    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for people, centres in SIZES:
            if not args.min_people <= people <= args.max_people:
                continue
            n_dir = people // 3
            for s in range(args.seeds):
                spec = SizeSpec(f"p{people}_c{centres}", centres, people, n_dir)
                inst = Path(
                    generate_instance(spec, seed=7 + s, out_dir=args.out / "instances" / f"s{s}")
                )
                problem = load_allocation_problem(
                    inst / "people.csv", inst / "centers.csv", inst / "travel.csv", n_dir=n_dir
                )
                cfg = _config(100, CALIBRATION_GENERATIONS, seed=BASE_SEED)
                static = precompute_fis_cache_fast(problem, cfg)
                soft = precompute_load_coupled_cache(problem, cfg, KAPPA_SOFT, static=static)
                cap_hard = capacities_from_free_factor(problem, KAPPA_FREE_HARD)
                hard = precompute_load_coupled_cache(problem, cfg, cap_hard, static=static)
                base = {"people": people, "centres": centres, "n_dir": n_dir, "instance_seed": s}

                def emit(method: str, pairs, cache, capacity, **extra) -> None:  # noqa: ANN001, ANN003
                    row = {
                        **base,
                        "method": method,
                        "quality": _quality(pairs, cache),
                        "overload": overload(pairs, problem, capacity),
                        **extra,
                    }
                    writer.writerow({k: row.get(k, "") for k in FIELDS})
                    fh.flush()
                    print(f"  {people}x{centres} s{s} {method}: {row}", flush=True)

                # deterministic fallback: the separable model's exact greedy, scored here
                greedy = exact_weighted_sum_pairs(static, n_dir, OBJECTIVES)
                emit("greedy_static", greedy, soft, KAPPA_SOFT, seconds=0.0)
                limits = hard_load_limits(problem, cap_hard)
                for method, cache, capacity, max_load in (
                    ("mip_soft", soft, KAPPA_SOFT, None),
                    ("mip_hard", hard, cap_hard, limits),
                ):
                    try:
                        r = solve_weighted_mip(
                            cache, n_dir, OBJECTIVES, time_limit=args.time_limit, max_load=max_load
                        )
                    except RuntimeError as exc:  # no incumbent within the limit: record, go on
                        failed = {**base, "method": method, "optimal": False}
                        failed["seconds"] = args.time_limit
                        writer.writerow({k: failed.get(k, "") for k in FIELDS})
                        fh.flush()
                        print(f"  {people}x{centres} s{s} {method}: no solution: {exc}")
                        continue
                    stats = {"gap": r.gap, "optimal": r.optimal, "seconds": r.seconds}
                    emit(method, r.pairs, cache, capacity, **stats)

                # MOEA on the soft model, generations calibrated to the same wall-clock
                t0 = time.perf_counter()
                solve(problem, cfg, cache=soft)
                per_gen = (time.perf_counter() - t0) / CALIBRATION_GENERATIONS
                gens = max(CALIBRATION_GENERATIONS, int(args.time_limit / per_gen))
                t0 = time.perf_counter()
                front = solve(problem, _config(100, gens, seed=BASE_SEED), cache=soft)
                secs = time.perf_counter() - t0
                canon = pairs_of(canonical_decision(front), problem)
                emit("moea_committed", canon, soft, KAPPA_SOFT, seconds=secs, generations=gens)
                best = min(
                    (pairs_of(sol, problem) for sol in front.solutions),
                    key=lambda p: _quality(p, soft),
                )
                emit("moea_best_point", best, soft, KAPPA_SOFT, seconds=secs, generations=gens)


if __name__ == "__main__":
    main()
