"""Deployment envelope, revised (Paper B revision, R1.2 / R2.4 / R2.7 / R3).

Replaces ``run_scaling`` for the revision. Three changes, each answering a review point:

* Fixed instances. Every size has three instances generated once (CRC-32-salted generator)
  under ``--instances``, and every method and time limit runs on the same files; each row
  records the instance's SHA-256. ``run_scaling`` regenerated instances on every call, and
  the old generator salted with ``hash()``, so its 60 s and 300 s rows were different
  instances.
* Equal and complete time accounting. The MOEAs are stopped by a wall-clock deadline
  instead of a calibrated generation count. Shared preparation (fuzzy scores; CAIL-per-load
  tables) is timed separately and every row reports solver time and total decision latency.
* Stronger references. Capacity-aware local search (soft and hard), NSGA-II seeded with the
  deterministic fallback and exact weight-sweep optima of the published model, and overload
  for every method.

Writes one manifest per (time limit, instance seed): ``<out>/envelope_t<limit>_s<seed>.csv``,
so instance seeds can run as parallel processes.

Usage::

    python -m experiments.run_envelope --time-limit 60 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from presidio_vol_assign.allocation.local_search import capacity_greedy, local_search
from presidio_vol_assign.allocation.solvers import evaluate_pairs, solve
from presidio_vol_assign.allocation.validation import load_allocation_problem

SIZES = ((150, 5), (300, 10), (600, 20), (1000, 25), (1500, 40), (3000, 50))
KAPPA_SOFT = 2.0
KAPPA_FREE_HARD = 1.5
INSTANCE_BASE_SEED = 7  # instance seed s is generated with 7 + s, as in run_scaling
SWEEP_STEPS = 5  # 21 weight vectors on the simplex for the seeded MOEA
MAX_GENERATIONS = 10**9  # the deadline, not this bound, stops the MOEA

FIELDS = [
    "people",
    "centres",
    "n_dir",
    "instance_seed",
    "instance_sha256",
    "limit",
    "method",
    "solved",
    "quality_soft",
    "quality_hard",
    "overload_soft",
    "overload_hard",
    "gap",
    "optimal",
    "solver_s",
    "fis_s",
    "table_s",
    "total_s",
    "generations",
    "moves",
]


def _sha256(directory: Path) -> str:
    h = hashlib.sha256()
    for name in ("people.csv", "centers.csv", "travel.csv"):
        h.update((directory / name).read_bytes())
    return h.hexdigest()


def instance_dir(root: Path, people: int, centres: int, seed: int) -> Path:
    """Generate the instance once; later calls reuse the stored files."""
    spec = SizeSpec(f"p{people}_c{centres}", centres, people, people // 3)
    target = root / f"s{seed}" / spec.name
    if not (target / "travel.csv").exists():
        generate_instance(spec, seed=INSTANCE_BASE_SEED + seed, out_dir=root / f"s{seed}")
    return target


def _sweep(steps: int) -> list[tuple[float, float, float]]:
    return [
        (a / steps, b / steps, (steps - a - b) / steps)
        for a in range(steps + 1)
        for b in range(steps + 1 - a)
    ]


def run_instance(people: int, centres: int, seed: int, limit: float, root: Path, writer, fh):  # noqa: ANN001
    inst = instance_dir(root, people, centres, seed)
    n_dir = people // 3
    problem = load_allocation_problem(
        inst / "people.csv", inst / "centers.csv", inst / "travel.csv", n_dir=n_dir
    )
    cfg = _config(100, MAX_GENERATIONS, seed=BASE_SEED)
    t = time.perf_counter()
    static = precompute_fis_cache_fast(problem, cfg)
    fis_s = time.perf_counter() - t
    t = time.perf_counter()
    soft = precompute_load_coupled_cache(problem, cfg, KAPPA_SOFT, static=static)
    soft_table_s = time.perf_counter() - t
    cap_hard = capacities_from_free_factor(problem, KAPPA_FREE_HARD)
    limits = hard_load_limits(problem, cap_hard)
    t = time.perf_counter()
    hard = precompute_load_coupled_cache(problem, cfg, cap_hard, static=static)
    hard_table_s = time.perf_counter() - t
    base = {
        "people": people,
        "centres": centres,
        "n_dir": n_dir,
        "instance_seed": seed,
        "instance_sha256": _sha256(inst),
        "limit": limit,
        "fis_s": fis_s,
    }

    def emit(method: str, pairs, solver_s: float, table_s: float, **extra) -> None:  # noqa: ANN001, ANN003
        row = {**base, "method": method, "solver_s": solver_s, "table_s": table_s, **extra}
        row["total_s"] = fis_s + table_s + solver_s
        if pairs is None:
            row["solved"] = False
        else:
            row.update(
                solved=True,
                quality_soft=float(np.sum(evaluate_pairs(pairs, soft, OBJECTIVES))),
                quality_hard=float(np.sum(evaluate_pairs(pairs, hard, OBJECTIVES))),
                overload_soft=overload(pairs, problem, KAPPA_SOFT),
                overload_hard=overload(pairs, problem, cap_hard),
            )
        writer.writerow({k: row.get(k, "") for k in FIELDS})
        fh.flush()
        shown = {
            k: row.get(k) for k in ("method", "quality_soft", "quality_hard", "solver_s", "total_s")
        }
        print(f"  {people}x{centres} s{seed} t{limit:g}: {shown}", flush=True)

    t = time.perf_counter()
    fallback = exact_weighted_sum_pairs(static, n_dir, OBJECTIVES)
    emit("fallback", fallback, time.perf_counter() - t, 0.0)

    for method, cache, max_load, table_s in (
        ("mip_soft", soft, None, soft_table_s),
        ("mip_hard", hard, limits, hard_table_s),
    ):
        try:
            r = solve_weighted_mip(cache, n_dir, OBJECTIVES, time_limit=limit, max_load=max_load)
        except RuntimeError:  # no incumbent within the limit: a recorded failure, not a gap
            emit(method, None, limit, table_s, optimal=False)
            continue
        emit(method, r.pairs, r.seconds, table_s, gap=r.gap, optimal=r.optimal)

    r = local_search(soft, n_dir, OBJECTIVES, fallback, time_budget=limit)
    emit("ls_soft", r.pairs, r.seconds, soft_table_s, moves=r.moves, optimal="")
    t = time.perf_counter()
    start = capacity_greedy(static, n_dir, OBJECTIVES, limits)
    start_s = time.perf_counter() - t
    emit("capacity_greedy", start, start_s, 0.0)
    r = local_search(hard, n_dir, OBJECTIVES, start, max_load=limits, time_budget=limit)
    emit("ls_hard", r.pairs, start_s + r.seconds, hard_table_s, moves=r.moves)

    seeds = [fallback] + [
        exact_weighted_sum_pairs(static, n_dir, OBJECTIVES, w) for w in _sweep(SWEEP_STEPS)
    ]
    for method, seed_allocations in (("nsga2", None), ("nsga2_seeded", seeds)):
        t = time.perf_counter()
        front = solve(
            problem, cfg, cache=soft, time_budget=limit, seed_allocations=seed_allocations
        )
        secs = time.perf_counter() - t
        canon = pairs_of(canonical_decision(front), problem)
        emit(f"{method}_committed", canon, secs, soft_table_s, generations=front.generations_run)
        best = min(
            (pairs_of(sol, problem) for sol in front.solutions),
            key=lambda p: float(np.sum(evaluate_pairs(p, soft, OBJECTIVES))),
        )
        emit(f"{method}_best_point", best, secs, soft_table_s, generations=front.generations_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--seed", type=int, required=True, help="instance seed (0, 1, 2)")
    parser.add_argument("--min-people", type=int, default=0)
    parser.add_argument("--max-people", type=int, default=3000)
    parser.add_argument(
        "--instances", type=Path, default=Path("experiments/results/envelope/instances")
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/results/envelope"))
    parser.add_argument("--tag", default="", help="suffix for the manifest name (split runs)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / f"envelope_t{args.time_limit:g}_s{args.seed}{args.tag}.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for people, centres in SIZES:
            if args.min_people <= people <= args.max_people:
                run_instance(
                    people, centres, args.seed, args.time_limit, args.instances, writer, fh
                )


if __name__ == "__main__":
    main()
