"""RQ3 operability profiler (Paper B): is the decision system deployable under pressure?

Measures the wall-clock latency and peak Python memory of one ``solve()`` call per
(size, algorithm), across repeated seeds, and reports the operability envelope —
latency, peak memory, and a throughput figure (decisions/hour). Writes a manifest.

Timing is only meaningful on an otherwise-idle machine; do not run this alongside
the turbulence sweep.
"""

from __future__ import annotations

import argparse
import csv
import time
import tracemalloc
from pathlib import Path

import numpy as np

from experiments.generate_instances import SIZES
from experiments.run_h1_h2_h4 import BASE_SEED, SEED_STEP
from presidio_vol_assign.allocation.models import (
    AllocationConfig,
    AllocationSolverType,
    Weights,
)
from presidio_vol_assign.allocation.solvers import precompute_fis_cache, solve
from presidio_vol_assign.allocation.validation import load_allocation_problem

OBJECTIVES = 3  # Paper B: the 3-objective relief core
_ALGORITHMS = (AllocationSolverType.NSGA2, AllocationSolverType.NRGA)


def _config(
    solver: AllocationSolverType, pop_size: int, generations: int, seed: int
) -> AllocationConfig:
    return AllocationConfig(
        solver=solver,
        objectives=OBJECTIVES,
        weights=Weights(),
        pop_size=pop_size,
        generations=generations,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="small,large", help="comma-separated size names")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("experiments/results/operability"))
    args = parser.parse_args()

    sizes = [s for s in args.sizes.split(",") if s]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "operability_manifest.csv"
    header = [
        "size",
        "algorithm",
        "rep",
        "seed",
        "n_people",
        "n_centers",
        "n_dir",
        "nns",
        "latency_sec",
        "peak_mem_mb",
    ]

    print(
        f"RQ3 operability: {sizes} × {[a.value for a in _ALGORITHMS]} × {args.reps} reps",
        flush=True,
    )
    rows: list[dict] = []
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for size in sizes:
            spec = SIZES[size]
            base = Path("experiments/instances") / size
            problem = load_allocation_problem(
                base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=spec.n_dir
            )
            for algorithm in _ALGORITHMS:
                for rep in range(args.reps):
                    seed = BASE_SEED + rep * SEED_STEP
                    cfg = _config(algorithm, args.pop_size, args.generations, seed)
                    cache = precompute_fis_cache(problem, cfg)  # excluded from the timed region
                    tracemalloc.start()
                    start = time.perf_counter()
                    front = solve(problem, cfg, cache=cache)
                    latency = time.perf_counter() - start
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    row = {
                        "size": size,
                        "algorithm": algorithm.value,
                        "rep": rep,
                        "seed": seed,
                        "n_people": problem.n_people,
                        "n_centers": problem.n_centers,
                        "n_dir": problem.n_dir,
                        "nns": front.nns,
                        "latency_sec": round(latency, 4),
                        "peak_mem_mb": round(peak / 1e6, 3),
                    }
                    writer.writerow(row)
                    rows.append(row)
                lat = [
                    r["latency_sec"]
                    for r in rows
                    if r["size"] == size and r["algorithm"] == algorithm.value
                ]
                mem = [
                    r["peak_mem_mb"]
                    for r in rows
                    if r["size"] == size and r["algorithm"] == algorithm.value
                ]
                med = float(np.median(lat))
                print(
                    f"  {size}/{algorithm.value}: median latency {med:.3f}s "
                    f"(~{3600 / med:.0f} decisions/h), peak {float(np.median(mem)):.1f} MB",
                    flush=True,
                )

    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
