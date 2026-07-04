"""RQ2 reproducibility driver (Paper B): is the decision bit-for-bit reproducible?

For each (size, seed) it solves twice in-process and confirms the allocation front
hashes identically (in-process determinism, REP = 1.0), records the front signature
alongside an environment fingerprint (platform, Python, and library versions), and
writes a manifest. Cross-environment reproducibility is then established by running
this same driver on each target OS/Python (e.g. the CI matrix) and comparing the
``signature`` column for matching (size, seed): identical signatures across
environments -> REP = 1.0; any divergence is the audit trail that shows where.

This is the operational-trust guardrail that RQ1's input-fragility finding makes
load-bearing: because a single run's decision is input-fragile, you must at least be
able to prove that the run itself is reproducible and auditable.
"""

from __future__ import annotations

import argparse
import csv
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from experiments.generate_instances import SIZES
from experiments.run_h1_h2_h4 import BASE_SEED, SEED_STEP
from presidio_vol_assign.allocation.models import (
    AllocationConfig,
    AllocationSolverType,
    Weights,
)
from presidio_vol_assign.allocation.repro import allocation_front_signature, rep_score
from presidio_vol_assign.allocation.solvers import precompute_fis_cache, solve
from presidio_vol_assign.allocation.validation import load_allocation_problem

OBJECTIVES = 3  # Paper B: the 3-objective relief core


def _env_fingerprint() -> dict[str, str]:
    def _v(pkg: str) -> str:
        try:
            return version(pkg)
        except PackageNotFoundError:
            return "n/a"

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": _v("numpy"),
        "scipy": _v("scipy"),
        "deap": _v("deap"),
        "pymoo": _v("pymoo"),
    }


def _config(size_pop: int, generations: int, seed: int) -> AllocationConfig:
    return AllocationConfig(
        solver=AllocationSolverType.NSGA2,
        objectives=OBJECTIVES,
        weights=Weights(),
        pop_size=size_pop,
        generations=generations,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="small", help="comma-separated size names")
    parser.add_argument("--reps", type=int, default=5, help="distinct seeds")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("experiments/results/repro"))
    args = parser.parse_args()

    env = _env_fingerprint()
    sizes = [s for s in args.sizes.split(",") if s]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "repro_manifest.csv"
    fields = [*env, "size", "seed", "signature", "in_process_rep"]

    print(f"RQ2 reproducibility on {env['platform']} / py{env['python']}", flush=True)
    all_rep: list[float] = []
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for size in sizes:
            spec = SIZES[size]
            base = Path("experiments/instances") / size
            problem = load_allocation_problem(
                base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=spec.n_dir
            )
            cfg0 = _config(args.pop_size, args.generations, seed=BASE_SEED)
            cache = precompute_fis_cache(problem, cfg0)
            for rep in range(args.reps):
                seed = BASE_SEED + rep * SEED_STEP
                cfg = _config(args.pop_size, args.generations, seed=seed)
                sig1 = allocation_front_signature(solve(problem, cfg, cache=cache))
                sig2 = allocation_front_signature(solve(problem, cfg, cache=cache))
                in_process = rep_score([sig1, sig2])
                all_rep.append(in_process)
                writer.writerow(
                    {
                        **env,
                        "size": size,
                        "seed": seed,
                        "signature": sig1,
                        "in_process_rep": in_process,
                    }
                )
            print(f"  {size}: {args.reps} seeds done", flush=True)

    verdict = min(all_rep) if all_rep else 0.0  # fail closed: no runs -> not reproducible
    print(f"within-environment REP (min over seeds): {verdict}")
    print(f"wrote {manifest}")
    print("cross-environment: run this on each target OS/Python and diff the 'signature' column.")


if __name__ == "__main__":
    main()
