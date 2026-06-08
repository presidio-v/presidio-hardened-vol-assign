"""H1+H2+H4 experiment driver.

Orchestrates the (size × formulation × algorithm × rep) cells of the
MDPI experiment plan, reusing one `FISCache` per (size, formulation)
combination so scikit-fuzzy precompute only runs 6 times for the entire
matrix instead of once per individual run.

Cells covered (default full matrix):
    sizes        ∈ {small, medium, large}        — 3 levels
    formulation  ∈ {3, 4}                        — 2 levels
    algorithm    ∈ {nsga2, nrga, nsga3}          — 3 levels
    reps         ∈ {0, …, 29}                    — 30 reps
    Total: 3 × 2 × 3 × 30 = 540 runs.

Per-rep seed is `BASE_SEED + rep_index * SEED_STEP`, so reruns of any
single cell are bit-exact. Each run writes a Pareto CSV + metrics JSON
under `experiments/results/h1_h2_h4/<run_id>/`. A top-level
`manifest.csv` records every run with its (size, formulation, algorithm,
rep, seed), wall-clock time, and the standard NNS/MID/SM/HV metrics.

Pilot mode (`--pilot`) runs a small subset (small × 4-obj × 3 algos × 5
reps = 15 runs) to validate the pipeline before committing to the full
matrix. Output goes to a separate `pilot/` subdirectory so it does not
collide with full-matrix runs.

Usage:
    .venv/bin/python -m experiments.run_h1_h2_h4 --pilot
    .venv/bin/python -m experiments.run_h1_h2_h4 --full
    .venv/bin/python -m experiments.run_h1_h2_h4 --sizes small medium --algorithms nsga3
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

from experiments.generate_instances import SIZES
from presidio_vol_assign.allocation.metrics import compute_allocation_metrics
from presidio_vol_assign.allocation.models import (
    AllocationConfig,
    AllocationSolverType,
    Weights,
)
from presidio_vol_assign.allocation.solvers import precompute_fis_cache, solve
from presidio_vol_assign.allocation.validation import load_allocation_problem
from presidio_vol_assign.allocation.writers import (
    write_allocation_csv,
    write_allocation_metrics_json,
    write_allocation_pareto_csv,
)

BASE_SEED = 1000
SEED_STEP = 7919  # large prime to spread seeds across the int domain
DEFAULT_POP_SIZE = 100
DEFAULT_GENERATIONS = 200
DEFAULT_REPS = 30


@dataclass(frozen=True)
class Cell:
    size: str
    objectives: int
    algorithm: AllocationSolverType
    rep: int

    @property
    def run_id(self) -> str:
        return f"{self.size}_{self.objectives}obj_{self.algorithm.value}_rep{self.rep:02d}"

    @property
    def seed(self) -> int:
        return BASE_SEED + self.rep * SEED_STEP


def _enumerate_cells(
    sizes: list[str],
    formulations: list[int],
    algorithms: list[AllocationSolverType],
    reps: int,
) -> list[Cell]:
    cells: list[Cell] = []
    for size in sizes:
        for objs in formulations:
            for algo in algorithms:
                for rep in range(reps):
                    cells.append(Cell(size=size, objectives=objs, algorithm=algo, rep=rep))
    return cells


def _load_problems(sizes: list[str], instances_dir: Path) -> dict:
    problems: dict = {}
    for size in sizes:
        spec = SIZES[size]
        path = instances_dir / size
        problems[size] = load_allocation_problem(
            path / "people.csv",
            path / "centers.csv",
            path / "travel.csv",
            n_dir=spec.n_dir,
        )
    return problems


def run_matrix(
    cells: list[Cell],
    instances_dir: Path,
    out_dir: Path,
    pop_size: int,
    generations: int,
    nsga3_divisions: int,
) -> Path:
    """Execute every Cell in `cells`, write per-run outputs and a manifest CSV.

    Caches FIS evaluations per (size, objectives) — the most expensive
    step under scikit-fuzzy. Returns the manifest CSV Path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    sizes = sorted({c.size for c in cells})
    problems = _load_problems(sizes, instances_dir)
    weights = Weights()

    # Pre-compute one cache per (size, objectives). Use a sample config
    # for the precompute call; objectives is the only field that matters.
    cache_keys = sorted({(c.size, c.objectives) for c in cells})
    caches: dict = {}
    for size, objs in cache_keys:
        sample_cfg = AllocationConfig(
            solver=AllocationSolverType.NSGA2,
            objectives=objs,
            weights=weights,
        )
        t0 = time.monotonic()
        caches[(size, objs)] = precompute_fis_cache(problems[size], sample_cfg)
        print(
            f"  [cache] {size:6s} {objs}-obj precomputed in {time.monotonic() - t0:6.1f}s",
            flush=True,
        )

    header = (
        "run_id",
        "size",
        "objectives",
        "algorithm",
        "rep",
        "seed",
        "n_people",
        "n_centers",
        "n_dir",
        "pop_size",
        "generations",
        "nns",
        "mid",
        "sm",
        "hv",
        "cpu_time_sec",
        "wall_time_sec",
    )

    with manifest_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        total = len(cells)
        for idx, cell in enumerate(cells, start=1):
            problem = problems[cell.size]
            cache = caches[(cell.size, cell.objectives)]
            run_dir = out_dir / cell.run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            cfg = AllocationConfig(
                solver=cell.algorithm,
                objectives=cell.objectives,
                weights=weights,
                pop_size=pop_size,
                generations=generations,
                seed=cell.seed,
                nsga3_divisions=nsga3_divisions,
                output_dir=str(run_dir),
            )

            t0 = time.monotonic()
            front = solve(problem, cfg, cache=cache)
            wall = time.monotonic() - t0

            write_allocation_pareto_csv(front, run_dir)
            write_allocation_csv(front, run_dir)
            metrics = compute_allocation_metrics(front)
            write_allocation_metrics_json(metrics, run_dir)

            writer.writerow(
                [
                    cell.run_id,
                    cell.size,
                    cell.objectives,
                    cell.algorithm.value,
                    cell.rep,
                    cell.seed,
                    problem.n_people,
                    problem.n_centers,
                    problem.n_dir,
                    pop_size,
                    generations,
                    metrics.nns,
                    round(metrics.mid, 6),
                    round(metrics.sm, 6),
                    round(metrics.hv, 6),
                    round(metrics.cpu_time_sec, 3),
                    round(wall, 3),
                ]
            )
            fh.flush()

            print(
                f"  [{idx:3d}/{total}] {cell.run_id:48s} "
                f"NNS={metrics.nns:3d}  HV={metrics.hv:>14.0f}  wall={wall:6.2f}s",
                flush=True,
            )

    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        type=Path,
        default=Path(__file__).parent / "instances",
        help="Directory holding small/medium/large subdirs (default experiments/instances).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results" / "h1_h2_h4",
        help="Output directory for per-run files and manifest.csv.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pilot", action="store_true", help="Pilot mode: small × 4-obj × 3 algos × 5 reps."
    )
    mode.add_argument("--full", action="store_true", help="Full matrix (default if no flag given).")
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=sorted(SIZES.keys()),
        default=None,
        help="Override sizes (otherwise driven by --pilot/--full).",
    )
    parser.add_argument(
        "--formulations",
        nargs="+",
        type=int,
        choices=[3, 4],
        default=None,
        help="Override formulations.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=[a.value for a in AllocationSolverType],
        default=None,
        help="Override algorithms.",
    )
    parser.add_argument("--reps", type=int, default=None, help="Override repetition count.")
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--nsga3-divisions", type=int, default=4)
    args = parser.parse_args()

    if args.pilot:
        sizes = ["small"]
        formulations = [4]
        algorithms = list(AllocationSolverType)
        reps = 5
        out_dir = args.out.parent / "h1_h2_h4_pilot"
    else:
        sizes = sorted(SIZES.keys())
        formulations = [3, 4]
        algorithms = list(AllocationSolverType)
        reps = DEFAULT_REPS
        out_dir = args.out

    # CLI overrides take precedence
    if args.sizes:
        sizes = args.sizes
    if args.formulations:
        formulations = args.formulations
    if args.algorithms:
        algorithms = [AllocationSolverType(a) for a in args.algorithms]
    if args.reps is not None:
        reps = args.reps

    cells = _enumerate_cells(sizes, formulations, algorithms, reps)
    print(
        f"Plan: {len(cells)} cells "
        f"(sizes={sizes}, formulations={formulations}, "
        f"algos={[a.value for a in algorithms]}, reps={reps})"
    )

    t0 = time.monotonic()
    manifest = run_matrix(
        cells,
        instances_dir=args.instances,
        out_dir=out_dir,
        pop_size=args.pop_size,
        generations=args.generations,
        nsga3_divisions=args.nsga3_divisions,
    )
    print(f"Done: {len(cells)} runs in {time.monotonic() - t0:.1f}s. Manifest → {manifest}")


if __name__ == "__main__":
    main()
