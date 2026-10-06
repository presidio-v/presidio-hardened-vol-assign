"""H3a experiment driver — rule-base perturbation sensitivity.

Operationalisation per `pubs/v0.2.0-mdpi/explore/hypothesis-rq.md`:

    For each single-rule deletion across the four 4-obj FIS rule bases
    (FIS1: 27 rules, FIS2A: 9, FIS2B: 3, FIS3: 27 — 66 deletions total),
    run NSGA-II × 30 reps on the medium-size 4-obj problem. H3a is
    confirmed when median ΔHV ≤ 5% across all single-rule-deletion
    variants per FIS.

Per-deletion protocol:
    1. Apply override via `fis_overrides({fis_name: [rule_index]})`.
    2. Rebuild FIS cache with the perturbed system.
    3. Run NSGA-II × 30 reps with rep-specific seeds.
    4. Record mean objectives + HV.

A baseline row (no overrides) is recorded first, against which every
deletion's ΔHV is computed.

Usage:
    .venv/bin/python -m experiments.run_h3a [--reps 30]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from experiments.generate_instances import SIZES
from experiments.run_h1_h2_h4 import BASE_SEED, SEED_STEP
from presidio_vol_assign.allocation.fis import RULE_COUNTS, fis_overrides
from presidio_vol_assign.allocation.metrics import compute_allocation_metrics
from presidio_vol_assign.allocation.models import (
    AllocationConfig,
    AllocationSolverType,
    Weights,
)
from presidio_vol_assign.allocation.solvers import precompute_fis_cache, solve
from presidio_vol_assign.allocation.validation import load_allocation_problem

# 4-obj uses FIS1, FIS2A, FIS2B, FIS3 — never FIS2_TIL.
FOUR_OBJ_FIS_NAMES = ("fis1", "fis2a_trd", "fis2b_rpd", "fis3")


def _enumerate_deletions() -> list[tuple[str, int]]:
    """Return [(fis_name, rule_index), …] across the four 4-obj FIS rule bases."""
    deletions: list[tuple[str, int]] = []
    for name in FOUR_OBJ_FIS_NAMES:
        for idx in range(RULE_COUNTS[name]):
            deletions.append((name, idx))
    return deletions


def _run_cell(
    problem,
    overrides: dict[str, list[int]] | None,
    reps: int,
    pop_size: int,
    generations: int,
) -> dict:
    """Run NSGA-II × `reps` reps with the given FIS overrides; return aggregated metrics."""
    cfg_template = AllocationConfig(
        solver=AllocationSolverType.NSGA2,
        objectives=4,
        weights=Weights(),
        pop_size=pop_size,
        generations=generations,
    )
    if overrides:
        ctx = fis_overrides(overrides)
    else:
        from contextlib import nullcontext

        ctx = nullcontext()

    with ctx:
        cache = precompute_fis_cache(problem, cfg_template)
        ulpp_means: list[float] = []
        trd_means: list[float] = []
        rpd_means: list[float] = []
        cail_means: list[float] = []
        nns_list: list[int] = []
        hv_list: list[float] = []
        cpu_list: list[float] = []

        for rep in range(reps):
            cfg = AllocationConfig(
                solver=AllocationSolverType.NSGA2,
                objectives=4,
                weights=Weights(),
                pop_size=pop_size,
                generations=generations,
                seed=BASE_SEED + rep * SEED_STEP,
            )
            front = solve(problem, cfg, cache=cache)
            metrics = compute_allocation_metrics(front)
            if front.solutions:
                ulpp_means.append(float(np.mean([s.mn_ulpp for s in front.solutions])))
                trd_means.append(float(np.mean([s.mn_trd for s in front.solutions])))
                rpd_means.append(float(np.mean([s.mn_rpd for s in front.solutions])))
                cail_means.append(float(np.mean([s.mn_cail for s in front.solutions])))
            nns_list.append(metrics.nns)
            hv_list.append(metrics.hv)
            cpu_list.append(metrics.cpu_time_sec)

    return {
        "mn_ulpp_avg": float(np.mean(ulpp_means)) if ulpp_means else 0.0,
        "mn_trd_avg": float(np.mean(trd_means)) if trd_means else 0.0,
        "mn_rpd_avg": float(np.mean(rpd_means)) if rpd_means else 0.0,
        "mn_cail_avg": float(np.mean(cail_means)) if cail_means else 0.0,
        "nns_avg": float(np.mean(nns_list)),
        "hv_avg": float(np.mean(hv_list)),
        "cpu_avg": float(np.mean(cpu_list)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--size", default="medium", choices=sorted(SIZES.keys()))
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("experiments/results/h3a"))
    args = parser.parse_args()

    spec = SIZES[args.size]
    base = Path("experiments/instances") / args.size
    problem = load_allocation_problem(
        base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=spec.n_dir
    )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "h3a_manifest.csv"

    deletions = _enumerate_deletions()
    print(
        f"H3a sweep: 1 baseline + {len(deletions)} deletions × {args.reps} reps "
        f"on {args.size} 4-obj NSGA-II",
        flush=True,
    )

    header = (
        "fis_name",
        "rule_index",
        "mn_ulpp_avg",
        "mn_trd_avg",
        "mn_rpd_avg",
        "mn_cail_avg",
        "nns_avg",
        "hv_avg",
        "cpu_avg",
        "delta_hv_pct",
        "wall_time_sec",
    )

    total_t0 = time.monotonic()
    with manifest_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        # Baseline first
        t0 = time.monotonic()
        baseline = _run_cell(problem, None, args.reps, args.pop_size, args.generations)
        wall = time.monotonic() - t0
        writer.writerow(
            [
                "baseline",
                -1,
                round(baseline["mn_ulpp_avg"], 6),
                round(baseline["mn_trd_avg"], 6),
                round(baseline["mn_rpd_avg"], 6),
                round(baseline["mn_cail_avg"], 6),
                round(baseline["nns_avg"], 2),
                round(baseline["hv_avg"], 2),
                round(baseline["cpu_avg"], 4),
                0.0,
                round(wall, 2),
            ]
        )
        fh.flush()
        baseline_hv = baseline["hv_avg"]
        print(
            f"  baseline                       hv_avg={baseline_hv:>14.0f}  wall={wall:6.1f}s",
            flush=True,
        )

        for idx, (fis_name, rule_idx) in enumerate(deletions, start=1):
            t0 = time.monotonic()
            cell = _run_cell(
                problem, {fis_name: [rule_idx]}, args.reps, args.pop_size, args.generations
            )
            wall = time.monotonic() - t0
            delta = 100.0 * (cell["hv_avg"] - baseline_hv) / baseline_hv if baseline_hv else 0.0
            writer.writerow(
                [
                    fis_name,
                    rule_idx,
                    round(cell["mn_ulpp_avg"], 6),
                    round(cell["mn_trd_avg"], 6),
                    round(cell["mn_rpd_avg"], 6),
                    round(cell["mn_cail_avg"], 6),
                    round(cell["nns_avg"], 2),
                    round(cell["hv_avg"], 2),
                    round(cell["cpu_avg"], 4),
                    round(delta, 4),
                    round(wall, 2),
                ]
            )
            fh.flush()
            print(
                f"  [{idx:3d}/{len(deletions)}] {fis_name:10s} rule {rule_idx:2d}  "
                f"hv_avg={cell['hv_avg']:>14.0f}  ΔHV={delta:+6.2f}%  wall={wall:6.1f}s",
                flush=True,
            )

    total_min = (time.monotonic() - total_t0) / 60
    print(f"\nDone: {len(deletions)} deletions in {total_min:.1f} min. Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
