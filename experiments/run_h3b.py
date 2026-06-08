"""H3b experiment driver — weight perturbation sensitivity.

Operationalisation per `pubs/v0.2.0-mdpi/explore/hypothesis-rq.md`:

    For 100 Latin-Hypercube samples of (WAS, WDS, WIL, WLS, WRC, WPH) drawn
    from baseline·(1±0.20), run NSGA-II × 30 reps on the medium-size 4-obj
    problem. H3b is confirmed when CV (= std/mean) of the mean objectives
    across the 100 samples is ≤ 10% for every objective.

Per-sample protocol:
    1. Rebuild FIS cache with that sample's weights (~10s on medium).
    2. Run NSGA-II × 30 reps with rep-specific seeds.
    3. Average the four mean objectives and HV across the 30 reps.
    4. Append a row to h3b_manifest.csv.

Usage:
    .venv/bin/python -m experiments.run_h3b [--n-samples 100] [--reps 30] [--bound 0.2]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from experiments.generate_instances import SIZES
from experiments.run_h1_h2_h4 import BASE_SEED, SEED_STEP
from presidio_vol_assign.allocation.metrics import compute_allocation_metrics
from presidio_vol_assign.allocation.models import (
    AllocationConfig,
    AllocationSolverType,
    Weights,
)
from presidio_vol_assign.allocation.sensitivity import lhs_weight_samples
from presidio_vol_assign.allocation.solvers import precompute_fis_cache, solve
from presidio_vol_assign.allocation.validation import load_allocation_problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--bound", type=float, default=0.2)
    parser.add_argument("--lhs-seed", type=int, default=2026)
    parser.add_argument("--size", default="medium", choices=sorted(SIZES.keys()))
    parser.add_argument(
        "--solver",
        default="nsga2",
        choices=[a.value for a in AllocationSolverType],
        help="Solver held fixed across the sweep (default nsga2 per H3b spec).",
    )
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/results/h3b"),
        help="Output directory.",
    )
    args = parser.parse_args()

    spec = SIZES[args.size]
    base = Path("experiments/instances") / args.size
    problem = load_allocation_problem(
        base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=spec.n_dir
    )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "h3b_manifest.csv"

    samples = lhs_weight_samples(
        baseline=Weights(),
        n_samples=args.n_samples,
        bound=args.bound,
        seed=args.lhs_seed,
    )

    print(
        f"H3b sweep: {args.n_samples} LHS samples × {args.reps} reps "
        f"(±{args.bound:.0%}) on {args.size} 4-obj {args.solver}",
        flush=True,
    )

    header = (
        "sample_id",
        "was",
        "wds",
        "wil",
        "wls",
        "wrc",
        "wph",
        "n_reps",
        # per-objective mean-of-rep-means
        "mn_ulpp_avg",
        "mn_ulpp_min_avg",
        "mn_trd_avg",
        "mn_trd_min_avg",
        "mn_rpd_avg",
        "mn_rpd_min_avg",
        "mn_cail_avg",
        "mn_cail_min_avg",
        # quality + cost
        "nns_avg",
        "hv_avg",
        "mid_avg",
        "sm_avg",
        "wall_time_sec",
    )

    solver_type = AllocationSolverType(args.solver)
    total_t0 = time.monotonic()

    with manifest_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for sample in samples:
            sample_t0 = time.monotonic()
            sweep_cfg = AllocationConfig(
                solver=solver_type,
                objectives=4,
                weights=sample.weights,
                pop_size=args.pop_size,
                generations=args.generations,
                seed=None,  # set per rep
            )
            cache = precompute_fis_cache(problem, sweep_cfg)

            ulpp_means: list[float] = []
            ulpp_mins: list[float] = []
            trd_means: list[float] = []
            trd_mins: list[float] = []
            rpd_means: list[float] = []
            rpd_mins: list[float] = []
            cail_means: list[float] = []
            cail_mins: list[float] = []
            nns_list: list[int] = []
            hv_list: list[float] = []
            mid_list: list[float] = []
            sm_list: list[float] = []

            for rep in range(args.reps):
                cfg = AllocationConfig(
                    solver=solver_type,
                    objectives=4,
                    weights=sample.weights,
                    pop_size=args.pop_size,
                    generations=args.generations,
                    seed=BASE_SEED + rep * SEED_STEP,
                )
                front = solve(problem, cfg, cache=cache)
                metrics = compute_allocation_metrics(front)

                if front.solutions:
                    ulpp_vals = [s.mn_ulpp for s in front.solutions]
                    trd_vals = [s.mn_trd for s in front.solutions]
                    rpd_vals = [s.mn_rpd for s in front.solutions]
                    cail_vals = [s.mn_cail for s in front.solutions]
                    ulpp_means.append(float(np.mean(ulpp_vals)))
                    ulpp_mins.append(float(np.min(ulpp_vals)))
                    trd_means.append(float(np.mean(trd_vals)))
                    trd_mins.append(float(np.min(trd_vals)))
                    rpd_means.append(float(np.mean(rpd_vals)))
                    rpd_mins.append(float(np.min(rpd_vals)))
                    cail_means.append(float(np.mean(cail_vals)))
                    cail_mins.append(float(np.min(cail_vals)))
                nns_list.append(metrics.nns)
                hv_list.append(metrics.hv)
                mid_list.append(metrics.mid)
                sm_list.append(metrics.sm)

            wall = time.monotonic() - sample_t0
            row = [
                sample.sample_id,
                round(sample.weights.was, 6),
                round(sample.weights.wds, 6),
                round(sample.weights.wil, 6),
                round(sample.weights.wls, 6),
                round(sample.weights.wrc, 6),
                round(sample.weights.wph, 6),
                args.reps,
                round(float(np.mean(ulpp_means)), 6) if ulpp_means else "",
                round(float(np.mean(ulpp_mins)), 6) if ulpp_mins else "",
                round(float(np.mean(trd_means)), 6) if trd_means else "",
                round(float(np.mean(trd_mins)), 6) if trd_mins else "",
                round(float(np.mean(rpd_means)), 6) if rpd_means else "",
                round(float(np.mean(rpd_mins)), 6) if rpd_mins else "",
                round(float(np.mean(cail_means)), 6) if cail_means else "",
                round(float(np.mean(cail_mins)), 6) if cail_mins else "",
                round(float(np.mean(nns_list)), 2),
                round(float(np.mean(hv_list)), 2),
                round(float(np.mean(mid_list)), 4),
                round(float(np.mean(sm_list)), 4),
                round(wall, 2),
            ]
            writer.writerow(row)
            fh.flush()
            print(
                f"  [{sample.sample_id + 1:3d}/{args.n_samples}] "
                f"weights=({sample.weights.was:.3f},{sample.weights.wds:.3f},"
                f"{sample.weights.wil:.3f},{sample.weights.wls:.3f},"
                f"{sample.weights.wrc:.3f},{sample.weights.wph:.3f})  "
                f"hv_avg={np.mean(hv_list):>14.0f}  wall={wall:6.1f}s",
                flush=True,
            )

    total_wall = time.monotonic() - total_t0
    print(
        f"\nDone: {args.n_samples} samples in {total_wall / 60:.1f} min "
        f"(mean per sample {total_wall / args.n_samples:.1f}s)."
    )
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
