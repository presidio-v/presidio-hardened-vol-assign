"""How complete is the sampled supported front? (Paper B revision, R3.2)

The exact reference front of the paper solves the weighted sum exactly at 231 weight
vectors (21 steps per axis). That samples the supported non-dominated points; it does not
prove that every one was found. This script refines the lattice (21, 41, 81, 161 steps per
axis) on both reference instances and reports, for each lattice, the number of distinct
supported points, the front's hypervolume, and the hypervolume share of the evolutionary
fronts at 150 and 9600 generations. One reference point (1.1x the component-wise maximum
over the finest exact front and every evolutionary front) serves all lattices, so shares
are comparable across lattices.

Reads fronts from ``--fronts`` (a ``seed_floor`` results directory); writes
``experiments/results/audit/weight_grid.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV

from experiments.analyze_audit import nondominated
from experiments.generate_instances import SIZES
from experiments.run_turbulence import OBJECTIVES, _config
from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs
from presidio_vol_assign.allocation.fast_fis import precompute_fis_cache_fast
from presidio_vol_assign.allocation.solvers import evaluate_pairs
from presidio_vol_assign.allocation.validation import load_allocation_problem

STEPS = (20, 40, 80, 160)
RUNS = {"small": ("small_gen150", "small_gen9600"), "large": ("large_gen150", "large_gen9600")}


def supported_front(cache, n_dir: int, steps: int) -> np.ndarray:  # noqa: ANN001
    pts = []
    for a in range(steps + 1):
        for b in range(steps + 1 - a):
            w = (a / steps, b / steps, (steps - a - b) / steps)
            pts.append(
                evaluate_pairs(exact_weighted_sum_pairs(cache, n_dir, OBJECTIVES, w), cache, 3)
            )
    return nondominated(np.array(pts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fronts", type=Path, default=Path("experiments/results/seed_floor"))
    args = parser.parse_args()
    rows = []
    for size, runs in RUNS.items():
        base = Path("experiments/instances") / size
        problem = load_allocation_problem(
            base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=SIZES[size].n_dir
        )
        cache = precompute_fis_cache_fast(problem, _config(100, 150))
        exact = {s: supported_front(cache, problem.n_dir, s) for s in STEPS}

        def load(run: str) -> list[np.ndarray]:
            return [
                np.array([sol["fitness"] for sol in json.loads(line)["solutions"]])
                for line in (args.fronts / run / "fronts.jsonl").open()
            ]

        fronts = {run: load(run) for run in runs}
        # Same reference point as analyze_audit: every front of this instance, plus the exact
        # front (here the finest), so the shares match Table 2.
        same_instance = [
            d.name
            for d in args.fronts.iterdir()
            if (d / "meta.json").exists()
            and json.loads((d / "meta.json").read_text())["size"] == size
            and not json.loads((d / "meta.json").read_text()).get("instances")
            and (d / "fronts.jsonl").exists()
        ]
        everything = [f for run in same_instance for f in load(run)]
        ref = np.vstack([exact[STEPS[-1]], *everything]).max(axis=0) * 1.1
        hv = HV(ref_point=ref)
        finest = float(hv(exact[STEPS[-1]]))
        for s in STEPS:
            hv_exact = float(hv(exact[s]))
            row = {
                "size": size,
                "steps": s,
                "weight_vectors": (s + 1) * (s + 2) // 2,
                "supported_points": len(exact[s]),
                "hv_exact": hv_exact,
                "hv_vs_finest": hv_exact / finest,
            }
            for run in runs:
                shares = [float(hv(nondominated(f))) / hv_exact for f in fronts[run]]
                tag = run.split("_")[1]
                row[f"share_{tag}_mean"] = float(np.mean(shares))
                row[f"share_{tag}_min"] = float(np.min(shares))
                row[f"share_{tag}_max"] = float(np.max(shares))
            rows.append(row)
            print(row, flush=True)
    out = Path("experiments/results/audit")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "weight_grid.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
