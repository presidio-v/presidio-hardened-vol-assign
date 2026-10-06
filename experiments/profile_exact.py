"""Timing of the exact decider and the shared fuzzy pre-computation (Paper B, RQ1/RQ3).

Times, per reference instance: the skfuzzy FIS cache, the compiled FIS cache, and one exact
greedy solve (median of 1000). Writes ``experiments/results/audit/exact_timing.csv`` with the
environment fingerprint, since the numbers are machine-specific.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from experiments.generate_instances import SIZES
from experiments.run_reproducibility import _env_fingerprint
from experiments.run_turbulence import OBJECTIVES, _config
from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs
from presidio_vol_assign.allocation.fast_fis import precompute_fis_cache_fast
from presidio_vol_assign.allocation.solvers import precompute_fis_cache
from presidio_vol_assign.allocation.validation import load_allocation_problem


def _timed(fn, repeat: int) -> float:  # noqa: ANN001
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def main() -> None:
    out = Path("experiments/results/audit")
    out.mkdir(parents=True, exist_ok=True)
    env = _env_fingerprint()
    rows = []
    for size in ("small", "large"):
        base = Path("experiments/instances") / size
        problem = load_allocation_problem(
            base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=SIZES[size].n_dir
        )
        cfg = _config(100, 150)
        cache = precompute_fis_cache_fast(problem, cfg)
        row = {
            "size": size,
            "skfuzzy_cache_s": _timed(lambda p=problem: precompute_fis_cache(p, cfg), 3),
            "compiled_cache_s": _timed(lambda p=problem: precompute_fis_cache_fast(p, cfg), 5),
            "exact_greedy_ms": 1e3
            * _timed(
                lambda c=cache, n=problem.n_dir: exact_weighted_sum_pairs(c, n, OBJECTIVES), 1000
            ),
            **{f"env_{k}": v for k, v in env.items()},
        }
        rows.append(row)
        print(row, flush=True)
    with (out / "exact_timing.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
