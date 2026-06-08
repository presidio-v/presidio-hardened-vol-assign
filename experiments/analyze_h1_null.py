"""H1 null-model control (M6).

Addresses Moderate issue M6 in the adversarial review: the 50%-projection-
dominance result in §6.2 is geometric in part — *some* fraction of any
4D Pareto front will be dominated when projected to 3D simply because the
projection loses information. We need a null distribution.

For each 4-obj front, we sample 100 random convex fusions of the form
    TIL_null = α · TRD + (1-α) · RPD,    α ~ U(0,1)
on a per-allocation basis (one α per fusion, applied uniformly across
allocations within that fusion). For each fusion we compute the
3-obj-projection dominance fraction. The 100 fusions form an empirical
null distribution against which the ATRes-RWS-weighted fusion's
dominance fraction is compared.

If the ATRes fraction is statistically distinguishable from the null
distribution mean, H1 is *informational*: the ATRes fusion's choice of
weighting carries content beyond generic projection loss. If it falls
inside the null distribution, H1 is geometric.

Output: experiments/results/h1_h2_h4/h1_null_analysis.csv with one row
per (size, algorithm, rep) carrying the actual fraction, the null mean,
the null 95% CI, and a z-score / percentile.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze_h1 import (
    _dominance_mask,
    precompute_til,
    reconstruct_front,
)
from experiments.generate_instances import SIZES
from presidio_vol_assign.allocation.models import AllocationProblem, Weights
from presidio_vol_assign.allocation.validation import load_allocation_problem


def _null_dominance(
    front,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an array of `n_samples` projection-dominance fractions under
    random convex fusions of TRD and RPD."""
    if not front.solutions:
        return np.array([])
    ulpp = np.array([s.mn_ulpp for s in front.solutions])
    cail = np.array([s.mn_cail for s in front.solutions])
    # Per-allocation TRD and RPD averaged to per-solution means already
    # exist on the solution; for the null we work with these.
    trd = np.array([s.mn_trd for s in front.solutions])
    rpd = np.array([s.mn_rpd for s in front.solutions])

    fractions = np.empty(n_samples, dtype=float)
    n = len(front.solutions)
    for k in range(n_samples):
        alpha = float(rng.random())
        til_null = alpha * trd + (1.0 - alpha) * rpd
        pts = np.column_stack([ulpp, til_null, cail])
        dominated = _dominance_mask(pts)
        fractions[k] = float(dominated.sum()) / n
    return fractions


def main() -> None:
    n_null_samples = 100
    rng = np.random.default_rng(seed=2026)

    manifest = pd.read_csv("experiments/results/h1_h2_h4/manifest.csv")
    h1 = pd.read_csv("experiments/results/h1_h2_h4/h1_analysis.csv")

    problems: dict[str, AllocationProblem] = {}
    til_caches: dict[str, dict[tuple[str, str], float]] = {}
    for size in SIZES:
        spec = SIZES[size]
        base = Path("experiments/instances") / size
        problems[size] = load_allocation_problem(
            base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=spec.n_dir
        )
        til_caches[size] = precompute_til(problems[size], Weights())
        print(f"  TIL cache for {size:6s}: {len(til_caches[size])} pairs")

    four = manifest[manifest["objectives"] == 4]
    print(f"\nProcessing {len(four)} 4-obj fronts with {n_null_samples} null samples each...")
    t0 = time.monotonic()
    rows = []
    for _, row in four.iterrows():
        run_dir = Path("experiments/results/h1_h2_h4") / row["run_id"]
        front = reconstruct_front(run_dir)
        # Actual ATRes-RWS-weighted fraction from h1_analysis.csv
        h1_row = h1[
            (h1["size"] == row["size"])
            & (h1["algorithm"] == row["algorithm"])
            & (h1["rep"] == row["rep"])
        ].iloc[0]
        actual_frac = float(h1_row["fraction_dominated"])

        # Null distribution
        null_fracs = _null_dominance(front, n_null_samples, rng)
        if null_fracs.size == 0:
            continue
        null_mean = float(null_fracs.mean())
        null_std = float(null_fracs.std())
        # Percentile rank of the actual fraction in the null distribution
        if null_std > 0:
            z = (actual_frac - null_mean) / null_std
        else:
            z = float("nan")
        pct_le = 100.0 * float((null_fracs <= actual_frac).mean())

        rows.append(
            {
                "size": row["size"],
                "algorithm": row["algorithm"],
                "rep": row["rep"],
                "actual_fraction": actual_frac,
                "null_mean": null_mean,
                "null_std": null_std,
                "null_p05": float(np.percentile(null_fracs, 5)),
                "null_p95": float(np.percentile(null_fracs, 95)),
                "z_score": z,
                "pct_below_actual": pct_le,
            }
        )

    print(f"  Done in {time.monotonic() - t0:.1f}s.\n")
    out = pd.DataFrame(rows)

    print("=" * 80)
    print("Null-model summary by (size × algorithm)")
    print("=" * 80)
    agg = (
        out.groupby(["size", "algorithm"])
        .agg(
            actual_frac_mean=("actual_fraction", "mean"),
            null_frac_mean=("null_mean", "mean"),
            mean_z=("z_score", "mean"),
            mean_pct=("pct_below_actual", "mean"),
        )
        .round(3)
    )
    print(agg.to_string())

    # Pooled
    print()
    print("Pooled across all 270 fronts:")
    print(f"  actual fraction mean = {out['actual_fraction'].mean():.3f}")
    print(f"  null   fraction mean = {out['null_mean'].mean():.3f}")
    print(
        f"  delta (actual - null mean) = {(out['actual_fraction'] - out['null_mean']).mean():+.3f}"
    )
    print(f"  mean z-score (actual vs null) = {out['z_score'].dropna().mean():+.3f}")
    print(
        f"  fraction of fronts with actual > null p95 = "
        f"{100.0 * (out['actual_fraction'] > out['null_p95']).mean():.1f}%"
    )
    print(
        f"  fraction of fronts with actual < null p05 = "
        f"{100.0 * (out['actual_fraction'] < out['null_p05']).mean():.1f}%"
    )

    out_path = Path("experiments/results/h1_h2_h4/h1_null_analysis.csv")
    out.to_csv(out_path, index=False)
    print(f"\nDetail → {out_path}")


if __name__ == "__main__":
    main()
