"""H1 analysis with cached TIL precompute.

Naive `project_pareto_4_to_3` re-evaluates FIS2_TIL per allocation, which
is hours for the full matrix. Here we precompute TIL once per (problem,
weights) — same idea as `precompute_fis_cache` — and the projection
becomes O(1) numpy lookups.

Output: experiments/results/h1_h2_h4/h1_analysis.csv with one row per
(size, algorithm, rep) and the joint H1 verdict per Section 6 of the
manuscript.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.generate_instances import SIZES
from presidio_vol_assign.allocation.fis import compute_rws, evaluate_fis2_til
from presidio_vol_assign.allocation.models import (
    Allocation,
    AllocationParetoFront,
    AllocationProblem,
    AllocationSolution,
    AllocationSolverType,
    Weights,
)
from presidio_vol_assign.allocation.validation import load_allocation_problem


def precompute_til(problem: AllocationProblem, weights: Weights) -> dict[tuple[str, str], float]:
    """Compute TIL_{j,i} = FIS2(TD_{j,i}, RWS_{j,i}) for every (person, center) pair."""
    cache: dict[tuple[str, str], float] = {}
    for key, travel in problem.travel.items():
        rws = compute_rws(travel, weights)
        cache[key] = evaluate_fis2_til(travel.travel_duration, rws)
    return cache


def reconstruct_front(run_dir: Path) -> AllocationParetoFront:
    pareto_csv = next(run_dir.glob("pareto_alloc_*.csv"))
    alloc_csv = next(run_dir.glob("allocations_alloc_*.csv"))
    pareto_df = pd.read_csv(pareto_csv)
    alloc_df = pd.read_csv(alloc_csv)
    solver = AllocationSolverType(str(pareto_df["solver"].iloc[0]).strip())
    sols: list[AllocationSolution] = []
    for sid, prow in pareto_df.iterrows():
        a_rows = alloc_df[alloc_df["solution_id"] == sid]
        allocations = [
            Allocation(
                person_id=str(ar["person_id"]),
                center_id=str(ar["center_id"]),
                ulpp=float(ar["ulpp"]),
                cail_contrib=float(ar["cail_contrib"]),
                trd=float(ar.get("trd", 0)),
                rpd=float(ar.get("rpd", 0)),
            )
            for _, ar in a_rows.iterrows()
        ]
        sols.append(
            AllocationSolution(
                allocations=allocations,
                objectives_count=4,
                mn_ulpp=float(prow["mn_ulpp"]),
                mn_trd=float(prow["mn_trd"]),
                mn_rpd=float(prow["mn_rpd"]),
                mn_cail=float(prow["mn_cail"]),
            )
        )
    return AllocationParetoFront(solver=solver, objectives_count=4, solutions=sols)


def project_with_cache(
    front: AllocationParetoFront,
    til_cache: dict[tuple[str, str], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (n, 3) projected fitness array and a boolean dominance mask."""
    rows = []
    for sol in front.solutions:
        til_values = [til_cache[(a.person_id, a.center_id)] for a in sol.allocations]
        mn_til = float(np.mean(til_values)) if til_values else 0.0
        rows.append((sol.mn_ulpp, mn_til, sol.mn_cail))
    pts = np.asarray(rows, dtype=float)
    dominated = _dominance_mask(pts)
    return pts, dominated


def _dominance_mask(pts: np.ndarray) -> np.ndarray:
    """O(n²) but fully vectorised: True at i iff some j strictly dominates i."""
    n = len(pts)
    if n < 2:
        return np.zeros(n, dtype=bool)
    diff = pts[None, :, :] - pts[:, None, :]  # (n, n, 3) — pts[j] - pts[i]
    le = (diff <= 0).all(axis=2)              # j dominates i in ≤ sense
    lt = (diff < 0).any(axis=2)               # at least one strictly less
    np.fill_diagonal(le, False)
    return (le & lt).any(axis=1)


def spearman_rho_array(trd: np.ndarray, rpd: np.ndarray) -> float:
    if len(trd) < 2 or len(set(trd)) < 2 or len(set(rpd)) < 2:
        return float("nan")
    # Manual Spearman via rank correlation
    rt = pd.Series(trd).rank().values
    rr = pd.Series(rpd).rank().values
    rt = rt - rt.mean()
    rr = rr - rr.mean()
    denom = math.sqrt((rt * rt).sum() * (rr * rr).sum())
    if denom == 0:
        return float("nan")
    return float((rt * rr).sum() / denom)


def main() -> None:
    manifest = pd.read_csv("experiments/results/h1_h2_h4/manifest.csv")
    out_path = Path("experiments/results/h1_h2_h4/h1_analysis.csv")

    # Pre-load problems and TIL caches
    problems: dict[str, AllocationProblem] = {}
    til_caches: dict[str, dict[tuple[str, str], float]] = {}
    for size in SIZES:
        spec = SIZES[size]
        base = Path("experiments/instances") / size
        problems[size] = load_allocation_problem(
            base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=spec.n_dir
        )
        t0 = time.monotonic()
        til_caches[size] = precompute_til(problems[size], Weights())
        print(
            f"  TIL cache for {size:6s}: {len(til_caches[size])} pairs in "
            f"{time.monotonic() - t0:.1f}s"
        )

    four = manifest[manifest["objectives"] == 4]
    print(f"\nProcessing {len(four)} 4-obj fronts...")
    t0 = time.monotonic()
    rows = []
    for _, row in four.iterrows():
        front = reconstruct_front(Path("experiments/results/h1_h2_h4") / row["run_id"])
        cache = til_caches[row["size"]]
        pts, dominated = project_with_cache(front, cache)
        trd = np.array([s.mn_trd for s in front.solutions])
        rpd = np.array([s.mn_rpd for s in front.solutions])
        rho = spearman_rho_array(trd, rpd)
        n = len(front.solutions)
        n_dom = int(dominated.sum())
        frac = n_dom / n if n else 0.0
        confirms = (frac >= 0.20) and (not math.isnan(rho)) and (abs(rho) < 0.5)
        rows.append(
            {
                "size": row["size"],
                "algorithm": row["algorithm"],
                "rep": row["rep"],
                "n_solutions": n,
                "spearman_rho": rho,
                "n_dominated": n_dom,
                "fraction_dominated": frac,
                "confirms_h1": confirms,
            }
        )
    print(f"  Done in {time.monotonic() - t0:.1f}s.\n")

    res = pd.DataFrame(rows)
    print("=" * 90)
    print("H1 summary by (size × algorithm) — n=30 reps")
    print("=" * 90)
    agg = (
        res.groupby(["size", "algorithm"])
        .agg(
            rho_mean=("spearman_rho", "mean"),
            pct_rho_below_0_5=("spearman_rho", lambda v: 100 * (v.abs() < 0.5).mean()),
            frac_dom_mean=("fraction_dominated", "mean"),
            pct_frac_above_20=("fraction_dominated", lambda v: 100 * (v >= 0.20).mean()),
            confirms_pct=("confirms_h1", lambda v: 100 * v.mean()),
        )
        .round(3)
    )
    print(agg.to_string())

    print()
    print("=" * 90)
    print("H1 verdict per cell (≥50% of reps confirm = CONFIRMED)")
    print("=" * 90)
    for size in ["small", "medium", "large"]:
        for algo in ["nsga2", "nrga", "nsga3"]:
            sub = res[(res["size"] == size) & (res["algorithm"] == algo)]
            rate = sub["confirms_h1"].mean()
            verdict = "CONFIRMED" if rate >= 0.5 else "REFUTED  "
            print(
                f"  {size:6s} {algo:6s}  {verdict}  "
                f"({rate:.0%} reps; mean ρ={sub['spearman_rho'].mean():+.3f}, "
                f"frac dom={sub['fraction_dominated'].mean():.2f})"
            )

    print()
    print("Pooled across algorithms (90 reps per size):")
    for size in ["small", "medium", "large"]:
        sub = res[res["size"] == size]
        print(
            f"  {size:6s}: H1 confirms in {sub['confirms_h1'].mean():.0%}; "
            f"mean ρ={sub['spearman_rho'].mean():+.3f}, "
            f"mean frac dom={sub['fraction_dominated'].mean():.2f}"
        )

    res.to_csv(out_path, index=False)
    print(f"\nDetail → {out_path}")


if __name__ == "__main__":
    main()
