"""Paper A's H1 re-tested on the exact four-objective front (Paper B audit, correction support).

Paper A (Appl. Sci. 16(15):7581, Section 6) tested H1 on evolutionary fronts: (i) Spearman
rho(TRD, RPD) across each front, pass if |rho| < 0.5; (ii) the fraction of 4-objective
solutions whose 3-objective projection (TIL recomputed via the baseline FIS2 pathway) is
dominated, compared with a random-fusion null TIL = a*TRD + (1-a)*RPD, a ~ U(0, 1).

The published model is separable, so its exact supported front is computable: the exact
greedy at every weight of a simplex lattice on the 4-objective simplex. This script applies
both H1 tests to that front for small / medium / large and, for reference, reports the same
statistics pooled over Paper A's stored evolutionary fronts.
Writes ``experiments/results/audit/paperA_h1_exact.csv``.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.generate_instances import SIZES
from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs
from presidio_vol_assign.allocation.fast_fis import precompute_fis_cache_fast
from presidio_vol_assign.allocation.fis import compute_rws, evaluate_fis2_til
from presidio_vol_assign.allocation.models import (
    AllocationConfig,
    AllocationSolverType,
    Weights,
)
from presidio_vol_assign.allocation.solvers import evaluate_pairs
from presidio_vol_assign.allocation.validation import load_allocation_problem

OUT = Path("experiments/results/audit")
_RNG = np.random.default_rng(20261006)
_NULL_DRAWS = 100


def lattice(dim: int, steps: int) -> list[tuple[float, ...]]:
    """All weight vectors on the regular simplex lattice with `steps` divisions."""

    def rec(remaining: int, parts: int) -> list[tuple[int, ...]]:
        if parts == 1:
            return [(remaining,)]
        return [(i, *rest) for i in range(remaining + 1) for rest in rec(remaining - i, parts - 1)]

    return [tuple(x / steps for x in v) for v in rec(steps, dim)]


def dominated_fraction(points: np.ndarray) -> float:
    """Share of points strictly dominated by another point (Paper A's projection test)."""
    flags = [bool(((points <= p).all(axis=1) & (points < p).any(axis=1)).any()) for p in points]
    return float(np.mean(flags))


def nondominated_unique(points: np.ndarray, pairs: list) -> tuple[np.ndarray, list]:
    keep, seen = [], set()
    for i, p in enumerate(points):
        key = tuple(np.round(p, 9))
        if key in seen:
            continue
        if ((points <= p).all(axis=1) & (points < p).any(axis=1)).any():
            continue
        seen.add(key)
        keep.append(i)
    return points[keep], [pairs[i] for i in keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12, help="lattice divisions per axis")
    parser.add_argument(
        "--paper-a-results",
        default="../presidio-hardened-vol-asssign/experiments/results/h1_h2_h4",
        help="Paper A's stored fronts (pareto_*.csv per run)",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    weights_vs = Weights()
    rows = []
    for size in ("small", "medium", "large"):
        base = Path("experiments/instances") / size
        problem = load_allocation_problem(
            base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=SIZES[size].n_dir
        )
        cfg = AllocationConfig(solver=AllocationSolverType.NSGA2, objectives=4, weights=weights_vs)
        cache = precompute_fis_cache_fast(problem, cfg)
        pts, pairs = [], []
        for w in lattice(4, args.steps):
            p = exact_weighted_sum_pairs(cache, problem.n_dir, 4, w)
            pts.append(evaluate_pairs(p, cache, 4))  # (ULPP, TRD, RPD, CAIL)
            pairs.append(p)
        front, front_pairs = nondominated_unique(np.array(pts), pairs)

        # baseline 3-obj projection: TIL via RWS + FIS2, as in Paper A Section 6.2
        person_ids = [p.person_id for p in problem.people]
        centre_ids = [c.center_id for c in problem.centers]
        til_cache: dict[tuple[int, int], float] = {}

        def til_of(pairs_, til_cache=til_cache, problem=problem):
            vals = []
            for j, i in pairs_:
                if (j, i) not in til_cache:
                    travel = problem.travel[(person_ids[j], centre_ids[i])]
                    til_cache[(j, i)] = evaluate_fis2_til(
                        travel.travel_duration, compute_rws(travel, weights_vs)
                    )
                vals.append(til_cache[(j, i)])
            return float(np.mean(vals))

        til = np.array([til_of(p) for p in front_pairs])
        projected = np.column_stack([front[:, 0], til, front[:, 3]])
        frac = dominated_fraction(projected)
        null = []
        for _ in range(_NULL_DRAWS):
            a = _RNG.uniform()
            null.append(
                dominated_fraction(
                    np.column_stack(
                        [front[:, 0], a * front[:, 1] + (1 - a) * front[:, 2], front[:, 3]]
                    )
                )
            )
        rho, _ = spearmanr(front[:, 1], front[:, 2])

        # Paper A's evolutionary fronts on the same instance, for reference
        moea_rho = []
        for f in glob.glob(f"{args.paper_a_results}/{size}_4obj_*_rep*/pareto_*.csv"):
            r = list(csv.DictReader(open(f)))
            trd = [float(x["mn_trd"]) for x in r]
            rpd = [float(x["mn_rpd"]) for x in r]
            if len(set(trd)) > 1 and len(set(rpd)) > 1:
                moea_rho.append(spearmanr(trd, rpd)[0])
        row = {
            "size": size,
            "lattice_weights": len(pts),
            "exact_front_points": len(front),
            "exact_rho_trd_rpd": float(rho),
            "exact_h1_spearman_pass": bool(abs(rho) < 0.5),
            "exact_projection_dominated": frac,
            "exact_null_mean": float(np.mean(null)),
            "exact_null_p05": float(np.percentile(null, 5)),
            "exact_null_p95": float(np.percentile(null, 95)),
            "paperA_fronts": len(moea_rho),
            "paperA_rho_mean": float(np.mean(moea_rho)) if moea_rho else float("nan"),
            "paperA_share_pass": float(np.mean(np.abs(moea_rho) < 0.5))
            if moea_rho
            else float("nan"),
        }
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    with (OUT / "paperA_h1_exact.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
