"""Near-optimal multiplicity and the deterministic tie-break (Paper B revision, R2.5 / R3).

Part 1. For each reference instance and model (published separable; repaired soft
kappa=2; repaired hard kappa_f=1.5) and each tolerance epsilon, find up to five allocations
within epsilon of the exact optimum, each maximally different from the optimum and from the
ones found before (by pairs: maximal churn; by directed set: minimal Jaccard). The largest
epsilon is the evolutionary system's own quality deficit at the operational budget, so the
seed floor can be read against the multiplicity of the problem at the same quality.
Writes ``experiments/results/multiplicity/alternatives.csv`` and ``summary.csv``.

Part 2. The tie-break: the same instance presented to the solver with the people in ten
random orders. The plain MIP returns whichever optimal vertex it reaches; the canonical
optimum (a fixed seeded key per (person, centre) pair, second lexicographic stage) must
return one allocation for all orders. Writes ``tiebreak.csv``.
"""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path

import numpy as np

from experiments.generate_instances import SIZES
from experiments.run_turbulence import OBJECTIVES, _config
from presidio_vol_assign.allocation import multiplicity as mu
from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs
from presidio_vol_assign.allocation.exact_mip import solve_weighted_mip
from presidio_vol_assign.allocation.fast_fis import precompute_fis_cache_fast
from presidio_vol_assign.allocation.load_coupling import (
    capacities_from_free_factor,
    hard_load_limits,
    precompute_load_coupled_cache,
)
from presidio_vol_assign.allocation.solvers import FISCache, evaluate_pairs
from presidio_vol_assign.allocation.validation import load_allocation_problem

OUT = Path("experiments/results/multiplicity")
EPSILONS = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)
# The evolutionary system's best-point deficit at 150 generations (paper Table 2).
MOEA_DEFICIT = {"small": 17.8, "large": 33.3}
K_ALTERNATIVES = 5
TIME_LIMIT = 60.0
N_ORDERS = 10


def _problem(size: str):
    base = Path("experiments/instances") / size
    return load_allocation_problem(
        base / "people.csv", base / "centers.csv", base / "travel.csv", n_dir=SIZES[size].n_dir
    )


def _models(problem):
    """(name, cache, max_load) for the published model and the two repaired variants."""
    static = precompute_fis_cache_fast(problem, _config(100, 150))
    soft = precompute_load_coupled_cache(problem, _config(100, 150), 2.0, static=static)
    cap = capacities_from_free_factor(problem, 1.5)
    hard = precompute_load_coupled_cache(problem, _config(100, 150), cap, static=static)
    return [
        ("published", static, None),
        ("soft_k2", soft, None),
        ("hard_kf1.5", hard, hard_load_limits(problem, cap)),
    ]


def _optimum(cache: FISCache, problem, max_load):
    if cache.cail_load is None:
        pairs = exact_weighted_sum_pairs(cache, problem.n_dir, OBJECTIVES)
    else:
        r = solve_weighted_mip(cache, problem.n_dir, OBJECTIVES, time_limit=120, max_load=max_load)
        if not r.optimal:
            raise RuntimeError("reference MIP not proven optimal")
        pairs = r.pairs
    return pairs, float(np.sum(evaluate_pairs(pairs, cache, OBJECTIVES)))


def part1() -> None:
    alt_rows, summary_rows = [], []
    for size in ("small", "large"):
        problem = _problem(size)
        for name, cache, max_load in _models(problem):
            reference, z_star = _optimum(cache, problem, max_load)
            model = mu.model_for(cache, problem.n_dir, OBJECTIVES, max_load=max_load)
            for eps in (*EPSILONS, MOEA_DEFICIT[size]):
                for mode in ("pairs", "directed"):
                    alts = mu.near_optimal_alternatives(
                        model,
                        problem.n_dir,
                        reference,
                        z_star,
                        eps,
                        k=K_ALTERNATIVES,
                        mode=mode,
                        time_limit=TIME_LIMIT,
                    )
                    pool = [reference] + [a.pairs for a in alts]
                    for idx, a in enumerate(alts, start=1):
                        alt_rows.append(
                            {
                                "size": size,
                                "model": name,
                                "epsilon": eps,
                                "mode": mode,
                                "alternative": idx,
                                "delta_quality": float(
                                    np.sum(evaluate_pairs(a.pairs, cache, OBJECTIVES))
                                )
                                - z_star,
                                "churn_vs_optimum": mu.churn(reference, a.pairs),
                                "jaccard_vs_optimum": mu.directed_jaccard(reference, a.pairs),
                                "optimal": a.optimal,
                                "gap": a.gap,
                                "seconds": a.seconds,
                            }
                        )
                    pairs = [(x, y) for x in range(len(pool)) for y in range(x + 1, len(pool))]
                    churns = [mu.churn(pool[x], pool[y]) for x, y in pairs]
                    jacc = [mu.directed_jaccard(pool[x], pool[y]) for x, y in pairs]
                    row = {
                        "size": size,
                        "model": name,
                        "z_star": z_star,
                        "epsilon": eps,
                        "epsilon_rel": eps / z_star,
                        "mode": mode,
                        "found": len(alts),
                        "max_churn_vs_optimum": max(
                            (mu.churn(reference, a.pairs) for a in alts), default=0.0
                        ),
                        "min_jaccard_vs_optimum": min(
                            (mu.directed_jaccard(reference, a.pairs) for a in alts), default=1.0
                        ),
                        "mean_pairwise_churn": float(np.mean(churns)) if churns else 0.0,
                        "mean_pairwise_jaccard": float(np.mean(jacc)) if jacc else 1.0,
                        "all_proven": all(a.optimal for a in alts),
                    }
                    summary_rows.append(row)
                    print(row, flush=True)
    _write(OUT / "alternatives.csv", alt_rows)
    _write(OUT / "summary.csv", summary_rows)


def _permuted(cache: FISCache, perm: np.ndarray) -> FISCache:
    """The same cache with people presented in the order ``perm``."""
    fields = {f.name: getattr(cache, f.name) for f in dataclasses.fields(cache)}
    return FISCache(**{k: (None if v is None else np.asarray(v)[perm]) for k, v in fields.items()})


def part2() -> None:
    rows = []
    for size in ("small", "large"):
        problem = _problem(size)
        for name, cache, max_load in _models(problem):
            _, z_star = _optimum(cache, problem, max_load)
            m, n = cache.til.shape
            key = mu.tie_break_key(m * n).reshape(m, n)
            plain, canon = set(), set()
            for order in range(N_ORDERS):
                perm = np.random.Generator(np.random.PCG64(1000 + order)).permutation(m)
                pc = _permuted(cache, perm)
                model = mu.model_for(pc, problem.n_dir, OBJECTIVES, max_load=max_load)
                if pc.cail_load is None:
                    p_pairs = exact_weighted_sum_pairs(pc, problem.n_dir, OBJECTIVES)
                else:
                    p_pairs = solve_weighted_mip(
                        pc, problem.n_dir, OBJECTIVES, time_limit=120, max_load=max_load
                    ).pairs
                c = mu.canonical_optimum(
                    model, problem.n_dir, z_star, key=key[perm].ravel(), time_limit=TIME_LIMIT
                )
                back = lambda pairs: tuple(sorted((int(perm[j]), i) for j, i in pairs))  # noqa: E731
                plain.add(back(p_pairs))
                canon.add(back(c.pairs))
                if not math.isclose(c.objective, z_star, rel_tol=0, abs_tol=1e-5):
                    raise RuntimeError("canonical optimum left the optimal face")
            row = {
                "size": size,
                "model": name,
                "orders": N_ORDERS,
                "distinct_plain_optima": len(plain),
                "distinct_canonical": len(canon),
            }
            rows.append(row)
            print(row, flush=True)
    _write(OUT / "tiebreak.csv", rows)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    part1()
    part2()
