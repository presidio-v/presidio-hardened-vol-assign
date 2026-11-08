"""Near-optimal multiplicity: how different can allocations of almost equal quality be?

The seed floor measures how much two committed decisions of the stochastic system differ.
That spread has two possible sources: the solver (it stops far from the optimum, and where
it stops depends on the seed) and the problem (many quite different allocations score
almost the same). This module measures the second one directly. For a tolerance ``epsilon``
in objective-sum units it finds allocations whose weighted objective sum is at most
``z_star + epsilon`` and that are maximally different from a reference allocation and
from each other (modelling to generate alternatives: add the quality bound as a
constraint and minimise overlap with the pool found so far).

Both models are covered. The published, separable model is a MIP over ``x`` alone; the
load-coupled model reuses ``build_weighted_mip`` with ``x`` declared binary, because the
quality bound and the overlap objective break the transportation structure that keeps a
continuous ``x`` integral.

``canonical_optimum`` is the deterministic tie-break: among allocations within a tight
tolerance of the optimum it returns the one with the smallest index weight, so the
committed decision no longer depends on which optimal vertex the solver happens to reach.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack

from presidio_vol_assign.allocation.exact_mip import MipModel, build_weighted_mip
from presidio_vol_assign.allocation.solvers import FISCache

# Slack on the quality bound so a solver tolerance cannot exclude the optimum itself.
_BOUND_SLACK = 1e-7


@dataclass(frozen=True)
class Alternative:
    pairs: list[tuple[int, int]]
    objective: float  # weighted objective sum of this allocation
    overlap: float  # value of the overlap objective that selected it
    optimal: bool  # the overlap objective was solved to proven optimality
    gap: float
    seconds: float


def build_static_mip(
    cache: FISCache,
    n_dir: int,
    objectives: int,
    weights: tuple[float, ...] | None = None,
) -> MipModel:
    """The published (separable) model as a MIP over ``x`` only, same layout as the MIP."""
    w = tuple(weights) if weights is not None else (1.0,) * objectives
    if len(w) != objectives or any(x < 0 for x in w):
        raise ValueError("weights must be one non-negative value per objective")
    if cache.ulpp_objective != "directed":  # the per-person cost below is the published form
        raise NotImplementedError("exact deciders implement the published ULPP objective only")
    m, n = cache.til.shape
    if not 0 < n_dir <= m:
        raise ValueError("n_dir must be in (0, number of people]")
    pair_terms = (cache.trd, cache.rpd, cache.cail) if objectives == 4 else (cache.til, cache.cail)
    pair_cost = sum(wk * term for wk, term in zip(w[1:], pair_terms, strict=True))
    cost = ((w[0] * cache.ulpp)[:, None] + pair_cost).ravel() / n_dir

    rows = np.repeat(np.arange(m), n)
    cols = np.arange(m * n)
    per_person = coo_matrix((np.ones(m * n), (rows, cols)), shape=(m, m * n))
    total = csr_matrix(np.ones((1, m * n)))
    a = vstack([per_person.tocsr(), total]).tocsr()
    lb = np.concatenate([np.full(m, -np.inf), [n_dir]])
    ub = np.concatenate([np.ones(m), [n_dir]])
    return MipModel(cost=cost, a=a, lb=lb, ub=ub, integrality=np.ones(m * n), m=m, n=n)


def model_for(
    cache: FISCache,
    n_dir: int,
    objectives: int,
    weights: tuple[float, ...] | None = None,
    max_load: np.ndarray | None = None,
) -> MipModel:
    """The static model when the cache has no load table, else the load-coupled MIP."""
    if cache.cail_load is None:
        if max_load is not None:
            raise ValueError("hard capacity needs a load-coupled cache")
        return build_static_mip(cache, n_dir, objectives, weights)
    return build_weighted_mip(cache, n_dir, objectives, weights, max_load)


def _pairs_from(x: np.ndarray, model: MipModel, n_dir: int) -> list[tuple[int, int]]:
    chosen = np.argwhere(x[: model.n_x].reshape(model.m, model.n) > 0.5)
    pairs = [(int(j), int(i)) for j, i in chosen]
    if len(pairs) != n_dir or len({j for j, _ in pairs}) != n_dir:
        raise RuntimeError("solution is not an integral allocation of n_dir people")
    return pairs


def _solve_bounded(
    model: MipModel,
    objective: np.ndarray,
    bound: float,
    time_limit: float,
) -> tuple[object, float]:
    """Minimise ``objective`` subject to the model and ``cost @ v <= bound``; x binary."""
    integrality = model.integrality.copy()
    integrality[: model.n_x] = 1
    quality_row = csr_matrix(model.cost[None, :])
    a = vstack([model.a, quality_row]).tocsr()
    lb = np.concatenate([model.lb, [-np.inf]])
    ub = np.concatenate([model.ub, [bound]])
    start = time.perf_counter()
    res = milp(
        objective,
        constraints=LinearConstraint(a, lb, ub),
        integrality=integrality,
        bounds=Bounds(0.0, 1.0),
        options={"time_limit": time_limit, "disp": False},
    )
    return res, time.perf_counter() - start


def _bound(z_star: float, epsilon: float) -> float:
    return z_star + epsilon + _BOUND_SLACK * max(1.0, abs(z_star))


def near_optimal_alternatives(
    model: MipModel,
    n_dir: int,
    reference: list[tuple[int, int]],
    z_star: float,
    epsilon: float,
    k: int = 5,
    mode: str = "pairs",
    time_limit: float = 60.0,
) -> list[Alternative]:
    """Up to ``k`` allocations within ``epsilon`` of ``z_star``, each maximally different.

    ``mode="pairs"`` minimises the number of (person, centre) pairs shared with the pool,
    i.e. maximises allocation churn; ``mode="directed"`` minimises the number of directed
    people shared with the pool, i.e. minimises the Jaccard index of the directed sets.
    The pool starts with ``reference`` and grows with every alternative found. A run that
    stops at the time limit still returns a feasible allocation, so the churn it reports
    is a lower bound on the maximum; ``optimal`` says which.
    """
    if mode not in ("pairs", "directed"):
        raise ValueError("mode must be 'pairs' or 'directed'")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    pool = [list(reference)]
    found: list[Alternative] = []
    bound = _bound(z_star, epsilon)
    for _ in range(k):
        weight = np.zeros(model.cost.size)
        x_weight = weight[: model.n_x].reshape(model.m, model.n)
        for alloc in pool:
            for j, i in alloc:
                if mode == "pairs":
                    x_weight[j, i] += 1.0
                else:
                    x_weight[j, :] += 1.0
        res, seconds = _solve_bounded(model, weight, bound, time_limit)
        if res.x is None:
            break
        pairs = _pairs_from(res.x, model, n_dir)
        found.append(
            Alternative(
                pairs=pairs,
                objective=float(model.cost @ res.x),
                overlap=float(res.fun),
                optimal=res.status == 0,
                gap=float(getattr(res, "mip_gap", 0.0) or 0.0),
                seconds=seconds,
            )
        )
        pool.append(pairs)
    return found


def tie_break_key(n_pairs: int, seed: int = 0) -> np.ndarray:
    """A fixed weight in [1, 2) per (person, centre) pair.

    PCG64 with a stated seed yields the same key on every platform, so the key can be
    published with the decision. Index-based weights would not do: swapping two people
    between two centres leaves any sum of row and column indices unchanged.
    """
    return 1.0 + np.random.Generator(np.random.PCG64(seed)).random(n_pairs)


def canonical_optimum(
    model: MipModel,
    n_dir: int,
    z_star: float,
    tolerance: float = 1e-6,
    time_limit: float = 60.0,
    seed: int = 0,
    key: np.ndarray | None = None,
) -> Alternative:
    """Deterministic tie-break among (near-)optimal allocations.

    Second stage of a lexicographic solve: keep the quality within ``tolerance`` of
    ``z_star`` and minimise the published ``tie_break_key``. Allocations of equal quality
    then differ in key weight, so the committed decision no longer depends on which optimal
    vertex the solver reaches first, nor on the platform it runs on. Pass ``key`` (one
    weight per x, flattened person-major) to keep the key attached to each (person, centre)
    pair when the same instance is presented in a different order.
    """
    weight = np.zeros(model.cost.size)
    weight[: model.n_x] = tie_break_key(model.n_x, seed) if key is None else key
    res, seconds = _solve_bounded(model, weight, _bound(z_star, tolerance), time_limit)
    if res.x is None:
        raise RuntimeError(f"tie-break stage returned no solution: {res.message}")
    return Alternative(
        pairs=_pairs_from(res.x, model, n_dir),
        objective=float(model.cost @ res.x),
        overlap=float(res.fun),
        optimal=res.status == 0,
        gap=float(getattr(res, "mip_gap", 0.0) or 0.0),
        seconds=seconds,
    )


def churn(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    """Fraction of ``a``'s directed people whose assignment differs in ``b``.

    With both allocations directing exactly ``n_dir`` people this is symmetric and equals
    ``1 - |a & b| / n_dir`` over (person, centre) pairs.
    """
    mapping = dict(b)
    return sum(1 for j, i in a if mapping.get(j) != i) / len(a)


def directed_jaccard(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    da, db = {j for j, _ in a}, {j for j, _ in b}
    return len(da & db) / len(da | db)
