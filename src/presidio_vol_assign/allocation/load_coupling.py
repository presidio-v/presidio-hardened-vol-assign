"""Load-coupled CAIL (Paper B): make the Redundancy objective see the load it creates.

In the published model (Paper A, Eq. obj-cail) every CAIL_{j,i} is evaluated at centre i's
*input* occupancy COR_i, which the allocation never changes. All objectives are then means
of static per-person or per-pair scores, the model is separable, and CAIL cannot penalise
piling people onto one centre — although the model's prose says it does. This module adds
the minimal change that makes that prose true: occupancy becomes endogenous,

    COR'_i(l) = min(100, COR_i + 100 * l / cap_i),

where ``l`` is the number of people the decision sends to centre i and ``cap_i`` its total
capacity in persons. CAIL_{j,i} is then FIS3(COR'_i(l_i), RDR_i, TD_{j,i}): same rule base,
same per-(person, centre) form, same travel-duration input. Capacity is soft (the clip at
100 makes the objective indifferent beyond full); ``overload`` reports how far a decision
exceeds free capacity so that physically impossible allocations stay visible.

Instances carry no capacity field, so ``cap_i = ceil(kappa * n_dir / n_centres)`` is derived
from a stated design parameter. ``kappa = inf`` gives zero coupling and reproduces the
separable model exactly — the knob that links the audit to the fix.
"""

from __future__ import annotations

import math

import numpy as np

from presidio_vol_assign.allocation.fast_fis import compiled_fis3, precompute_fis_cache_fast
from presidio_vol_assign.allocation.models import AllocationConfig, AllocationProblem
from presidio_vol_assign.allocation.solvers import FISCache


def centre_capacities(problem: AllocationProblem, kappa: float) -> np.ndarray:
    """Capacity of each centre in persons: ``ceil(kappa * n_dir / n_centres)``; inf if uncoupled."""
    if not kappa > 0:
        raise ValueError("kappa must be positive (use math.inf for the separable model)")
    n = problem.n_centers
    if math.isinf(kappa):
        return np.full(n, np.inf)
    return np.full(n, float(math.ceil(kappa * problem.n_dir / n)))


def capacities_from_free_factor(problem: AllocationProblem, kappa_f: float) -> np.ndarray:
    """Equal centre size c chosen so free places sum to ``kappa_f * n_dir``.

    Free places at centre i are c * (1 - COR_i / 100); this pins total slack rather than
    total size, which is what decides whether a hard capacity limit is feasible at all.
    """
    if not kappa_f > 0 or math.isinf(kappa_f):
        raise ValueError("kappa_f must be a positive finite number")
    cor = np.array([c.center_occupancy_rate for c in problem.centers], dtype=float)
    free_share = (1.0 - np.clip(cor, 0.0, 100.0) / 100.0).sum()
    if free_share <= 0:
        raise ValueError("every centre is full; no free capacity to scale")
    return np.full(problem.n_centers, kappa_f * problem.n_dir / free_share)


def hard_load_limits(problem: AllocationProblem, capacity: np.ndarray) -> np.ndarray:
    """Whole persons each centre can still take: floor(cap_i * (1 - COR_i / 100))."""
    cor = np.array([c.center_occupancy_rate for c in problem.centers], dtype=float)
    return np.floor(capacity * (1.0 - np.clip(cor, 0.0, 100.0) / 100.0) + 1e-9).astype(int)


def post_occupancy(cor: np.ndarray, capacity: np.ndarray, loads: np.ndarray) -> np.ndarray:
    """COR' for each centre given its load; broadcasts over a trailing load axis."""
    with np.errstate(divide="ignore"):
        added = np.where(np.isinf(capacity), 0.0, 100.0 * loads / capacity)
    return np.minimum(100.0, cor + added)


def free_capacity(problem: AllocationProblem, kappa: float | np.ndarray) -> np.ndarray:
    """Places still free at each centre before the decision: cap_i * (1 - COR_i / 100).

    ``kappa`` is either the size factor or an explicit capacity array.
    """
    cap = _as_capacity(problem, kappa)
    cor = np.array([c.center_occupancy_rate for c in problem.centers], dtype=float)
    return cap * (1.0 - np.clip(cor, 0.0, 100.0) / 100.0)


def _as_capacity(problem: AllocationProblem, kappa: float | np.ndarray) -> np.ndarray:
    if isinstance(kappa, np.ndarray):
        if kappa.shape != (problem.n_centers,) or (kappa <= 0).any():
            raise ValueError("capacity must be one positive value per centre")
        return kappa.astype(float)
    return centre_capacities(problem, kappa)


def overload(
    pairs: list[tuple[int, int]], problem: AllocationProblem, kappa: float | np.ndarray
) -> float:
    """Persons sent beyond free capacity, summed over centres (0 = physically feasible)."""
    loads = np.bincount([c for _, c in pairs], minlength=problem.n_centers)
    return float(np.maximum(0.0, loads - free_capacity(problem, kappa)).sum())


def precompute_load_coupled_cache(
    problem: AllocationProblem,
    config: AllocationConfig,
    kappa: float | np.ndarray,
    static: FISCache | None = None,
) -> FISCache:
    """Static FIS cache plus ``cail_load[j, i, l]`` for every load l = 0..n_dir.

    Pass ``static`` to reuse an existing cache (perturbations of person-only fields leave
    every CAIL input unchanged). ``kappa`` is the size factor or an explicit capacity array
    (e.g. from ``capacities_from_free_factor``). The load table goes through the compiled
    FIS3, which is pinned to scikit-fuzzy by tests; at l = 0 it equals the static CAIL.
    """
    base = static if static is not None else precompute_fis_cache_fast(problem, config)
    m, n, n_dir = problem.n_people, problem.n_centers, problem.n_dir
    cap = _as_capacity(problem, kappa)
    cor = np.array([c.center_occupancy_rate for c in problem.centers], dtype=float)
    rdr = np.array([c.resource_depletion_rate for c in problem.centers], dtype=float)
    td = np.zeros((m, n))
    person_idx = {p.person_id: j for j, p in enumerate(problem.people)}
    centre_idx = {c.center_id: i for i, c in enumerate(problem.centers)}
    for (pid, cid), travel in problem.travel.items():
        td[person_idx[pid], centre_idx[cid]] = travel.travel_duration

    loads = np.arange(n_dir + 1, dtype=float)
    cor_post = post_occupancy(cor[:, None], cap[:, None], loads[None, :])  # (n, L+1)
    fis3 = compiled_fis3()
    table = np.empty((m, n, n_dir + 1))
    for i in range(n):
        # Distinct COR' values only: with finite capacity the clip at 100 repeats the top level.
        levels, inverse = np.unique(cor_post[i], return_inverse=True)
        cor_grid, td_grid = np.meshgrid(levels, td[:, i], indexing="xy")  # (m, n_levels)
        values = fis3.batch(cor_grid.ravel(), np.full(cor_grid.size, rdr[i]), td_grid.ravel())
        table[:, i, :] = values.reshape(m, levels.size)[:, inverse]
    return FISCache(
        ulpp=base.ulpp,
        til=base.til,
        trd=base.trd,
        rpd=base.rpd,
        cail=base.cail,
        cail_load=table,
    )
