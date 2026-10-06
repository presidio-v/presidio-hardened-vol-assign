"""Demo runs for the published relief-allocation model and its repair (Paper B).

The other demo scenarios go through ``engine.run`` and a domain adapter. The
relief-allocation model lives in :mod:`presidio_vol_assign.allocation` with its
own solver and objective cache, so this module drives it directly and emits the
same payload shape the page already consumes (``results`` with ``solver``,
``metrics`` and compactly encoded ``solutions``).

Every run carries an extra result row, ``exact``, next to the evolutionary
fronts — that comparison is the audit the page exists to show:

* **published model** — every objective is a mean of per-person or
  per-(person, centre) scores, so a weighted sum decomposes per person and
  :func:`~presidio_vol_assign.allocation.baselines.exact_weighted_sum_pairs`
  solves it exactly in milliseconds. A simplex-lattice weight sweep yields the
  exact *supported* Pareto front.
* **repaired model** — centre imbalance (CAIL) is read at each centre's
  realised load and capacity is a hard limit, so the model is no longer
  separable. The reference is a mixed-integer program per weight
  (:func:`~presidio_vol_assign.allocation.exact_mip.solve_weighted_mip`) under a
  per-solve time limit, and whether each solve was proven optimal is reported.
  The evolutionary search has no capacity handling, so it runs on the soft
  load-coupled objectives and each of its options reports how many people it
  sends beyond free capacity.

Nothing here touches the filesystem or the network; the whole run is a pure
function of (scenario, knobs, seed, solver settings).
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs
from presidio_vol_assign.allocation.exact_mip import solve_weighted_mip
from presidio_vol_assign.allocation.fast_fis import precompute_fis_cache_fast
from presidio_vol_assign.allocation.load_coupling import overload, precompute_load_coupled_cache
from presidio_vol_assign.allocation.metrics import compute_allocation_metrics
from presidio_vol_assign.allocation.models import (
    AllocationConfig,
    AllocationParetoFront,
    AllocationSolution,
    AllocationSolverType,
)
from presidio_vol_assign.allocation.solvers import FISCache, evaluate_pairs, solve
from presidio_vol_assign.web.runner import LIMITS

OBJECTIVES = 3
"""The three-objective model as published (ULPP, TIL, CAIL)."""

EXACT_LABEL = "exact"

_EPS = 1e-9
"""Objective-space tolerance for duplicate and dominance checks."""

LATTICE_STEPS = 10
"""Simplex-lattice resolution of the published-model sweep: 66 weight vectors
(step 0.1), plus the equal-weight point, which the lattice misses."""

MIP_WEIGHTS: tuple[tuple[float, float, float], ...] = (
    (1 / 3, 1 / 3, 1 / 3),
    (0.8, 0.1, 0.1),
    (0.1, 0.8, 0.1),
    (0.1, 0.1, 0.8),
    (0.45, 0.45, 0.1),
    (0.45, 0.1, 0.45),
    (0.1, 0.45, 0.45),
)
"""Repaired-model sweep: equal weights first (the most informative single point),
then the three near-extremes, then the pairwise compromises. Strictly positive
weights keep every MIP optimum Pareto-optimal rather than weakly so."""

MIP_TIME_LIMIT_SEC = 3.0
"""Per-MIP HiGHS time limit. A solve stopped by it is reported, never hidden."""

MIP_BUDGET_SEC = 12.0
"""Total MIP time per run. Weights not reached within it are reported as skipped,
so the largest instance stays well inside the runner's wall-clock timeout."""

EXACT_TO_OPTIMALITY_ENV = "PVA_DEMO_EXACT_TO_OPTIMALITY"
"""Set to "1" by the static build. A wall-clock limit makes a stopped MIP depend on
machine speed, so pre-built pages would differ between builds; the static build
instead runs every MIP to proven optimality, under a generous safety cap."""

_STATIC_MIP_TIME_LIMIT_SEC = 300.0
_STATIC_MIP_BUDGET_SEC = 1800.0


def _mip_limits() -> tuple[float, float]:
    """(per-MIP limit, per-run budget): live-server limits, or the static-build caps."""
    if os.environ.get(EXACT_TO_OPTIMALITY_ENV) == "1":
        return _STATIC_MIP_TIME_LIMIT_SEC, _STATIC_MIP_BUDGET_SEC
    return MIP_TIME_LIMIT_SEC, MIP_BUDGET_SEC


HV_BOX = 100.0
"""Every FIS output lies in [0, 100]. Hypervolume is reported as the fraction of
the [0, 100]^3 objective box dominated by the front (reference point 100 on
each axis), so every row of one run is on the same, comparable scale."""


@dataclass
class AllocationPrep:
    """Per-instance precompute, memoised by the runner across solver settings.

    Attributes:
        cache: The objective cache every solution is scored against — static
            (published) or load-coupled (repaired).
        capacity: Centre capacities in persons (repaired only).
        limits: Whole free places per centre, the hard limits (repaired only).
        exact: The exact reference row, filled on first use. It depends only on
            the instance, so changing the algorithm or generation count reuses it.
    """

    cache: FISCache
    capacity: np.ndarray | None = None
    limits: np.ndarray | None = None
    exact: dict[str, Any] | None = None


def prepare_allocation(scenario: Any, instance: Any) -> AllocationPrep:
    """Build the objective cache for *instance* (the expensive, memoised step)."""
    problem = instance.problem
    config = AllocationConfig(solver=AllocationSolverType.NSGA2, objectives=OBJECTIVES)
    static = precompute_fis_cache_fast(problem, config)
    if not scenario.hard_capacity:
        return AllocationPrep(cache=static)
    capacity = instance.extras["capacity"]
    cache = precompute_load_coupled_cache(problem, config, capacity, static=static)
    return AllocationPrep(cache=cache, capacity=capacity, limits=instance.extras["limits"])


def solve_allocation(
    request: Any, scenario: Any, instance: Any, prep: AllocationPrep
) -> list[dict[str, Any]]:
    """Run the requested MOEA(s), then append the exact reference row.

    The MOEA rows come first because the page's map and trade-off slider
    start from the first result.
    """
    problem = instance.problem
    person_index = {p.person_id: j for j, p in enumerate(problem.people)}
    centre_index = {c.center_id: i for i, c in enumerate(problem.centers)}

    if request.solver == "both":
        solver_types = [AllocationSolverType.NSGA2, AllocationSolverType.NRGA]
    else:
        solver_types = [AllocationSolverType(request.solver)]

    results = []
    for solver_type in solver_types:
        config = AllocationConfig(
            solver=solver_type,
            objectives=OBJECTIVES,
            pop_size=request.pop_size,
            generations=request.generations,
            seed=request.seed,
            output_dir=".",  # unused: the demo never writes result files
        )
        front = solve(problem, config, cache=prep.cache)
        points = [
            (
                tuple(sol.fitness),
                [(person_index[a.person_id], centre_index[a.center_id]) for a in sol.allocations],
            )
            for sol in front.solutions
        ]
        results.append(
            _result_row(
                solver_type.value,
                front,
                points,
                problem,
                prep,
                note=_moea_note(solver_type.value, points, problem, prep),
            )
        )

    if prep.exact is None:
        prep.exact = (
            _exact_repaired(problem, prep)
            if prep.limits is not None
            else (_exact_published(problem, prep))
        )
    results.append(copy.deepcopy(prep.exact))
    return results


# ---------------------------------------------------------------------------
# Exact references
# ---------------------------------------------------------------------------


def simplex_lattice(steps: int) -> list[tuple[float, float, float]]:
    """All 3-weight vectors on the simplex with spacing ``1/steps``, plus equal weights."""
    weights = [
        (i / steps, j / steps, (steps - i - j) / steps)
        for i in range(steps + 1)
        for j in range(steps + 1 - i)
    ]
    weights.append((1 / 3, 1 / 3, 1 / 3))
    return weights


def _exact_published(problem: Any, prep: AllocationPrep) -> dict[str, Any]:
    """Exact supported front of the separable model by a weighted-sum sweep."""
    weights = simplex_lattice(LATTICE_STEPS)
    start = time.perf_counter()
    candidates = []
    for w in weights:
        pairs = exact_weighted_sum_pairs(prep.cache, problem.n_dir, OBJECTIVES, w)
        candidates.append((evaluate_pairs(pairs, prep.cache, OBJECTIVES), pairs, True))
    seconds = time.perf_counter() - start

    points = _non_dominated(candidates)
    front = _as_front(points, seconds)
    note = (
        f"EXACT: the best trade-offs of the published model, computed directly. Each "
        f"of {len(weights)} weightings of the three goals has a provably optimal "
        f"answer because the objectives decompose person by person; "
        f"{len(points)} distinct non-dominated options remain."
    )
    row = _result_row(EXACT_LABEL, front, points, problem, prep, note=note)
    row["exact"] = {
        "method": "separable-weighted-sum",
        "weights": len(weights),
        "solved": len(weights),
        "provenOptimal": len(weights),
        "maxGap": 0.0,
        "skipped": 0,
    }
    return row


def _exact_repaired(problem: Any, prep: AllocationPrep) -> dict[str, Any]:
    """Weighted-sum MIP sweep of the load-coupled model under hard capacity."""
    per_mip, budget = _mip_limits()
    start = time.perf_counter()
    candidates = []
    proven = 0
    max_gap = 0.0
    failed = 0
    skipped = 0
    for w in MIP_WEIGHTS:
        remaining = budget - (time.perf_counter() - start)
        if remaining <= 0.1:
            skipped += 1
            continue
        try:
            res = solve_weighted_mip(
                prep.cache,
                problem.n_dir,
                OBJECTIVES,
                w,
                time_limit=min(per_mip, remaining),
                max_load=prep.limits,
            )
        except RuntimeError:
            # No incumbent within the time limit: count it, never fake a point.
            failed += 1
            continue
        proven += int(res.optimal)
        max_gap = max(max_gap, res.gap)
        candidates.append(
            (evaluate_pairs(res.pairs, prep.cache, OBJECTIVES), res.pairs, bool(res.optimal))
        )
    seconds = time.perf_counter() - start

    points = _non_dominated(candidates)
    front = _as_front(points, seconds)
    solved = len(candidates)
    status = (
        "all proven optimal"
        if proven == solved and solved == len(MIP_WEIGHTS)
        else f"{proven} of {len(MIP_WEIGHTS)} proven optimal"
        + (f", {skipped} skipped for time" if skipped else "")
        + (f", {failed} without a solution" if failed else "")
    )
    note = (
        f"EXACT: a mixed-integer program solved at {len(MIP_WEIGHTS)} weightings of the "
        f"three goals, never exceeding any centre's free places ({status}; largest "
        f"optimality gap {max_gap:.2%}). These options are always physically feasible."
    )
    row = _result_row(EXACT_LABEL, front, points, problem, prep, note=note)
    row["exact"] = {
        "method": "mip-weighted-sum",
        "weights": len(MIP_WEIGHTS),
        "solved": solved,
        "provenOptimal": proven,
        "maxGap": round(max_gap, 6),
        "skipped": skipped,
        "failed": failed,
        "timeLimitSec": per_mip,
    }
    return row


def _non_dominated(
    candidates: list[tuple[tuple[float, ...], list[tuple[int, int]], bool]],
) -> list[tuple[tuple[float, ...], list[tuple[int, int]], bool]]:
    """Distinct objective vectors that no other candidate dominates, sorted."""
    items: list[tuple[tuple[float, ...], list[tuple[int, int]], bool]] = []
    for fit, pairs, optimal in candidates:
        if not any(all(abs(a - b) <= _EPS for a, b in zip(fit, seen)) for seen, _, _ in items):
            items.append((tuple(fit), pairs, optimal))
    kept = []
    for fit, pairs, optimal in items:
        # Tolerant comparison: the same set of people summed in a different
        # order gives means that differ in the last bits, which must not keep a
        # dominated point alive.
        dominated = any(
            all(o <= f + _EPS for o, f in zip(other, fit, strict=True))
            and any(o < f - _EPS for o, f in zip(other, fit, strict=True))
            for other, _, _ in items
        )
        if not dominated:
            kept.append((fit, pairs, optimal))
    kept.sort(key=lambda item: item[0])
    return kept


def _as_front(
    points: list[tuple[tuple[float, ...], list[tuple[int, int]], bool]], seconds: float
) -> AllocationParetoFront:
    """Wrap exact points so the allocation metrics module can score them.

    The metrics read only each solution's objective vector and the front's
    timing; the ``solver`` field is a placeholder required by the dataclass.
    """
    solutions = [
        AllocationSolution(
            allocations=[],
            objectives_count=OBJECTIVES,
            mn_ulpp=fit[0],
            mn_til=fit[1],
            mn_cail=fit[2],
        )
        for fit, _, _ in points
    ]
    return AllocationParetoFront(
        solver=AllocationSolverType.NSGA2,
        objectives_count=OBJECTIVES,
        solutions=solutions,
        cpu_time_sec=seconds,
    )


# ---------------------------------------------------------------------------
# Payload encoding
# ---------------------------------------------------------------------------


def _moea_note(
    label: str,
    points: list[tuple[tuple[float, ...], list[tuple[int, int]]]],
    problem: Any,
    prep: AllocationPrep,
) -> str:
    if prep.capacity is None:
        return ""
    over = sum(1 for _, pairs in points if overload(pairs, problem, prep.capacity) > 0)
    return (
        f"{label.upper()}: {over} of {len(points)} options send people beyond a centre's "
        "free places. The evolutionary search has no capacity handling."
    )


def _result_row(
    label: str,
    front: AllocationParetoFront,
    points: list[tuple],
    problem: Any,
    prep: AllocationPrep,
    *,
    note: str,
) -> dict[str, Any]:
    """One ``results[]`` entry in the shape app.js consumes."""
    metrics = compute_allocation_metrics(front)
    solutions = _encode(points, problem, prep)
    row: dict[str, Any] = {
        "solver": label,
        "metrics": {
            "nns": metrics.nns,
            "hv": round(metrics.hv / HV_BOX**OBJECTIVES, 6),
            "sm": round(metrics.sm, 6),
            "mid": round(metrics.mid, 6),
            "cpuTimeSec": round(metrics.cpu_time_sec, 3),
        },
        "solutions": solutions,
        "note": note,
    }
    if prep.capacity is not None:
        # Counted over the full front, not just the subsample sent to the page.
        row["metrics"]["overloaded"] = sum(
            1 for point in points if overload(point[1], problem, prep.capacity) > 0
        )
    return row


def _encode(points: list[tuple], problem: Any, prep: AllocationPrep) -> list[dict[str, Any]]:
    """Objectives plus one centre index per person (-1 = not directed this round)."""
    if len(points) > LIMITS.max_solutions_returned:
        stride = len(points) / LIMITS.max_solutions_returned
        points = [points[int(i * stride)] for i in range(LIMITS.max_solutions_returned)]

    encoded = []
    for point in points:
        fit, pairs = point[0], point[1]
        alloc = [-1] * problem.n_people
        for person, centre in pairs:
            alloc[person] = centre
        entry: dict[str, Any] = {
            "objectives": [round(float(v), 6) for v in fit],
            "alloc": alloc,
        }
        if prep.capacity is not None:
            entry["overload"] = round(overload(pairs, problem, prep.capacity), 3)
        if len(point) > 2:
            entry["optimal"] = bool(point[2])
        encoded.append(entry)
    return encoded
