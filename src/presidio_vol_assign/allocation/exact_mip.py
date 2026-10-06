"""Exact weighted-sum optimum of the load-coupled model by mixed-integer programming.

With load-coupled CAIL (``allocation.load_coupling``) the model is no longer separable: a
person's CAIL depends on how many others the decision sends to the same centre. The exact
reference is a MIP over

* ``y[i, l]`` (binary): centre i receives load level l — only these need to be integer;
* ``x[j, i]`` in [0, 1]: person j is directed to centre i;
* ``z[j, i, l]`` in [0, 1]: person j is at centre i *and* centre i is at level l.

For fixed ``y`` the remainder is a transportation problem (totally unimodular), so the
continuous ``x``/``z`` come out integral at any integral ``y``. Load levels at which the
centre's post-allocation occupancy is already clipped at 100 share one CAIL value and are
merged into a single *saturated* level covering [l_sat, n_dir]; that keeps the model small
without approximation. Solved with HiGHS via ``scipy.optimize.milp`` under a time limit;
the MIP gap is always returned, so a stopped run is reported as such, never as optimal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from presidio_vol_assign.allocation.solvers import FISCache


@dataclass(frozen=True)
class MipResult:
    pairs: list[tuple[int, int]]
    objective: float  # weighted objective sum of the returned allocation
    gap: float  # relative MIP gap reported by HiGHS (0.0 = proven optimal)
    optimal: bool
    seconds: float


def _levels(cail_load_i: np.ndarray) -> list[tuple[int, int]]:
    """Load ranges [lo, hi] per CAIL level at one centre; the tail of equal columns merges."""
    n_levels = cail_load_i.shape[1]
    last = n_levels - 1
    while last > 0 and np.array_equal(cail_load_i[:, last - 1], cail_load_i[:, last]):
        last -= 1
    return [(lvl, lvl) for lvl in range(last)] + [(last, n_levels - 1)]


def solve_weighted_mip(
    cache: FISCache,
    n_dir: int,
    objectives: int,
    weights: tuple[float, ...] | None = None,
    time_limit: float = 60.0,
    max_load: np.ndarray | None = None,
) -> MipResult:
    """Exact (or gap-bounded) minimiser of the weighted objective sum, load-coupled CAIL.

    ``max_load`` (persons per centre) turns capacity into a hard constraint by removing
    load levels above it; without it capacity stays soft, as in the published model.
    """
    if cache.cail_load is None:
        raise ValueError("cache has no load table; the static model is solved by the greedy")
    w = tuple(weights) if weights is not None else (1.0,) * objectives
    if len(w) != objectives or any(x < 0 for x in w):
        raise ValueError("weights must be one non-negative value per objective")
    m, n, _ = cache.cail_load.shape
    if not 0 < n_dir <= m:
        raise ValueError("n_dir must be in (0, number of people]")

    pair_static = (
        w[1] * cache.trd + w[2] * cache.rpd if objectives == 4 else w[1] * cache.til
    )  # (m, n)
    w_cail = w[-1]
    levels = [_levels(cache.cail_load[:, i, :]) for i in range(n)]
    if max_load is not None:
        if len(max_load) != n or (np.asarray(max_load) < 0).any():
            raise ValueError("max_load must be one non-negative limit per centre")
        if int(np.sum(max_load)) < n_dir:
            raise ValueError("hard capacity is infeasible: total free places < n_dir")
        levels = [
            [(lo, min(hi, int(cap))) for lo, hi in lv if lo <= cap]
            for lv, cap in zip(levels, max_load, strict=True)
        ]

    # Variable layout: x (m*n) | y (sum levels) | z (m * sum levels)
    n_x = m * n
    y_index: list[list[int]] = []
    z_index: list[list[int]] = []
    cursor = n_x
    for i in range(n):
        y_index.append(list(range(cursor, cursor + len(levels[i]))))
        cursor += len(levels[i])
    for i in range(n):
        z_index.append([])
        for _ in levels[i]:
            z_index[i].append(cursor)
            cursor += m  # one z per person at this (centre, level)
    n_var = cursor

    cost = np.zeros(n_var)
    cost[:n_x] = ((w[0] * cache.ulpp)[:, None] + pair_static).ravel() / n_dir
    for i in range(n):
        for k, (lo, _) in enumerate(levels[i]):
            start = z_index[i][k]
            cost[start : start + m] = w_cail * cache.cail_load[:, i, lo] / n_dir

    rows, cols, vals, lb, ub = [], [], [], [], []

    def add_row(entries: list[tuple[int, float]], lo: float, hi: float) -> None:
        r = len(lb)
        for c, v in entries:
            rows.append(r)
            cols.append(c)
            vals.append(v)
        lb.append(lo)
        ub.append(hi)

    for j in range(m):  # each person at most one centre
        add_row([(j * n + i, 1.0) for i in range(n)], -np.inf, 1.0)
    add_row([(c, 1.0) for c in range(n_x)], n_dir, n_dir)  # exactly n_dir directed
    for i in range(n):
        add_row([(c, 1.0) for c in y_index[i]], 1.0, 1.0)  # one load level per centre
        for k, (lo, hi) in enumerate(levels[i]):
            z_cols = range(z_index[i][k], z_index[i][k] + m)
            # people counted at this level match the level's load range, or zero if inactive
            add_row([(c, 1.0) for c in z_cols] + [(y_index[i][k], -lo)], 0.0, np.inf)
            add_row([(c, 1.0) for c in z_cols] + [(y_index[i][k], -hi)], -np.inf, 0.0)
        for j in range(m):  # x splits across the active level only
            entries = [(j * n + i, -1.0)] + [
                (z_index[i][k] + j, 1.0) for k in range(len(levels[i]))
            ]
            add_row(entries, 0.0, 0.0)

    a = coo_matrix((vals, (rows, cols)), shape=(len(lb), n_var)).tocsr()
    integrality = np.zeros(n_var)
    for idx in y_index:
        integrality[idx] = 1
    start = time.perf_counter()
    res = milp(
        cost,
        constraints=LinearConstraint(a, lb, ub),
        integrality=integrality,
        bounds=Bounds(0.0, 1.0),
        options={"time_limit": time_limit, "disp": False},
    )
    seconds = time.perf_counter() - start
    if res.x is None:
        raise RuntimeError(f"MIP returned no solution: {res.message}")

    x = res.x[:n_x].reshape(m, n)
    chosen = np.argwhere(x > 0.5)
    pairs = [(int(j), int(i)) for j, i in chosen]
    if len(pairs) != n_dir or len({j for j, _ in pairs}) != n_dir:
        raise RuntimeError("MIP solution is not an integral allocation of n_dir people")
    gap = float(getattr(res, "mip_gap", 0.0) or 0.0)
    return MipResult(
        pairs=pairs,
        objective=float(res.fun),
        gap=gap,
        optimal=res.status == 0,
        seconds=seconds,
    )
