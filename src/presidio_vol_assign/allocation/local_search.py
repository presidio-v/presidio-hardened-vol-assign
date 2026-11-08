"""Capacity-aware local search on the load-coupled model: the strongest cheap heuristic.

The stochastic optimiser of the audited system was designed for the published, separable
model. On the repaired model a fair heuristic reference should know about load: this one
starts from a given allocation (the deterministic fallback, by default) and applies
best-improvement moves until none improves or the wall-clock budget runs out:

* re-route: a directed person moves to another centre;
* swap: a directed person leaves and an undirected person enters, at any centre.

Every move changes the load of at most two centres, so its exact effect on the weighted
objective sum is computed from the per-centre sums of the CAIL-per-load table at the
neighbouring load levels. With hard capacity a move may not raise a centre above its limit.
It is deterministic: ties go to the lowest index.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from presidio_vol_assign.allocation.solvers import FISCache

# Moves must improve by more than this, so float noise cannot cycle.
_IMPROVE = 1e-12
# Swap candidates: the best removals combined with the best entries at each centre.
_TOP = 25


@dataclass(frozen=True)
class LocalSearchResult:
    pairs: list[tuple[int, int]]
    objective: float  # weighted objective sum
    moves: int
    seconds: float
    converged: bool  # stopped at a local optimum, not at the time budget


class _State:
    def __init__(self, cache: FISCache, n_dir: int, objectives: int, weights, pairs, max_load):
        if cache.cail_load is None:
            raise ValueError("local search needs a load-coupled cache")
        self.m, self.n, _ = cache.cail_load.shape
        w = tuple(weights) if weights is not None else (1.0,) * objectives
        pair_static = w[1] * cache.trd + w[2] * cache.rpd if objectives == 4 else w[1] * cache.til
        self.static = (w[0] * cache.ulpp)[:, None] + pair_static  # (m, n)
        self.cail = w[-1] * cache.cail_load  # (m, n, n_dir + 1)
        self.n_dir = n_dir
        self.limit = np.full(self.n, n_dir) if max_load is None else np.asarray(max_load, dtype=int)
        self.centre = np.full(self.m, -1)
        for j, i in pairs:
            self.centre[j] = i
        if (self.centre >= 0).sum() != n_dir:
            raise ValueError("start allocation must direct exactly n_dir distinct people")
        self.load = np.bincount(self.centre[self.centre >= 0], minlength=self.n)
        if (self.load > self.limit).any():
            raise ValueError("start allocation violates the hard capacity")

    def group_sum(self, i: int, level: int) -> float:
        """Sum of CAIL at centre i over its current members, evaluated at load ``level``."""
        members = np.flatnonzero(self.centre == i)
        return float(self.cail[members, i, level].sum()) if members.size else 0.0

    def total(self) -> float:
        directed = np.flatnonzero(self.centre >= 0)
        c = self.centre[directed]
        cost = self.static[directed, c].sum() + self.cail[directed, c, self.load[c]].sum()
        return float(cost) / self.n_dir

    def removal_delta(self) -> tuple[np.ndarray, np.ndarray]:
        """Change in raw cost if each directed person leaves (centre loads drop by one)."""
        directed = np.flatnonzero(self.centre >= 0)
        c = self.centre[directed]
        down = np.array([self.group_sum(i, max(self.load[i] - 1, 0)) for i in range(self.n)])
        here = np.array([self.group_sum(i, self.load[i]) for i in range(self.n)])
        # the others at c move to level load-1; the person's own terms disappear
        delta = (
            down[c] - self.cail[directed, c, self.load[c] - 1] - here[c] - self.static[directed, c]
        )
        return directed, delta

    def entry_delta(self, people: np.ndarray) -> np.ndarray:
        """Change in raw cost if each person in ``people`` enters each centre (load + 1)."""
        up_level = np.minimum(self.load + 1, self.n_dir)
        up = np.array([self.group_sum(i, up_level[i]) for i in range(self.n)])
        here = np.array([self.group_sum(i, self.load[i]) for i in range(self.n)])
        cail_in = self.cail[people[:, None], np.arange(self.n)[None, :], up_level[None, :]]
        delta = up - here + cail_in + self.static[people]
        delta[:, self.load + 1 > self.limit] = np.inf
        return delta

    def apply(self, out: int | None, person: int, centre: int) -> None:
        if out is not None:
            self.load[self.centre[out]] -= 1
            self.centre[out] = -1
        if self.centre[person] >= 0:
            self.load[self.centre[person]] -= 1
        self.centre[person] = centre
        self.load[centre] += 1


def _move_value(state: _State, out: int | None, person: int, centre: int) -> float:
    """Exact objective after a move, computed by applying it to a copy."""
    trial = _State.__new__(_State)
    trial.__dict__ = {**state.__dict__, "centre": state.centre.copy(), "load": state.load.copy()}
    trial.apply(out, person, centre)
    if (trial.load > trial.limit).any():
        return np.inf
    return trial.total()


def local_search(
    cache: FISCache,
    n_dir: int,
    objectives: int,
    start: list[tuple[int, int]],
    weights: tuple[float, ...] | None = None,
    max_load: np.ndarray | None = None,
    time_budget: float = 60.0,
) -> LocalSearchResult:
    """Best-improvement re-route / swap descent from ``start`` on the load-coupled model."""
    t0 = time.perf_counter()
    state = _State(cache, n_dir, objectives, weights, start, max_load)
    current = state.total()
    moves, converged = 0, False
    while time.perf_counter() - t0 < time_budget:
        best_value, best_move = current, None

        # re-route: leave one centre, enter another, evaluated exactly for the best few
        directed, rem = state.removal_delta()
        enter = state.entry_delta(directed)
        enter[np.arange(directed.size), state.centre[directed]] = np.inf
        approx = rem[:, None] + enter
        for flat in np.argsort(approx, axis=None, kind="stable")[:_TOP]:
            k, i = np.unravel_index(flat, approx.shape)
            if not np.isfinite(approx[k, i]):
                break
            value = _move_value(state, None, int(directed[k]), int(i))
            if value < best_value - _IMPROVE:
                best_value, best_move = value, (None, int(directed[k]), int(i))

        # swap: one directed person out, one undirected person in
        outside = np.flatnonzero(state.centre < 0)
        if outside.size:
            enter_out = state.entry_delta(outside)
            leavers = directed[np.argsort(rem, kind="stable")[:_TOP]]
            entrants = np.argsort(enter_out, axis=None, kind="stable")[:_TOP]
            for j in leavers:
                for flat in entrants:
                    k, i = np.unravel_index(flat, enter_out.shape)
                    if not np.isfinite(enter_out[k, i]):
                        continue
                    value = _move_value(state, int(j), int(outside[k]), int(i))
                    if value < best_value - _IMPROVE:
                        best_value, best_move = value, (int(j), int(outside[k]), int(i))

        if best_move is None:
            converged = True
            break
        state.apply(*best_move)
        current = best_value
        moves += 1

    pairs = [(int(j), int(state.centre[j])) for j in np.flatnonzero(state.centre >= 0)]
    return LocalSearchResult(
        pairs=pairs,
        objective=current,
        moves=moves,
        seconds=time.perf_counter() - t0,
        converged=converged,
    )


def capacity_greedy(
    cache: FISCache,
    n_dir: int,
    objectives: int,
    max_load: np.ndarray,
    weights: tuple[float, ...] | None = None,
) -> list[tuple[int, int]]:
    """A deterministic allocation that respects hard capacity: the start for local search.

    Takes (person, centre) pairs in order of their static weighted cost (CAIL at zero load,
    as in the published model) and keeps a pair when the person is still free and the
    centre still has room, until ``n_dir`` people are directed. Ties go to the lower index.
    """
    w = tuple(weights) if weights is not None else (1.0,) * objectives
    pair_terms = (cache.trd, cache.rpd, cache.cail) if objectives == 4 else (cache.til, cache.cail)
    cost = w[0] * cache.ulpp[:, None] + sum(
        wk * term for wk, term in zip(w[1:], pair_terms, strict=True)
    )
    limit = np.asarray(max_load, dtype=int)
    if int(limit.sum()) < n_dir:
        raise ValueError("hard capacity is infeasible: total free places < n_dir")
    m, n = cost.shape
    order = np.lexsort((np.arange(m * n), cost.ravel()))
    taken = np.zeros(m, dtype=bool)
    load = np.zeros(n, dtype=int)
    pairs: list[tuple[int, int]] = []
    for flat in order:
        j, i = divmod(int(flat), n)
        if taken[j] or load[i] >= limit[i]:
            continue
        pairs.append((j, i))
        taken[j] = True
        load[i] += 1
        if len(pairs) == n_dir:
            break
    return pairs
