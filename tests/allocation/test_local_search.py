"""Local search on the load-coupled model: exact bookkeeping, never worse, respects capacity."""

from __future__ import annotations

import math

import numpy as np
import pytest

from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs
from presidio_vol_assign.allocation.exact_mip import solve_weighted_mip
from presidio_vol_assign.allocation.load_coupling import precompute_load_coupled_cache
from presidio_vol_assign.allocation.local_search import local_search
from presidio_vol_assign.allocation.solvers import evaluate_pairs, precompute_fis_cache


@pytest.mark.parametrize("kappa", [1.0, 3.0, math.inf])
def test_never_worse_than_start_and_not_better_than_optimum(problem, base_config, kappa):
    k = base_config.objectives
    static = precompute_fis_cache(problem, base_config)
    cache = precompute_load_coupled_cache(problem, base_config, kappa, static=static)
    start = exact_weighted_sum_pairs(static, problem.n_dir, k)
    res = local_search(cache, problem.n_dir, k, start, time_budget=10)
    q = lambda p: float(np.sum(evaluate_pairs(p, cache, k)))  # noqa: E731
    assert res.objective == pytest.approx(q(res.pairs), abs=1e-9)
    assert res.objective <= q(start) + 1e-9
    opt = solve_weighted_mip(cache, problem.n_dir, k, time_limit=30)
    assert res.objective >= opt.objective - 1e-6
    assert res.converged


def test_hard_capacity_is_respected(problem, base_config):
    k = base_config.objectives
    static = precompute_fis_cache(problem, base_config)
    cache = precompute_load_coupled_cache(problem, base_config, 1.0, static=static)
    limit = np.array([2, 2, 2])
    start = [(0, 0), (1, 0), (2, 1), (3, 2)]
    res = local_search(cache, problem.n_dir, k, start, max_load=limit, time_budget=10)
    loads = np.bincount([c for _, c in res.pairs], minlength=problem.n_centers)
    assert (loads <= limit).all()
    assert len({j for j, _ in res.pairs}) == problem.n_dir


def test_rejects_infeasible_start(problem, base_config):
    k = base_config.objectives
    cache = precompute_load_coupled_cache(problem, base_config, 1.0)
    start = [(0, 0), (1, 0), (2, 0), (3, 0)]
    with pytest.raises(ValueError):
        local_search(cache, problem.n_dir, k, start, max_load=np.array([2, 2, 2]))
    with pytest.raises(ValueError):
        local_search(precompute_fis_cache(problem, base_config), problem.n_dir, k, start)


def test_capacity_greedy_is_feasible_and_deterministic(problem, base_config):
    from presidio_vol_assign.allocation.local_search import capacity_greedy

    k = base_config.objectives
    cache = precompute_fis_cache(problem, base_config)
    limit = np.array([1, 2, 1])
    a = capacity_greedy(cache, problem.n_dir, k, limit)
    assert a == capacity_greedy(cache, problem.n_dir, k, limit)
    loads = np.bincount([c for _, c in a], minlength=problem.n_centers)
    assert (loads <= limit).all() and len({j for j, _ in a}) == problem.n_dir
    with pytest.raises(ValueError):
        capacity_greedy(cache, problem.n_dir, k, np.array([1, 1, 1]))
