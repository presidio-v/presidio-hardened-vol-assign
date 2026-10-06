"""The load-coupled MIP must reach the brute-force optimum on a small instance."""

from __future__ import annotations

import dataclasses
import itertools
import math

import pytest

from presidio_vol_assign.allocation.exact_mip import solve_weighted_mip
from presidio_vol_assign.allocation.load_coupling import precompute_load_coupled_cache
from presidio_vol_assign.allocation.solvers import evaluate_pairs


@pytest.mark.parametrize("objectives", [3, 4])
@pytest.mark.parametrize("kappa", [1.0, 3.0, math.inf])
def test_mip_matches_brute_force(problem, base_config, objectives, kappa) -> None:
    config = dataclasses.replace(base_config, objectives=objectives)
    cache = precompute_load_coupled_cache(problem, config, kappa)
    weights = tuple(float(i + 1) for i in range(objectives))

    def cost(pairs):
        return sum(w * f for w, f in zip(weights, evaluate_pairs(pairs, cache, objectives)))

    best = min(
        cost(list(zip(people, centres, strict=True)))
        for people in itertools.combinations(range(len(problem.people)), problem.n_dir)
        for centres in itertools.product(range(problem.n_centers), repeat=problem.n_dir)
    )
    result = solve_weighted_mip(cache, problem.n_dir, objectives, weights, time_limit=30)
    assert result.optimal
    assert cost(result.pairs) == pytest.approx(best, abs=1e-7)
    assert result.objective == pytest.approx(best, abs=1e-6)


def test_mip_requires_load_table(problem, base_config) -> None:
    from presidio_vol_assign.allocation.solvers import precompute_fis_cache

    with pytest.raises(ValueError):
        solve_weighted_mip(precompute_fis_cache(problem, base_config), problem.n_dir, 4)


def test_hard_capacity_respected_and_optimal(problem, base_config) -> None:
    import numpy as np

    from presidio_vol_assign.allocation.load_coupling import (
        capacities_from_free_factor,
        hard_load_limits,
    )

    config = dataclasses.replace(base_config, objectives=3)
    cap = capacities_from_free_factor(problem, 1.5)
    limits = hard_load_limits(problem, cap)
    cache = precompute_load_coupled_cache(problem, config, cap)

    def cost(pairs):
        return sum(evaluate_pairs(pairs, cache, 3))

    feasible = []
    for people in itertools.combinations(range(len(problem.people)), problem.n_dir):
        for centres in itertools.product(range(problem.n_centers), repeat=problem.n_dir):
            if (np.bincount(centres, minlength=problem.n_centers) <= limits).all():
                feasible.append(cost(list(zip(people, centres, strict=True))))
    result = solve_weighted_mip(cache, problem.n_dir, 3, max_load=limits, time_limit=30)
    loads = np.bincount([c for _, c in result.pairs], minlength=problem.n_centers)
    assert (loads <= limits).all()
    assert cost(result.pairs) == pytest.approx(min(feasible), abs=1e-7)


def test_hard_capacity_infeasible_is_rejected(problem, base_config) -> None:
    import numpy as np

    cache = precompute_load_coupled_cache(problem, base_config, 2.0)
    with pytest.raises(ValueError):
        solve_weighted_mip(cache, problem.n_dir, 4, max_load=np.zeros(problem.n_centers, int))
