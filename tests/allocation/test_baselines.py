"""Tests for the crisp greedy baseline allocator (Paper B, RQ1)."""

from __future__ import annotations

import math

import pytest

from presidio_vol_assign.allocation.baselines import (
    crisp_greedy_pairs,
    crisp_greedy_solution,
)
from presidio_vol_assign.allocation.solvers import precompute_fis_cache


def test_pairs_are_feasible(problem, base_config) -> None:
    pairs = crisp_greedy_pairs(problem, base_config)
    assert len(pairs) == problem.n_dir
    persons = [p for p, _ in pairs]
    assert len(set(persons)) == problem.n_dir  # each directed person is distinct
    assert all(0 <= p < problem.n_people for p, _ in pairs)
    assert all(0 <= c < problem.n_centers for _, c in pairs)


def test_decision_is_deterministic(problem, base_config) -> None:
    assert crisp_greedy_pairs(problem, base_config) == crisp_greedy_pairs(problem, base_config)


def test_highest_priority_person_is_selected(problem, base_config) -> None:
    # P4 (index 4): age 90, severe disability, life-threatening injury, IDL 95,
    # RTR 2h — the most urgent person; must be among the n_dir directed.
    selected = {p for p, _ in crisp_greedy_pairs(problem, base_config)}
    assert 4 in selected


def test_solution_shape_matches_config(problem, base_config) -> None:
    cache = precompute_fis_cache(problem, base_config)
    solution = crisp_greedy_solution(problem, base_config, cache)
    assert solution.objectives_count == 4
    assert solution.n_allocations == problem.n_dir
    assert all(math.isfinite(x) for x in solution.fitness)


@pytest.mark.parametrize("objectives", [3, 4])
def test_exact_weighted_sum_is_optimal_by_brute_force(problem, base_config, objectives) -> None:
    import dataclasses
    import itertools

    base_config = dataclasses.replace(base_config, objectives=objectives)

    from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs
    from presidio_vol_assign.allocation.solvers import evaluate_pairs

    cache = precompute_fis_cache(problem, base_config)
    k = base_config.objectives
    for weights in [(1.0,) * k, tuple(float(i + 1) for i in range(k))]:
        pairs = exact_weighted_sum_pairs(cache, problem.n_dir, k, weights)

        def cost(ps, w=weights):
            return sum(wi * fi for wi, fi in zip(w, evaluate_pairs(ps, cache, k)))

        n, m = problem.n_centers, len(problem.people)
        best = min(
            cost(list(zip(people, centers)))
            for people in itertools.combinations(range(m), problem.n_dir)
            for centers in itertools.product(range(n), repeat=problem.n_dir)
        )
        assert math.isclose(cost(pairs), best, rel_tol=0, abs_tol=1e-9)
        assert len({p for p, _ in pairs}) == problem.n_dir


def test_exact_weighted_sum_rejects_bad_weights(problem, base_config) -> None:
    from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs

    cache = precompute_fis_cache(problem, base_config)
    with pytest.raises(ValueError):
        bad = (1.0, -1.0, 1.0, 1.0)
        exact_weighted_sum_pairs(cache, problem.n_dir, base_config.objectives, bad)
    with pytest.raises(ValueError):
        exact_weighted_sum_pairs(cache, 0, base_config.objectives)
