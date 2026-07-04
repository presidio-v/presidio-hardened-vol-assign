"""Tests for the crisp greedy baseline allocator (Paper B, RQ1)."""

from __future__ import annotations

import math

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
