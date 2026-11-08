"""Near-optimal alternatives must match brute force on the small fixture instance."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from presidio_vol_assign.allocation import multiplicity as mu
from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs
from presidio_vol_assign.allocation.load_coupling import precompute_load_coupled_cache
from presidio_vol_assign.allocation.solvers import evaluate_pairs, precompute_fis_cache


def _all_allocations(problem):
    for people in itertools.combinations(range(len(problem.people)), problem.n_dir):
        for centres in itertools.product(range(problem.n_centers), repeat=problem.n_dir):
            yield list(zip(people, centres, strict=True))


def _cache(problem, config, kappa):
    static = precompute_fis_cache(problem, config)
    if kappa is None:
        return static
    return precompute_load_coupled_cache(problem, config, kappa, static=static)


@pytest.mark.parametrize("kappa", [None, 1.0, math.inf])
@pytest.mark.parametrize("epsilon", [0.0, 0.5, 3.0])
def test_max_churn_matches_brute_force(problem, base_config, kappa, epsilon) -> None:
    k = base_config.objectives
    cache = _cache(problem, base_config, kappa)
    scored = [(float(np.sum(evaluate_pairs(p, cache, k))), p) for p in _all_allocations(problem)]
    z_star, reference = min(scored, key=lambda t: t[0])
    best_churn = max(mu.churn(reference, p) for q, p in scored if q <= z_star + epsilon + 1e-9)

    model = mu.model_for(cache, problem.n_dir, k)
    alts = mu.near_optimal_alternatives(model, problem.n_dir, reference, z_star, epsilon, k=1)
    assert alts and alts[0].optimal
    assert alts[0].objective <= z_star + epsilon + 1e-6
    assert mu.churn(reference, alts[0].pairs) == pytest.approx(best_churn)


def test_static_model_reaches_the_greedy_optimum(problem, base_config) -> None:
    k = base_config.objectives
    cache = precompute_fis_cache(problem, base_config)
    greedy = exact_weighted_sum_pairs(cache, problem.n_dir, k)
    z_star = float(np.sum(evaluate_pairs(greedy, cache, k)))
    model = mu.build_static_mip(cache, problem.n_dir, k)
    canon = mu.canonical_optimum(model, problem.n_dir, z_star)
    assert canon.objective == pytest.approx(z_star, abs=1e-6)


def test_canonical_optimum_is_the_key_minimum_among_optima(problem, base_config) -> None:
    k = base_config.objectives
    cache = _cache(problem, base_config, 1.0)
    scored = [(float(np.sum(evaluate_pairs(p, cache, k))), p) for p in _all_allocations(problem)]
    z_star = min(q for q, _ in scored)
    model = mu.model_for(cache, problem.n_dir, k)
    key = mu.tie_break_key(model.n_x).reshape(model.m, model.n)
    optima = [p for q, p in scored if q <= z_star + 1e-6]
    expected = min(optima, key=lambda p: sum(key[j, i] for j, i in p))
    canon = mu.canonical_optimum(model, problem.n_dir, z_star)
    assert sorted(canon.pairs) == sorted(expected)


def test_tie_break_key_is_fixed_by_its_seed() -> None:
    np.testing.assert_array_equal(mu.tie_break_key(50, seed=3), mu.tie_break_key(50, seed=3))
    assert not np.array_equal(mu.tie_break_key(50, seed=3), mu.tie_break_key(50, seed=4))
    assert ((mu.tie_break_key(1000) >= 1.0) & (mu.tie_break_key(1000) < 2.0)).all()


def test_churn_and_jaccard() -> None:
    a = [(0, 0), (1, 1), (2, 0)]
    b = [(0, 0), (1, 0), (3, 1)]
    assert mu.churn(a, b) == pytest.approx(2 / 3)
    assert mu.churn(b, a) == pytest.approx(2 / 3)
    assert mu.directed_jaccard(a, b) == pytest.approx(2 / 4)


def test_rejects_bad_arguments(problem, base_config) -> None:
    k = base_config.objectives
    cache = precompute_fis_cache(problem, base_config)
    model = mu.model_for(cache, problem.n_dir, k)
    ref = exact_weighted_sum_pairs(cache, problem.n_dir, k)
    with pytest.raises(ValueError):
        mu.near_optimal_alternatives(model, problem.n_dir, ref, 0.0, -1.0)
    with pytest.raises(ValueError):
        mu.near_optimal_alternatives(model, problem.n_dir, ref, 0.0, 1.0, mode="seats")
    with pytest.raises(ValueError):
        mu.model_for(cache, problem.n_dir, k, max_load=np.ones(problem.n_centers))
