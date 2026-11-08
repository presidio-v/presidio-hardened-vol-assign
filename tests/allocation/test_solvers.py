"""Unit tests for allocation/solvers.py — chromosome encoding, decoding,
and end-to-end solve for all three MOEAs in both 3-obj and 4-obj modes."""

from __future__ import annotations

import dataclasses

import pytest

from presidio_vol_assign.allocation.models import AllocationSolverType
from presidio_vol_assign.allocation.solvers import (
    decode_chromosome,
    evaluate_pairs,
    precompute_fis_cache,
    solve,
)


class TestDecodeChromosome:
    def test_decode_round_trip(self):
        # n_dir=3, n_centers=4 → reals 0.0, 0.4, 0.99 map to centers 0, 1, 3
        ind = [10, 20, 30, 0.0, 0.4, 0.99]
        pairs = decode_chromosome(ind, n_dir=3, n_centers=4)
        assert pairs == [(10, 0), (20, 1), (30, 3)]

    def test_decode_clamps_boundary(self):
        # r=1.0 (rare but possible) clamps to n_centers - 1
        ind = [0, 1.0]
        pairs = decode_chromosome(ind, n_dir=1, n_centers=5)
        assert pairs == [(0, 4)]


class TestPrecomputeFISCache:
    def test_cache_shapes_match_problem(self, problem, base_config):
        cache = precompute_fis_cache(problem, base_config)
        assert cache.ulpp.shape == (problem.n_people,)
        assert cache.cail.shape == (problem.n_people, problem.n_centers)
        # 4-obj mode populates trd, rpd
        assert (cache.trd != 0).any()
        assert (cache.rpd != 0).any()
        # til left at zero in 4-obj mode
        assert (cache.til == 0).all()

    def test_cache_3obj_populates_til_only(self, problem, base_config):
        cfg = dataclasses.replace(base_config, objectives=3)
        cache = precompute_fis_cache(problem, cfg)
        assert (cache.til != 0).any()
        assert (cache.trd == 0).all()
        assert (cache.rpd == 0).all()


class TestSolveAllAlgorithms:
    @pytest.mark.parametrize(
        "solver",
        [AllocationSolverType.NSGA2, AllocationSolverType.NRGA, AllocationSolverType.NSGA3],
    )
    def test_4obj_solve_returns_front(self, problem, base_config, solver):
        cfg = dataclasses.replace(base_config, solver=solver, objectives=4)
        front = solve(problem, cfg)
        assert front.solver == solver
        assert front.objectives_count == 4
        assert front.nns >= 1
        # All solutions are valid: n_dir allocations, distinct persons
        for s in front.solutions:
            assert s.objectives_count == 4
            assert len(s.allocations) == problem.n_dir
            assert len({a.person_id for a in s.allocations}) == problem.n_dir

    @pytest.mark.parametrize(
        "solver",
        [AllocationSolverType.NSGA2, AllocationSolverType.NRGA, AllocationSolverType.NSGA3],
    )
    def test_3obj_solve_returns_front(self, problem, base_config, solver):
        cfg = dataclasses.replace(base_config, solver=solver, objectives=3)
        front = solve(problem, cfg)
        assert front.objectives_count == 3
        assert front.nns >= 1
        for s in front.solutions:
            assert s.objectives_count == 3

    def test_seed_determinism(self, problem, base_config):
        f1 = solve(problem, dataclasses.replace(base_config, seed=42))
        f2 = solve(problem, dataclasses.replace(base_config, seed=42))
        # Same seed → same first solution
        assert f1.solutions[0].fitness == f2.solutions[0].fitness

    def test_objectives_drive_fitness_dimensionality(self, problem, base_config):
        front4 = solve(problem, dataclasses.replace(base_config, objectives=4))
        front3 = solve(problem, dataclasses.replace(base_config, objectives=3))
        assert len(front4.solutions[0].fitness) == 4
        assert len(front3.solutions[0].fitness) == 3


class TestBudgetAndSeeding:
    def test_encode_inverts_decode(self):
        from deap import creator

        from presidio_vol_assign.allocation.solvers import encode_pairs

        pairs = [(4, 0), (1, 2), (7, 1)]
        ind = encode_pairs(pairs, 3, creator.AllocIndividual3)
        assert decode_chromosome(ind, n_dir=3, n_centers=3) == pairs

    def test_unbounded_unseeded_run_is_unchanged(self, problem, base_config):
        cfg = dataclasses.replace(base_config, objectives=3, generations=5, seed=11)
        a = solve(problem, cfg)
        b = solve(problem, cfg, time_budget=None, seed_allocations=None)
        assert [s.mn_ulpp for s in a.solutions] == [s.mn_ulpp for s in b.solutions]
        assert a.generations_run == b.generations_run == 5

    def test_time_budget_stops_early_after_one_generation(self, problem, base_config):
        cfg = dataclasses.replace(base_config, objectives=3, generations=10_000, seed=11)
        front = solve(problem, cfg, time_budget=0.0)
        assert front.generations_run == 1

    def test_seeded_optimum_survives_elitist_selection(self, problem, base_config):
        from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs
        from presidio_vol_assign.allocation.solvers import evaluate_pairs

        cfg = dataclasses.replace(base_config, objectives=3, generations=3, seed=11)
        cache = precompute_fis_cache(problem, cfg)
        exact = exact_weighted_sum_pairs(cache, problem.n_dir, 3)
        target = sum(evaluate_pairs(exact, cache, 3))
        front = solve(problem, cfg, cache=cache, seed_allocations=[exact])
        best = min(s.mn_ulpp + s.mn_til + s.mn_cail for s in front.solutions)
        assert best == pytest.approx(target, abs=1e-9)

    def test_rejects_malformed_seed(self, problem, base_config):
        cfg = dataclasses.replace(base_config, objectives=3, generations=2, seed=11)
        with pytest.raises(ValueError):
            solve(problem, cfg, seed_allocations=[[(0, 0)]])


class TestUlppObjective:
    """The corrected ULPP objective (Appl. Sci. correction): mean over the people left out."""

    def test_rejects_unknown_mode(self, base_config):
        with pytest.raises(ValueError):
            dataclasses.replace(base_config, ulpp_objective="both")

    def test_cache_carries_mode(self, problem, base_config):
        cfg = dataclasses.replace(base_config, ulpp_objective="left_out")
        assert precompute_fis_cache(problem, cfg).ulpp_objective == "left_out"
        assert precompute_fis_cache(problem, base_config).ulpp_objective == "directed"

    def test_left_out_is_mean_over_undirected(self, problem, base_config):
        cache = dataclasses.replace(
            precompute_fis_cache(problem, base_config), ulpp_objective="left_out"
        )
        pairs = [(j, 0) for j in range(problem.n_dir)]
        directed = set(range(problem.n_dir))
        expected = sum(cache.ulpp[j] for j in range(problem.n_people) if j not in directed) / (
            problem.n_people - problem.n_dir
        )
        assert evaluate_pairs(pairs, cache, 4)[0] == pytest.approx(expected)

    def test_left_out_reverses_preference(self, problem, base_config):
        """Directing the highest-ULPP people is best under the corrected objective."""
        base = precompute_fis_cache(problem, base_config)
        order = sorted(range(problem.n_people), key=lambda j: base.ulpp[j])
        low = [(j, 0) for j in order[: problem.n_dir]]
        high = [(j, 0) for j in order[-problem.n_dir :]]
        assert evaluate_pairs(low, base, 4)[0] < evaluate_pairs(high, base, 4)[0]
        fixed = dataclasses.replace(base, ulpp_objective="left_out")
        assert evaluate_pairs(high, fixed, 4)[0] < evaluate_pairs(low, fixed, 4)[0]

    def test_other_objectives_unchanged(self, problem, base_config):
        base = precompute_fis_cache(problem, base_config)
        fixed = dataclasses.replace(base, ulpp_objective="left_out")
        pairs = [(j, j % problem.n_centers) for j in range(problem.n_dir)]
        assert evaluate_pairs(pairs, base, 4)[1:] == evaluate_pairs(pairs, fixed, 4)[1:]

    def test_exact_decider_refuses_corrected_mode(self, problem, base_config):
        from presidio_vol_assign.allocation.baselines import exact_weighted_sum_pairs

        fixed = dataclasses.replace(
            precompute_fis_cache(problem, base_config), ulpp_objective="left_out"
        )
        with pytest.raises(NotImplementedError):
            exact_weighted_sum_pairs(fixed, problem.n_dir, 4)
