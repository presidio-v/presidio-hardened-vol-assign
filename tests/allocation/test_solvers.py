"""Unit tests for allocation/solvers.py — chromosome encoding, decoding,
and end-to-end solve for all three MOEAs in both 3-obj and 4-obj modes."""

from __future__ import annotations

import dataclasses

import pytest

from presidio_vol_assign.allocation.models import AllocationSolverType
from presidio_vol_assign.allocation.solvers import (
    decode_chromosome,
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
