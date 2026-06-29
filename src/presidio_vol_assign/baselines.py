"""Baseline (non-evolutionary) comparators for the multi-objective models.

The ATRES reviewers' single recurring technical objection was that the study
only compared NSGA-II against NRGA — two metaheuristics against each other —
without measuring the framework against any existing/baseline allocation method.
This module supplies that baseline: a deterministic **weighted-sum constructive
heuristic**.

A grid of weight vectors on the objective simplex is swept; each weight vector
is scalarised into a single greedy construction, yielding one candidate solution
per weight. The non-dominated subset of those candidates forms a baseline Pareto
front the evolutionary solvers can be compared against (same metrics, same
``ParetoFront`` plumbing). The construction uses no randomness, so the baseline
is reproducible regardless of ``--seed``.

This module only provides the generic ``weight_simplex`` helper; each
:class:`~presidio_vol_assign.domains.base.Domain` knows its own encoding and
builds its candidate genomes in ``Domain.baseline_population`` using it.
"""

from __future__ import annotations

from itertools import combinations


def weight_simplex(n_objectives: int, steps: int = 8) -> list[tuple[float, ...]]:
    """Enumerate weight vectors on the ``n_objectives`` unit simplex.

    Returns every vector of non-negative weights summing to 1 whose components
    are integer multiples of ``1 / steps`` — i.e. the standard simplex-lattice
    design. The objective-space vertices (e.g. ``(1, 0, 0)``) are included, so
    each objective is optimised on its own by at least one weight vector.

    Examples (counts): 2 objectives, ``steps=8`` -> 9 vectors;
    3 objectives, ``steps=6`` -> 28 vectors.
    """
    if n_objectives < 1:
        raise ValueError(f"n_objectives must be >= 1, got {n_objectives}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    # Compositions of `steps` into `n_objectives` non-negative integer parts via
    # the stars-and-bars bijection: choose the (n_objectives - 1) bar positions.
    weights: list[tuple[float, ...]] = []
    for bars in combinations(range(steps + n_objectives - 1), n_objectives - 1):
        prev = -1
        parts: list[int] = []
        for bar in bars:
            parts.append(bar - prev - 1)
            prev = bar
        parts.append(steps + n_objectives - 1 - prev - 1)
        weights.append(tuple(p / steps for p in parts))
    return weights
