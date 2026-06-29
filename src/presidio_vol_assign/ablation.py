"""Objective-ablation analysis — empirical justification of the qualitative indicators.

The ATRES reviewers (R2.2) asked for evidence that the three qualitative
indicators (fairness / transport feasibility / centre-balance for the
humanitarian model; importance / preference for the ED model) each capture
distinct, relevant information rather than being redundant.

This module answers that with a **leave-one-objective-out ablation**. For each
objective in turn it re-solves the problem with that objective *dropped from the
optimisation* (the solver no longer selects for it), then measures the dropped
objective — and the overall hypervolume — back in the **full** objective space:

* ``delta_dropped`` — how much worse the dropped objective gets (mean over the
  front) when it is no longer optimised. A large positive value means the
  objective is doing real work: nothing else in the model drives it down for
  free, so it is non-redundant.
* ``delta_hv`` — how much full-space hypervolume is lost by ignoring the
  objective. A large drop means the objective contributes Pareto structure the
  others cannot recover.

A near-zero ``delta`` would flag an indicator that the other objectives already
capture (redundant). The analysis is deterministic under ``--seed`` and reuses
the single pre-computed FIS cache, so it is cheap.

Public API:
    run_ablation(domain, problem, config) -> list[AblationRow]
    write_ablation_csv(rows, output_dir) -> Path
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from presidio_vol_assign.domains.base import Domain
from presidio_vol_assign.engine import _solvers_for, run
from presidio_vol_assign.metrics import compute_metrics
from presidio_vol_assign.models import RunConfig, Solution


@dataclass
class AblationRow:
    """One leave-one-out row: the effect of dropping a single objective."""

    dropped: str  # objective name removed from the optimisation
    solver: str
    nns_full: int  # front size with all objectives
    nns_ablated: int  # front size with `dropped` removed
    mean_dropped_full: float  # mean of `dropped` over the full-model front
    mean_dropped_ablated: float  # mean of `dropped` over the ablated front
    delta_dropped: float  # ablated - full (>0 => objective is non-redundant)
    hv_full: float  # full-space HV of the full-model front
    hv_ablated: float  # full-space HV of the ablated front
    delta_hv: float  # hv_full - hv_ablated (>0 => objective adds HV)


class _AblatedDomain(Domain):
    """Wrap a domain so the engine optimises every objective except one.

    Selection sees only the kept objectives (reduced fitness), but
    :meth:`to_solution` reports the **full** objective vector, so the resulting
    front can be measured in the original objective space.
    """

    def __init__(self, base: Domain, drop_idx: int) -> None:
        self._base = base
        self._drop = drop_idx
        self._keep = [i for i in range(base.n_objectives) if i != drop_idx]

        self.name = f"{base.name}-ablate-{base.objective_names[drop_idx]}"
        self.objective_names = tuple(base.objective_names[i] for i in self._keep)
        self.reference_point = tuple(base.reference_point[i] for i in self._keep)
        self.ideal_point = tuple(base.ideal_point[i] for i in self._keep)
        self.weights = tuple(base.weights[i] for i in self._keep)
        # Distinct DEAP creator names per (base, dropped index) so the reduced
        # fitness type never collides with the full-dimensional one.
        self.fitness_attr = f"{base.fitness_attr}_ab{drop_idx}"
        self.individual_attr = f"{base.individual_attr}_ab{drop_idx}"
        self.required_inputs = base.required_inputs
        self.assignment_fieldnames = base.assignment_fieldnames

    # --- delegated, behaviour-preserving hooks ---
    def load(self, primary: Path, secondary: Path) -> Any:
        return self._base.load(primary, secondary)

    def assignment_row(self, assignment: Any) -> dict[str, Any]:
        return self._base.assignment_row(assignment)

    def precompute(self, problem: Any) -> Any:
        return self._base.precompute(problem)

    def init_individual(self, problem: Any, individual_cls: type) -> list:
        return self._base.init_individual(problem, individual_cls)

    def mate(self, ind1: list, ind2: list) -> tuple[list, list]:
        return self._base.mate(ind1, ind2)

    def mutate(self, ind: list) -> tuple[list]:
        return self._base.mutate(ind)

    # --- projected evaluation / full-space reporting ---
    def evaluate(self, individual: list, cache: Any, problem: Any) -> tuple[float, ...]:
        full = self._base.evaluate(individual, cache, problem)
        return tuple(full[i] for i in self._keep)

    def to_solution(self, individual: list, cache: Any, problem: Any) -> Solution:
        # Report the FULL objective vector (assignments are not needed for the
        # ablation summary, so they are omitted to stay domain-agnostic).
        full = self._base.evaluate(individual, cache, problem)
        return Solution(assignments=[], objectives=full)


def run_ablation(domain: Domain, problem: object, config: RunConfig) -> list[AblationRow]:
    """Leave-one-objective-out ablation; one row per objective.

    Uses a single solver (the first one implied by ``config.solver``) for a
    clean like-for-like comparison. Raises ``ValueError`` for single-objective
    models, where ablation is undefined.
    """
    if domain.n_objectives < 2:
        raise ValueError("ablation requires a model with at least 2 objectives")

    solver_val = config.solver if isinstance(config.solver, str) else config.solver.value
    solver = _solvers_for(solver_val)[0]
    single_cfg = replace(config, solver=solver.value)

    cache = domain.precompute(problem)
    full_front = run(problem, single_cfg, domain, cache=cache)[0]
    full_metrics = compute_metrics(full_front)
    full_objs = np.array([s.objectives for s in full_front.solutions], dtype=float)

    rows: list[AblationRow] = []
    for k in range(domain.n_objectives):
        ablated = _AblatedDomain(domain, k)
        ab_front = run(problem, single_cfg, ablated, cache=cache)[0]
        ab_metrics = compute_metrics(ab_front)
        ab_objs = np.array([s.objectives for s in ab_front.solutions], dtype=float)

        mean_full = float(full_objs[:, k].mean())
        mean_ablated = float(ab_objs[:, k].mean())
        rows.append(
            AblationRow(
                dropped=domain.objective_names[k],
                solver=solver.value,
                nns_full=full_metrics.nns,
                nns_ablated=ab_metrics.nns,
                mean_dropped_full=mean_full,
                mean_dropped_ablated=mean_ablated,
                delta_dropped=mean_ablated - mean_full,
                hv_full=full_metrics.hv,
                hv_ablated=ab_metrics.hv,
                delta_hv=full_metrics.hv - ab_metrics.hv,
            )
        )
    return rows


def write_ablation_csv(rows: list[AblationRow], output_dir: Path) -> Path:
    """Write the ablation summary to ``ablation_<ts>.csv``. Returns the path."""
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = output_dir / f"ablation_{ts}.csv"
    pd.DataFrame([asdict(r) for r in rows]).to_csv(path, index=False)
    return path
