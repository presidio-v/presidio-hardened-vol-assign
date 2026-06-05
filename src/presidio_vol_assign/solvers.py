"""Backward-compatible facade over the generic engine + ED-staffing domain.

The solver logic now lives in two places:
    engine.py                  — domain-agnostic NSGA-II / NRGA machinery
    domains/ed_staffing.py     — the original two-objective model

This module preserves the v0.1.0 public API (``precompute_fis``,
``decode_chromosome``, ``evaluate_chromosome``, ``sel_nrga``, ``solve``) and the
legacy DEAP ``creator`` types (``PVAFitness`` / ``PVAIndividual``), so existing
imports and tests keep working unchanged.
"""

from __future__ import annotations

from presidio_vol_assign.domains.ed_staffing import (
    EDStaffingDomain,
    FISCache,
    decode_chromosome,
    evaluate_chromosome,
    precompute_fis,
)
from presidio_vol_assign.engine import ensure_creator_types, run, sel_nrga
from presidio_vol_assign.models import ParetoFront, ProblemInstance, RunConfig

# Register the legacy PVAFitness / PVAIndividual creator types on import, so
# callers that reference ``deap.creator.PVAIndividual`` continue to work.
ensure_creator_types(EDStaffingDomain())


def solve(problem: ProblemInstance, config: RunConfig) -> list[ParetoFront]:
    """Run the ED-staffing solver(s) and return one ParetoFront per solver.

    config.solver can be "nsga2", "nrga", or "both".
    """
    return run(problem, config, EDStaffingDomain())


__all__ = [
    "FISCache",
    "precompute_fis",
    "decode_chromosome",
    "evaluate_chromosome",
    "sel_nrga",
    "solve",
]
