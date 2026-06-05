"""Sensitivity analysis — robustness of the Pareto front to FIS rule-base uncertainty.

The three FIS rule bases are expert-elicited and therefore uncertain. This module
sweeps a multiplicative perturbation of the FIS output scores (e.g. -20 %, -10 %,
0, +10 %, +20 %) and reports how the Pareto-front quality metrics (NNS, MID, SM,
HV) shift, so a reader can judge how sensitive the solutions are to the rule-base
specification.

The FIS scores are pre-computed once; each perturbation only rescales the cached
scores (``Domain.perturb``) and re-runs the solver, so the sweep is cheap and
deterministic.

Public API:
    run_sensitivity(domain, problem, config, factors) -> list[SensitivityRow]
    write_sensitivity_csv(rows, output_dir) -> Path
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from presidio_vol_assign.domains.base import Domain
from presidio_vol_assign.engine import run
from presidio_vol_assign.metrics import compute_metrics
from presidio_vol_assign.models import RunConfig

# Default perturbation grid: ±10 % and ±20 % around the nominal rule base.
DEFAULT_FACTORS: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)


@dataclass
class SensitivityRow:
    """One (perturbation, solver) row of the sensitivity sweep."""

    factor: float
    solver: str
    nns: int
    mid: float
    sm: float
    hv: float
    cpu_time_sec: float


def run_sensitivity(
    domain: Domain,
    problem: object,
    config: RunConfig,
    factors: tuple[float, ...] = DEFAULT_FACTORS,
) -> list[SensitivityRow]:
    """Sweep FIS-output perturbations and return one row per (factor, solver).

    The unperturbed FIS scores are computed once; each ``factor`` rescales them
    via ``domain.perturb`` before re-running the solver(s) under ``config``.
    """
    base_cache = domain.precompute(problem)
    rows: list[SensitivityRow] = []
    for factor in factors:
        cache = domain.perturb(base_cache, factor)
        for front in run(problem, config, domain, cache=cache):
            m = compute_metrics(front)
            rows.append(
                SensitivityRow(
                    factor=factor,
                    solver=front.solver.value,
                    nns=m.nns,
                    mid=m.mid,
                    sm=m.sm,
                    hv=m.hv,
                    cpu_time_sec=m.cpu_time_sec,
                )
            )
    return rows


def write_sensitivity_csv(rows: list[SensitivityRow], output_dir: Path) -> Path:
    """Write the sensitivity sweep to ``sensitivity_<ts>.csv``. Returns the path."""
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = output_dir / f"sensitivity_{ts}.csv"
    pd.DataFrame([asdict(r) for r in rows]).to_csv(path, index=False)
    return path


def parse_factors(spec: str) -> tuple[float, ...]:
    """Parse a comma-separated ``--factors`` string into a tuple of floats."""
    try:
        factors = tuple(float(x) for x in spec.split(",") if x.strip() != "")
    except ValueError as exc:
        raise ValueError(f"--factors must be comma-separated numbers, got {spec!r}") from exc
    if not factors:
        raise ValueError("--factors must contain at least one value")
    return factors
