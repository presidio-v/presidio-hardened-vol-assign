"""Bit-for-bit reproducibility check.

The paper treats reproducibility as a first-class resilience criterion: given a
fixed seed, the tool must produce identical Pareto fronts on every run on stock
hardware. ``verify_reproducibility`` runs the same seeded configuration several
times and compares canonical front signatures.

This requires ``config.seed`` to be set; an unseeded run is non-deterministic by
design and cannot be reproducibility-checked.
"""

from __future__ import annotations

from presidio_vol_assign.domains.base import Domain
from presidio_vol_assign.engine import run
from presidio_vol_assign.metrics import fronts_signature, reproducibility_score
from presidio_vol_assign.models import ProblemInstance, ReproReport, RunConfig


def verify_reproducibility(
    problem: ProblemInstance,
    config: RunConfig,
    domain: Domain,
    n_runs: int = 2,
) -> ReproReport:
    """Run *domain* under *config* ``n_runs`` times and compare front signatures.

    Returns a :class:`ReproReport` with ``rep == 1.0`` iff every run produced an
    identical combined-front signature.

    Raises:
        ValueError: If ``config.seed`` is None (nothing to reproduce) or
            ``n_runs < 2``.
    """
    if config.seed is None:
        raise ValueError("reproducibility check requires config.seed to be set")
    if n_runs < 2:
        raise ValueError(f"n_runs must be >= 2, got {n_runs}")

    signatures = [fronts_signature(run(problem, config, domain)) for _ in range(n_runs)]
    rep = reproducibility_score(signatures)
    return ReproReport(
        n_runs=n_runs,
        rep=rep,
        signature=signatures[0],
        identical=rep == 1.0,
    )
