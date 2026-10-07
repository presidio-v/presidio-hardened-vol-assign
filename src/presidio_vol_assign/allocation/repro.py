"""Reproducibility signatures for allocation Pareto fronts (Paper B, RQ2).

This module owns one invariant: two solver runs are *the same* iff their
fronts hash to the same digest. The digest is canonicalised before hashing —
order-invariant over solutions, so solver output ordering cannot mask a
difference, yet sensitive to any change in an objective value or a
person->center assignment, so numeric drift cannot hide. Rounding precision
and the canonical ordering are pinned; the hash is SHA-256 only.

Evidence as a return value: callers get the digest (the receipt that a run
occurred with a given result) and a reproducibility verdict, not a bare
boolean buried in a log.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from presidio_vol_assign.allocation.models import AllocationParetoFront

# Pinned: objective rounding used when building the signature. A looser value
# would let genuine numeric drift collapse into an identical digest (a false
# "reproducible"); a tighter one would flag platform float noise as a change.
_SIGNATURE_PRECISION = 9


def allocation_front_signature(
    front: AllocationParetoFront,
    precision: int = _SIGNATURE_PRECISION,
) -> str:
    """Return the SHA-256 hex digest of a front's decisions and objectives.

    Each solution is canonicalised to its rounded objective tuple plus its
    sorted ``(person_id, center_id)`` assignment pairs; the solutions are then
    sorted. Ordering is pinned throughout, so the digest depends only on the
    content of the front, never on the order the solver emitted it in.
    """
    canonical: list[tuple] = []
    for solution in front.solutions:
        fitness = tuple(round(x, precision) for x in solution.fitness)
        assignments = tuple(sorted((a.person_id, a.center_id) for a in solution.allocations))
        canonical.append((fitness, assignments))
    canonical.sort()
    payload = repr((front.objectives_count, canonical)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rep_score(signatures: Sequence[str]) -> float:
    """Return 1.0 iff every run signature is identical, else 0.0.

    Fail closed: an empty set of runs has not demonstrated reproducibility, so
    it scores 0.0 — never a vacuous 1.0.
    """
    if not signatures:
        return 0.0
    first = signatures[0]
    return 1.0 if all(s == first for s in signatures) else 0.0
