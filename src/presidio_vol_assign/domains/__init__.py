"""Optimisation-domain adapters.

Each domain plugs a problem family (its FIS, encoding, objectives, and I/O
schema) into the shared evolutionary engine (``engine.py``). The engine itself
is domain-agnostic; only the operators and evaluator differ per domain.

Available domains:
    ed-staffing   — the original Rabiei et al. (2023) ED volunteer-staffing
                    model (2 objectives). Default; behaviour-identical to v0.1.0.
    humanitarian  — post-disaster allocation of affected people to relief
                    centres (3 objectives).  [added in v0.2.0]
"""

from __future__ import annotations

from presidio_vol_assign.domains.base import Domain
from presidio_vol_assign.domains.ed_staffing import EDStaffingDomain
from presidio_vol_assign.domains.humanitarian import HumanitarianDomain

# Registry mapping the CLI --model value to its domain factory.
_DOMAINS: dict[str, type[Domain]] = {
    EDStaffingDomain.name: EDStaffingDomain,
    HumanitarianDomain.name: HumanitarianDomain,
}


def get_domain(name: str) -> Domain:
    """Return a domain instance by its CLI name (e.g. ``"ed-staffing"``)."""
    try:
        return _DOMAINS[name]()
    except KeyError:
        raise ValueError(f"unknown model {name!r}; available: {sorted(_DOMAINS)!r}") from None


__all__ = ["Domain", "EDStaffingDomain", "HumanitarianDomain", "get_domain"]
