"""CLI glue for evidence-carrying allocation.

Resolves signing key material from the environment (fail-closed), builds and
seals one evidence record per solver front, and writes it next to the other
result files as ``evidence_<solver>_<timestamp>.json``.

Key material (checked in this order):

* ``PVA_EVIDENCE_ED25519_KEY`` — 32-byte Ed25519 private-key seed (hex).
  Requires the optional ``crypto`` extra (``pyca/cryptography``).
* ``PVA_EVIDENCE_KEY`` — HMAC-SHA256 secret (hex). Stdlib-only, always available.

The signer identity defaults to ``PVA_EVIDENCE_SIGNER`` or ``"pva-local"``.
If ``--emit-evidence`` is set and no key is present, emission fails
(fail-closed): no unsigned record is ever written.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from presidio_vol_assign import __version__
from presidio_vol_assign.evidence import (
    ALG_ED25519,
    ALG_HMAC,
    SigningKeyError,
    build_record,
    canonical_bytes,
    seal,
)

if TYPE_CHECKING:
    from presidio_vol_assign.models import Metrics, ParetoFront

_ENV_ED25519 = "PVA_EVIDENCE_ED25519_KEY"
_ENV_HMAC = "PVA_EVIDENCE_KEY"
_ENV_SIGNER = "PVA_EVIDENCE_SIGNER"
_DEFAULT_SIGNER = "pva-local"


def resolve_signing_key() -> tuple[str, str, bytes]:
    """Resolve (signer, alg, key_bytes) from the environment. Fail-closed.

    Raises:
        SigningKeyError: if no usable key material is present, or a provided hex
            key is malformed.
    """
    signer = os.environ.get(_ENV_SIGNER, _DEFAULT_SIGNER)

    ed_hex = os.environ.get(_ENV_ED25519)
    if ed_hex:
        try:
            key = bytes.fromhex(ed_hex.strip())
        except ValueError as exc:
            raise SigningKeyError(f"{_ENV_ED25519} is not valid hex: {exc}") from exc
        return signer, ALG_ED25519, key

    hmac_hex = os.environ.get(_ENV_HMAC)
    if hmac_hex:
        try:
            key = bytes.fromhex(hmac_hex.strip())
        except ValueError as exc:
            raise SigningKeyError(f"{_ENV_HMAC} is not valid hex: {exc}") from exc
        if not key:
            raise SigningKeyError(f"{_ENV_HMAC} is empty")
        return signer, ALG_HMAC, key

    raise SigningKeyError(
        "no signing key found. Set PVA_EVIDENCE_KEY (hex, HMAC-SHA256) or "
        "PVA_EVIDENCE_ED25519_KEY (hex, requires the 'crypto' extra). "
        "Evidence emission is fail-closed: no unsigned record is written."
    )


def emit_evidence(
    *,
    front: ParetoFront,
    metrics: Metrics,
    model: str,
    solver: str,
    seed: int | None,
    pop_size: int,
    generations: int,
    input_csv_paths: list[Path],
    objective_labels: tuple[str, ...],
    assignments_csv_path: Path,
    output_dir: Path,
    signer: str,
    alg: str,
    key: bytes,
    generated_at: datetime | None = None,
) -> Path:
    """Build, seal, and write one evidence record for *front*. Returns the path.

    The output filename mirrors the sibling result files:
    ``evidence_<solver>_<timestamp>.json``.
    """
    ts = generated_at or datetime.now(timezone.utc)

    metric_map: dict[str, float] = {
        "nns": float(metrics.nns),
        "mid": float(metrics.mid),
        "sm": float(metrics.sm),
        "hv": float(metrics.hv),
    }

    record = build_record(
        model=model,
        tool_version=__version__,
        solver=solver,
        seed=seed,
        pop_size=pop_size,
        generations=generations,
        input_csv_paths=input_csv_paths,
        objective_labels=objective_labels,
        front_objectives=[tuple(sol.objectives) for sol in front.solutions],
        metrics=metric_map,
        assignments_csv_path=assignments_csv_path,
        emitter=signer,
        generated_at=ts,
    )
    seal(record, signer=signer, alg=alg, key=key)

    stamp = ts.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = output_dir / f"evidence_{solver}_{stamp}.json"
    # Write the record with the same canonical bytes used for hashing, so the
    # on-disk file is itself canonical and re-verifiable byte-for-byte.
    path.write_bytes(canonical_bytes(record))
    return path
