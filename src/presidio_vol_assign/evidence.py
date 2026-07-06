"""Evidence-carrying allocation — a local Layer-0/1 mirror of the presidio-evidence family.

Every allocation run may emit a signed, content-addressed *evidence record* that
binds the run's inputs (as hashes), the rule-base / solver configuration, the
seed, the resulting Pareto front, and the assignments (as a digest) into one
canonical, offline-verifiable document. This is the humanitarian instantiation
of evidence-carrying decisions (computational jurisprudence; Stantchev 2026,
arXiv, ID pending).

What the record proves and does *not* prove is stated in the README section
"Evidence-carrying allocation". In short: it proves *this configuration produced
this front from these inputs*, verifiable by any third party with only the
record and the signer's public key / secret. It does **not** prove that the
model is correct, nor that the executing host was uncompromised.

Family conventions mirrored here (``presidio-evidence`` / ``evidence-ref@1``):

* **Canonical JSON** — sorted keys; ``separators=(",", ":")``; UTF-8;
  ``ensure_ascii=False``. **Bare floats are rejected**: any ``float`` reachable
  in the content raises :class:`FloatLeakError`. Solver objectives *are* floats,
  so every numeric value that enters the content is first encoded as its
  shortest round-trip decimal **string** via :func:`float_to_decimal_str`
  (``repr``), deterministically. This follows the arch-translucency precedent
  (their ADR-0010).
* **Content addressing** — SHA-256 over ``canonical_bytes(content)``.
* **Signature** — detached Ed25519 (via the optional ``crypto`` extra /
  ``pyca/cryptography``) or HMAC-SHA256, over
  ``canonical_bytes({"content_hash": ..., "signer": ...})``.
* **Trust store** — ``{signer: {"alg": ..., "public_key"|"secret": ...}}``.
* **Verification is fail-closed** — any anomaly returns/raises a distinct,
  named reason and never a "pass by default".

The **content / envelope split** for determinism: the volatile ``generated_at``
timestamp and the emitter identity live in the record ``content`` too, but the
determinism guarantee is scoped to the *hashed content excluding* volatile
fields. See :func:`build_record` — the ``content_hash`` is computed over the
content with ``generated_at`` and ``emitter`` moved out into the envelope, so a
fixed seed reproduces a byte-identical ``content_hash`` across runs. This
dovetails with the repo's existing bit-for-bit reproducibility (REP) claim.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "presidio-hardened/allocation-evidence@1"

# Signature algorithms.
ALG_HMAC = "hmac-sha256"
ALG_ED25519 = "ed25519"


# ---------------------------------------------------------------------------
# Errors (each verification failure has a distinct, named reason)
# ---------------------------------------------------------------------------


class EvidenceError(Exception):
    """Base class for all evidence errors."""


class FloatLeakError(EvidenceError):
    """A bare float reached canonical serialisation (must be a decimal string)."""


class HashMismatchError(EvidenceError):
    """content_hash does not match the SHA-256 of the canonical content."""


class UnknownSignerError(EvidenceError):
    """The record's signer is absent from the trust store."""


class BadSignatureError(EvidenceError):
    """The signature did not verify under the signer's key."""


class SchemaMismatchError(EvidenceError):
    """The record's schema is not the one this verifier understands."""


class SigningKeyError(EvidenceError):
    """Key material is missing or malformed (fail-closed: no unsigned emission)."""


# ---------------------------------------------------------------------------
# Float handling (arch-translucency ADR-0010 precedent)
# ---------------------------------------------------------------------------


def float_to_decimal_str(value: float) -> str:
    """Encode a float as its shortest round-trip decimal string, deterministically.

    ``repr(float)`` in Python 3 yields the shortest decimal string that round-trips
    to the same IEEE-754 double, so it is both minimal and reproducible across
    runs and platforms. Ints are rendered without a decimal point.

    Raises:
        ValueError: for non-finite floats (NaN / inf), which have no canonical
            decimal form and must never enter evidence content.
    """
    f = float(value)
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite value cannot be encoded in evidence: {value!r}")
    return repr(f)


# ---------------------------------------------------------------------------
# Canonical serialisation with float rejection
# ---------------------------------------------------------------------------


def _reject_floats(obj: Any, path: str = "$") -> None:
    """Recursively assert that *obj* contains no bare float. Fail-closed."""
    if isinstance(obj, bool):
        return  # bool is an int subclass; permitted
    if isinstance(obj, float):
        raise FloatLeakError(
            f"bare float at {path}: {obj!r}. Encode numbers as decimal strings "
            f"via float_to_decimal_str() before placing them in evidence content."
        )
    if isinstance(obj, dict):
        for key, val in obj.items():
            if not isinstance(key, str):
                raise EvidenceError(f"non-string dict key at {path}: {key!r}")
            _reject_floats(val, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            _reject_floats(val, f"{path}[{i}]")


def canonical_bytes(content: Any) -> bytes:
    """Serialise *content* to canonical JSON bytes (family convention).

    Sorted keys, compact separators, UTF-8, ``ensure_ascii=False``. Bare floats
    are rejected before serialisation.

    Raises:
        FloatLeakError: if any bare float is reachable in *content*.
    """
    _reject_floats(content)
    text = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return text.encode("utf-8")


def content_hash(content: Any) -> str:
    """Return the SHA-256 hex digest of ``canonical_bytes(content)``."""
    return hashlib.sha256(canonical_bytes(content)).hexdigest()


def sha256_hex(data: bytes) -> str:
    """SHA-256 hex digest of raw *data* bytes."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Signing / verification
# ---------------------------------------------------------------------------


def _signing_payload(content_hash_hex: str, signer: str) -> bytes:
    """The canonical bytes actually signed: binds the hash to the signer identity."""
    return canonical_bytes({"content_hash": content_hash_hex, "signer": signer})


def _ed25519_available() -> bool:
    try:
        import cryptography.hazmat.primitives.asymmetric.ed25519  # noqa: F401

        return True
    except ImportError:
        return False


def sign(content_hash_hex: str, signer: str, *, alg: str, key: bytes) -> str:
    """Produce a detached signature (hex) over the signing payload.

    Args:
        content_hash_hex: SHA-256 hex digest of the canonical content.
        signer: Signer identity (bound into the signed payload).
        alg: :data:`ALG_HMAC` or :data:`ALG_ED25519`.
        key: HMAC secret bytes, or an Ed25519 private-key seed (32 bytes).

    Raises:
        SigningKeyError: if *alg* is Ed25519 but the ``crypto`` extra is absent,
            or if key material is malformed.
    """
    payload = _signing_payload(content_hash_hex, signer)
    if alg == ALG_HMAC:
        return hmac.new(key, payload, hashlib.sha256).hexdigest()
    if alg == ALG_ED25519:
        if not _ed25519_available():
            raise SigningKeyError(
                "Ed25519 signing requires the optional 'crypto' extra "
                "(pip install 'presidio-hardened-vol-assign[crypto]')."
            )
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        try:
            private_key = Ed25519PrivateKey.from_private_bytes(key)
        except Exception as exc:  # noqa: BLE001
            raise SigningKeyError(f"invalid Ed25519 private key: {exc}") from exc
        return private_key.sign(payload).hex()
    raise SigningKeyError(f"unknown signature algorithm: {alg!r}")


def _verify_hmac(payload: bytes, secret: bytes, signature_hex: str) -> bool:
    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_hex)


def _verify_ed25519(payload: bytes, public_key_hex: str, signature_hex: str) -> bool:
    if not _ed25519_available():
        raise SigningKeyError(
            "Ed25519 verification requires the optional 'crypto' extra "
            "(pip install 'presidio-hardened-vol-assign[crypto]')."
        )
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
    except Exception as exc:  # noqa: BLE001
        raise SigningKeyError(f"invalid Ed25519 public key in trust store: {exc}") from exc
    try:
        public_key.verify(bytes.fromhex(signature_hex), payload)
        return True
    except InvalidSignature:
        return False


# ---------------------------------------------------------------------------
# Trust store
# ---------------------------------------------------------------------------


def load_trust_store(path: Path) -> dict[str, dict[str, str]]:
    """Load a trust store: ``{signer: {"alg": ..., "public_key"|"secret": ...}}``.

    HMAC entries carry ``"secret"`` (hex); Ed25519 entries carry ``"public_key"``
    (hex). Values are validated shallowly; malformed entries surface at verify time.

    Raises:
        FileNotFoundError: if *path* does not exist.
        EvidenceError: if the file is not a JSON object of the expected shape.
    """
    if not path.exists():
        raise FileNotFoundError(f"trust store not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"trust store is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise EvidenceError("trust store must be a JSON object keyed by signer")
    for signer, entry in data.items():
        if not isinstance(entry, dict) or "alg" not in entry:
            raise EvidenceError(f"trust-store entry for {signer!r} is missing 'alg'")
    return data


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------


def _input_snapshot(csv_path: Path) -> dict[str, Any]:
    """Return a PII-free snapshot of a CSV input: filename, byte-hash, row count.

    Only the *filename* (basename), the SHA-256 of the raw bytes, and the number
    of data rows are recorded — never any cell contents. people.csv carries
    vulnerability data; only its hash and row count ever leave the process.
    """
    raw = csv_path.read_bytes()
    # Row count = non-empty lines minus the header. Robust to a trailing newline.
    lines = [ln for ln in raw.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    row_count = max(len(lines) - 1, 0)
    return {
        "filename": csv_path.name,
        "sha256": sha256_hex(raw),
        "row_count": row_count,
    }


def assignments_digest(assignments_csv_path: Path) -> str:
    """SHA-256 of the canonical assignments payload = the raw bytes of the file.

    Person→centre mappings may be sensitive at scale, so the record carries only
    this digest plus the file *name*, never the inline rows.
    """
    return sha256_hex(Path(assignments_csv_path).read_bytes())


def _objectives_to_strings(objectives: tuple[float, ...]) -> list[str]:
    return [float_to_decimal_str(v) for v in objectives]


def build_record(
    *,
    model: str,
    tool_version: str,
    solver: str,
    seed: int | None,
    pop_size: int,
    generations: int,
    input_csv_paths: list[Path],
    objective_labels: tuple[str, ...],
    front_objectives: list[tuple[float, ...]],
    metrics: dict[str, float],
    assignments_csv_path: Path,
    emitter: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a complete evidence envelope (unsigned).

    The returned dict has this shape::

        {
          "content": { ... hashed, deterministic under a fixed seed ... },
          "content_hash": "<sha256 hex>",
          "generated_at": "<RFC3339 UTC>",       # volatile — NOT in content_hash
          "emitter": "<identity>",               # volatile — NOT in content_hash
          "assignments_file": "<filename>",      # volatile (timestamped) — NOT hashed
          "parents": [ "<input sha256>", ... ],
          "signer": null, "signature": null, "alg": null   # filled by seal()
        }

    Content / envelope split: ``generated_at``, ``emitter``, and the timestamped
    ``assignments_file`` name are volatile and live in the envelope, *outside* the
    hashed content, so a fixed seed and fixed inputs reproduce a byte-identical
    ``content_hash`` across runs. The assignments *digest* stays inside the
    content — it is the load-bearing binding; the filename merely names the file.

    All floats (objective values, metrics) are encoded as decimal strings; the
    content therefore contains no bare floats and passes ``canonical_bytes``.
    """
    snapshots = [_input_snapshot(p) for p in input_csv_paths]

    content: dict[str, Any] = {
        "schema": SCHEMA,
        "model": model,
        "tool_version": tool_version,
        "config": {
            "solver": solver,
            "seed": seed,
            "pop_size": pop_size,
            "generations": generations,
        },
        "inputs": snapshots,
        "objectives": list(objective_labels),
        "pareto_front": [{"objectives": _objectives_to_strings(obj)} for obj in front_objectives],
        # Only the digest binds the assignments — the filename is timestamped and
        # therefore volatile, so it lives in the envelope (see below), not here.
        "assignments": {
            "sha256": assignments_digest(assignments_csv_path),
        },
        "metrics": {k: float_to_decimal_str(v) for k, v in metrics.items()},
    }

    ts = generated_at or datetime.now(timezone.utc)
    generated_at_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "content": content,
        "content_hash": content_hash(content),
        "generated_at": generated_at_str,
        "emitter": emitter,
        # Provenance metadata that names, but does not bind, the assignments file.
        # Volatile (timestamped) → envelope, outside content_hash.
        "assignments_file": Path(assignments_csv_path).name,
        "parents": [snap["sha256"] for snap in snapshots],
        "signer": None,
        "signature": None,
        "alg": None,
    }


def seal(record: dict[str, Any], *, signer: str, alg: str, key: bytes) -> dict[str, Any]:
    """Sign *record* in place and return it (fail-closed on missing key).

    The signature is over ``canonical_bytes({"content_hash", "signer"})`` — the
    already-computed ``record["content_hash"]`` — so signing never re-derives the
    hash and cannot silently diverge from it.
    """
    signature = sign(record["content_hash"], signer, alg=alg, key=key)
    record["signer"] = signer
    record["signature"] = signature
    record["alg"] = alg
    return record


# ---------------------------------------------------------------------------
# Offline verification (fail-closed, distinct reasons)
# ---------------------------------------------------------------------------


def verify_record(record: dict[str, Any], trust_store: dict[str, dict[str, str]]) -> None:
    """Verify a sealed evidence record offline. Returns None on success.

    Fail-closed: raises a distinct, named exception for each failure mode:

    * :class:`SchemaMismatchError` — content schema is not :data:`SCHEMA`.
    * :class:`FloatLeakError` — a bare float is present in the content.
    * :class:`HashMismatchError` — ``content_hash`` != SHA-256 of the content.
    * :class:`UnknownSignerError` — signer absent from the trust store.
    * :class:`BadSignatureError` — signature does not verify (incl. alg mismatch).
    """
    content = record.get("content")
    if not isinstance(content, dict) or content.get("schema") != SCHEMA:
        found = content.get("schema") if isinstance(content, dict) else None
        raise SchemaMismatchError(f"expected schema {SCHEMA!r}, got {found!r}")

    # Float-leak check runs first via canonical_bytes (raises FloatLeakError).
    recomputed = content_hash(content)
    if not hmac.compare_digest(recomputed, str(record.get("content_hash", ""))):
        raise HashMismatchError("content_hash does not match the canonical content")

    signer = record.get("signer")
    if signer not in trust_store:
        raise UnknownSignerError(f"unknown signer: {signer!r}")

    entry = trust_store[signer]
    alg = entry.get("alg")
    if alg != record.get("alg"):
        raise BadSignatureError(
            f"algorithm mismatch: record {record.get('alg')!r} vs trust store {alg!r}"
        )

    payload = _signing_payload(record["content_hash"], signer)
    signature_hex = str(record.get("signature", ""))

    if alg == ALG_HMAC:
        secret = entry.get("secret")
        if not secret:
            raise BadSignatureError(f"trust-store HMAC entry for {signer!r} has no 'secret'")
        ok = _verify_hmac(payload, bytes.fromhex(secret), signature_hex)
    elif alg == ALG_ED25519:
        public_key = entry.get("public_key")
        if not public_key:
            raise BadSignatureError(f"trust-store Ed25519 entry for {signer!r} has no 'public_key'")
        ok = _verify_ed25519(payload, public_key, signature_hex)
    else:
        raise BadSignatureError(f"unsupported algorithm in trust store: {alg!r}")

    if not ok:
        raise BadSignatureError("signature did not verify under the signer's key")
