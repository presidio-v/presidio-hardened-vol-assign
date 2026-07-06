"""Unit tests for the evidence module (canonicalisation, sign/verify, determinism)."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from presidio_vol_assign import __version__
from presidio_vol_assign.evidence import (
    ALG_HMAC,
    SCHEMA,
    BadSignatureError,
    FloatLeakError,
    HashMismatchError,
    SchemaMismatchError,
    UnknownSignerError,
    build_record,
    canonical_bytes,
    content_hash,
    float_to_decimal_str,
    load_trust_store,
    seal,
    verify_record,
)

_FIXED_TS = datetime(2026, 7, 5, 12, 0, 0, tzinfo=timezone.utc)
_HMAC_SECRET = bytes.fromhex("aa" * 32)
_SIGNER = "test-signer"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    people = tmp_path / "people.csv"
    people.write_text(
        "person_id,vulnerability,mobility,group_size,distance_center_C1\n"
        "P1,9.0,2.0,3,5.0\n"
        "P2,4.0,7.0,1,12.0\n"
    )
    centers = tmp_path / "centers.csv"
    centers.write_text("center_id,capacity,service_level,road_accessibility\nC1,120,8.0,7.0\n")
    assigns = tmp_path / "assignments_nsga2_x.csv"
    assigns.write_text("solution_id,person_id,center_id\n0,P1,C1\n0,P2,C1\n")
    return people, centers, assigns


def _build(tmp_path: Path, **overrides) -> dict:
    people, centers, assigns = _write_inputs(tmp_path)
    kwargs = dict(
        model="humanitarian",
        tool_version=__version__,
        solver="nsga2",
        seed=42,
        pop_size=8,
        generations=4,
        input_csv_paths=[people, centers],
        objective_labels=("z1", "z2", "z3"),
        front_objectives=[(0.31245, 0.4213, 0.5), (0.29, 0.44, 0.61)],
        metrics={"nns": 2.0, "mid": 0.38, "sm": 0.041, "hv": 0.415},
        assignments_csv_path=assigns,
        emitter=_SIGNER,
        generated_at=_FIXED_TS,
    )
    kwargs.update(overrides)
    return build_record(**kwargs)


# ---------------------------------------------------------------------------
# Float encoding + canonical bytes float rejection
# ---------------------------------------------------------------------------


def test_float_to_decimal_str_round_trips() -> None:
    for v in (0.1, 0.31245, 1.0, 3.0, -0.2, 1e-9, 123456.789):
        s = float_to_decimal_str(v)
        assert isinstance(s, str)
        assert float(s) == v


def test_float_to_decimal_str_rejects_non_finite() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            float_to_decimal_str(bad)


def test_canonical_bytes_rejects_bare_float() -> None:
    with pytest.raises(FloatLeakError):
        canonical_bytes({"x": 1.5})
    with pytest.raises(FloatLeakError):
        canonical_bytes({"nested": {"list": [1, 2, 3.0]}})


def test_canonical_bytes_allows_int_bool_str() -> None:
    # Ints and bools are fine; strings (incl. decimal-string-encoded floats) are fine.
    b = canonical_bytes({"n": 3, "flag": True, "x": "0.5", "s": "café"})
    assert b == '{"flag":true,"n":3,"s":"café","x":"0.5"}'.encode()


def test_canonical_bytes_sorted_and_compact() -> None:
    b = canonical_bytes({"b": "1", "a": "2"})
    assert b == b'{"a":"2","b":"1"}'


# ---------------------------------------------------------------------------
# Record shape
# ---------------------------------------------------------------------------


def test_record_content_has_no_bare_floats(tmp_path: Path) -> None:
    record = _build(tmp_path)
    # canonical_bytes over content would raise if a bare float leaked in.
    canonical_bytes(record["content"])
    # Objectives and metrics are decimal strings.
    assert record["content"]["metrics"]["mid"] == "0.38"
    assert record["content"]["pareto_front"][0]["objectives"][0] == "0.31245"


def test_record_schema_and_parents(tmp_path: Path) -> None:
    record = _build(tmp_path)
    assert record["content"]["schema"] == SCHEMA
    # parents = the input snapshot hashes
    input_hashes = [snap["sha256"] for snap in record["content"]["inputs"]]
    assert record["parents"] == input_hashes
    assert len(record["parents"]) == 2


def test_generated_at_and_emitter_in_envelope_not_content(tmp_path: Path) -> None:
    record = _build(tmp_path)
    assert "generated_at" in record and "generated_at" not in record["content"]
    assert "emitter" in record and "emitter" not in record["content"]
    assert record["generated_at"] == "2026-07-05T12:00:00Z"


# ---------------------------------------------------------------------------
# Determinism of content_hash (fixed seed / inputs → byte-identical hash)
# ---------------------------------------------------------------------------


def test_content_hash_deterministic_across_builds(tmp_path: Path) -> None:
    r1 = _build(tmp_path)
    # Different generated_at should NOT change content_hash (it's in the envelope).
    r2 = _build(tmp_path, generated_at=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert r1["content_hash"] == r2["content_hash"]
    assert content_hash(r1["content"]) == r1["content_hash"]


def test_assignments_filename_is_volatile_not_hashed(tmp_path: Path) -> None:
    """A timestamped assignments filename must not perturb content_hash."""
    record = _build(tmp_path)
    # The name lives in the envelope; only the digest is in content.
    assert "assignments_file" in record
    assert "filename" not in record["content"]["assignments"]
    assert "sha256" in record["content"]["assignments"]

    # Rebuild with the assignments file renamed but bytes unchanged.
    people, centers, assigns = _write_inputs(tmp_path)
    renamed = tmp_path / "assignments_nsga2_DIFFERENT_TS.csv"
    renamed.write_bytes(assigns.read_bytes())
    record2 = _build(tmp_path, assignments_csv_path=renamed)
    assert record["content_hash"] == record2["content_hash"]
    assert record["assignments_file"] != record2["assignments_file"]


def test_content_hash_changes_with_seed(tmp_path: Path) -> None:
    r1 = _build(tmp_path, seed=42)
    r2 = _build(tmp_path, seed=7)
    assert r1["content_hash"] != r2["content_hash"]


# ---------------------------------------------------------------------------
# Sign / verify roundtrip + tamper detection
# ---------------------------------------------------------------------------


def _trust_hmac() -> dict:
    return {_SIGNER: {"alg": ALG_HMAC, "secret": _HMAC_SECRET.hex()}}


def test_sign_verify_roundtrip_hmac(tmp_path: Path) -> None:
    record = _build(tmp_path)
    seal(record, signer=_SIGNER, alg=ALG_HMAC, key=_HMAC_SECRET)
    assert record["signer"] == _SIGNER
    assert record["alg"] == ALG_HMAC
    # Should not raise.
    verify_record(record, _trust_hmac())


def test_tamper_content_detected(tmp_path: Path) -> None:
    record = _build(tmp_path)
    seal(record, signer=_SIGNER, alg=ALG_HMAC, key=_HMAC_SECRET)
    tampered = copy.deepcopy(record)
    tampered["content"]["config"]["seed"] = 999  # change content, keep old hash
    with pytest.raises(HashMismatchError):
        verify_record(tampered, _trust_hmac())


def test_tamper_signature_detected(tmp_path: Path) -> None:
    record = _build(tmp_path)
    seal(record, signer=_SIGNER, alg=ALG_HMAC, key=_HMAC_SECRET)
    record["signature"] = "00" * 32
    with pytest.raises(BadSignatureError):
        verify_record(record, _trust_hmac())


def test_wrong_key_detected(tmp_path: Path) -> None:
    record = _build(tmp_path)
    seal(record, signer=_SIGNER, alg=ALG_HMAC, key=_HMAC_SECRET)
    other = {_SIGNER: {"alg": ALG_HMAC, "secret": ("bb" * 32)}}
    with pytest.raises(BadSignatureError):
        verify_record(record, other)


def test_unknown_signer_detected(tmp_path: Path) -> None:
    record = _build(tmp_path)
    seal(record, signer=_SIGNER, alg=ALG_HMAC, key=_HMAC_SECRET)
    with pytest.raises(UnknownSignerError):
        verify_record(record, {"someone-else": {"alg": ALG_HMAC, "secret": "aa"}})


def test_schema_mismatch_detected(tmp_path: Path) -> None:
    record = _build(tmp_path)
    record["content"]["schema"] = "other/schema@9"
    record["content_hash"] = content_hash(record["content"])
    seal(record, signer=_SIGNER, alg=ALG_HMAC, key=_HMAC_SECRET)
    with pytest.raises(SchemaMismatchError):
        verify_record(record, _trust_hmac())


def test_float_leak_detected_on_verify(tmp_path: Path) -> None:
    record = _build(tmp_path)
    seal(record, signer=_SIGNER, alg=ALG_HMAC, key=_HMAC_SECRET)
    # Inject a bare float into content — verification recomputes the hash and
    # canonical_bytes must reject the float first.
    record["content"]["metrics"]["mid"] = 0.38
    with pytest.raises(FloatLeakError):
        verify_record(record, _trust_hmac())


# ---------------------------------------------------------------------------
# Trust store loading
# ---------------------------------------------------------------------------


def test_load_trust_store(tmp_path: Path) -> None:
    p = tmp_path / "trust.json"
    p.write_text(json.dumps(_trust_hmac()))
    ts = load_trust_store(p)
    assert _SIGNER in ts and ts[_SIGNER]["alg"] == ALG_HMAC


def test_load_trust_store_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_trust_store(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# PII-freedom on a synthetic PII-bearing corpus
# ---------------------------------------------------------------------------


def test_evidence_is_pii_free(tmp_path: Path) -> None:
    """No person/centre identifier or attribute value appears in the evidence JSON.

    The corpus carries deliberately-distinctive tokens; the sealed record's bytes
    must contain only filenames, counts, and hashes — never the row contents.
    """
    people = tmp_path / "people.csv"
    people.write_text(
        "person_id,vulnerability,mobility,group_size,distance_center_C1\n"
        "SECRETPERSON001,9.7654,2.1234,3,5.6789\n"
        "SECRETPERSON002,4.4321,7.8765,1,12.3456\n"
    )
    centers = tmp_path / "centers.csv"
    centers.write_text(
        "center_id,capacity,service_level,road_accessibility\nSECRETCENTERAAA,120,8.1111,7.2222\n"
    )
    assigns = tmp_path / "assignments_nsga2_x.csv"
    assigns.write_text("solution_id,person_id,center_id\n0,SECRETPERSON001,SECRETCENTERAAA\n")

    record = build_record(
        model="humanitarian",
        tool_version=__version__,
        solver="nsga2",
        seed=42,
        pop_size=8,
        generations=4,
        input_csv_paths=[people, centers],
        objective_labels=("z1", "z2", "z3"),
        front_objectives=[(0.31245, 0.4213, 0.5)],
        metrics={"nns": 1.0, "mid": 0.38, "sm": 0.041, "hv": 0.415},
        assignments_csv_path=assigns,
        emitter=_SIGNER,
        generated_at=_FIXED_TS,
    )
    seal(record, signer=_SIGNER, alg=ALG_HMAC, key=_HMAC_SECRET)

    blob = canonical_bytes(record).decode("utf-8")
    for token in (
        "SECRETPERSON001",
        "SECRETPERSON002",
        "SECRETCENTERAAA",
        "9.7654",
        "2.1234",
        "8.1111",
        "12.3456",
    ):
        assert token not in blob, f"PII token leaked into evidence: {token}"

    # Filenames and counts DO appear; hashes are present.
    assert "people.csv" in blob
    assert "centers.csv" in blob
    assert record["content"]["inputs"][0]["row_count"] == 2
    assert record["content"]["inputs"][1]["row_count"] == 1
