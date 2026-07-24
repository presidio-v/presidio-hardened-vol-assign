"""Integration tests for `pva assign --emit-evidence` and `pva verify-evidence`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from presidio_vol_assign.cli import app
from presidio_vol_assign.evidence import ALG_ED25519, ALG_HMAC, SCHEMA

runner = CliRunner()

FIXTURES = Path(__file__).parent / "fixtures"
PEOPLE = str(FIXTURES / "people_valid.csv")
CENTERS = str(FIXTURES / "centers_valid.csv")
VOLUNTEERS = str(FIXTURES / "volunteers_valid.csv")
EDS = str(FIXTURES / "eds_valid.csv")

FAST = ["--pop-size", "8", "--generations", "4", "--seed", "42"]
_HMAC_HEX = "aa" * 32


def _assign_humanitarian(tmp_path: Path, env: dict | None = None, extra: list | None = None):
    return runner.invoke(
        app,
        [
            "assign",
            "--model",
            "humanitarian",
            "--people",
            PEOPLE,
            "--centers",
            CENTERS,
            "--solver",
            "nsga2",
            "--output",
            str(tmp_path),
        ]
        + FAST
        + (extra or []),
        env=env,
    )


# ---------------------------------------------------------------------------
# Default OFF — behaviour unchanged, no evidence file
# ---------------------------------------------------------------------------


def test_default_off_no_evidence_file(tmp_path: Path) -> None:
    result = _assign_humanitarian(tmp_path)
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("evidence_*.json")) == []
    # Normal result files still present.
    assert list(tmp_path.glob("pareto_nsga2_*.csv"))
    assert list(tmp_path.glob("assignments_nsga2_*.csv"))


# ---------------------------------------------------------------------------
# Fail-closed — flag set but no key present
# ---------------------------------------------------------------------------


def test_emit_evidence_fails_closed_without_key(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "assign",
            "--model",
            "humanitarian",
            "--people",
            PEOPLE,
            "--centers",
            CENTERS,
            "--solver",
            "nsga2",
            "--output",
            str(tmp_path),
            "--emit-evidence",
        ]
        + FAST,
        env={"PVA_EVIDENCE_KEY": "", "PVA_EVIDENCE_ED25519_KEY": ""},
    )
    assert result.exit_code == 1
    assert list(tmp_path.glob("evidence_*.json")) == []


# ---------------------------------------------------------------------------
# HMAC end-to-end: emit → verify OK
# ---------------------------------------------------------------------------


def test_emit_and_verify_hmac(tmp_path: Path) -> None:
    env = {"PVA_EVIDENCE_KEY": _HMAC_HEX, "PVA_EVIDENCE_SIGNER": "eoc-1"}
    result = _assign_humanitarian(tmp_path, env=env, extra=["--emit-evidence"])
    assert result.exit_code == 0, result.output

    evidence_files = list(tmp_path.glob("evidence_nsga2_*.json"))
    assert len(evidence_files) == 1
    record = json.loads(evidence_files[0].read_text())
    assert record["content"]["schema"] == SCHEMA
    assert record["signer"] == "eoc-1"
    assert record["alg"] == ALG_HMAC

    trust = tmp_path / "trust.json"
    trust.write_text(json.dumps({"eoc-1": {"alg": ALG_HMAC, "secret": _HMAC_HEX}}))

    vr = runner.invoke(
        app,
        ["verify-evidence", "--evidence", str(evidence_files[0]), "--trust", str(trust)],
    )
    assert vr.exit_code == 0, vr.output
    assert "Verify OK" in vr.output


def test_verify_rejects_tampered_record(tmp_path: Path) -> None:
    env = {"PVA_EVIDENCE_KEY": _HMAC_HEX, "PVA_EVIDENCE_SIGNER": "eoc-1"}
    _assign_humanitarian(tmp_path, env=env, extra=["--emit-evidence"])
    evidence_file = next(tmp_path.glob("evidence_nsga2_*.json"))

    record = json.loads(evidence_file.read_text())
    record["content"]["config"]["seed"] = 999
    evidence_file.write_text(json.dumps(record))

    trust = tmp_path / "trust.json"
    trust.write_text(json.dumps({"eoc-1": {"alg": ALG_HMAC, "secret": _HMAC_HEX}}))

    vr = runner.invoke(
        app,
        ["verify-evidence", "--evidence", str(evidence_file), "--trust", str(trust)],
    )
    assert vr.exit_code == 1
    assert "FAILED" in vr.output


# ---------------------------------------------------------------------------
# Determinism through the CLI: same seed → identical content_hash
# ---------------------------------------------------------------------------


def test_cli_content_hash_deterministic(tmp_path: Path) -> None:
    env = {"PVA_EVIDENCE_KEY": _HMAC_HEX}
    d1 = tmp_path / "run1"
    d2 = tmp_path / "run2"
    r1 = _assign_humanitarian(d1, env=env, extra=["--emit-evidence"])
    r2 = _assign_humanitarian(d2, env=env, extra=["--emit-evidence"])
    assert r1.exit_code == 0 and r2.exit_code == 0
    h1 = json.loads(next(d1.glob("evidence_nsga2_*.json")).read_text())["content_hash"]
    h2 = json.loads(next(d2.glob("evidence_nsga2_*.json")).read_text())["content_hash"]
    assert h1 == h2


# ---------------------------------------------------------------------------
# Ed25519 (only when the optional crypto extra is installed)
# ---------------------------------------------------------------------------


def _crypto_available() -> bool:
    try:
        import cryptography.hazmat.primitives.asymmetric.ed25519  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _crypto_available(), reason="crypto extra not installed")
def test_emit_and_verify_ed25519(tmp_path: Path) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    priv_seed = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_hex = (
        priv.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )

    env = {"PVA_EVIDENCE_ED25519_KEY": priv_seed.hex(), "PVA_EVIDENCE_SIGNER": "eoc-ed"}
    result = _assign_humanitarian(tmp_path, env=env, extra=["--emit-evidence"])
    assert result.exit_code == 0, result.output

    evidence_file = next(tmp_path.glob("evidence_nsga2_*.json"))
    record = json.loads(evidence_file.read_text())
    assert record["alg"] == ALG_ED25519

    trust = tmp_path / "trust.json"
    trust.write_text(json.dumps({"eoc-ed": {"alg": ALG_ED25519, "public_key": pub_hex}}))

    vr = runner.invoke(
        app,
        ["verify-evidence", "--evidence", str(evidence_file), "--trust", str(trust)],
    )
    assert vr.exit_code == 0, vr.output
