"""Tests for the security module."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from presidio_vol_assign.security import (
    AuditResult,
    AuditStatus,
    StructuredLogger,
    _load_cache,
    _run_pip_audit,
    _save_cache,
    get_logger,
    log_startup,
    run_audit,
)

# ---------------------------------------------------------------------------
# AuditResult
# ---------------------------------------------------------------------------


def test_audit_result_summary_ok() -> None:
    r = AuditResult(status=AuditStatus.OK)
    assert "OK" in r.summary()
    assert "0 vulnerabilities" in r.summary()


def test_audit_result_summary_vulnerable() -> None:
    r = AuditResult(status=AuditStatus.VULNERABLE, n_vulnerabilities=3)
    assert "WARNING" in r.summary()
    assert "3" in r.summary()


def test_audit_result_summary_skipped() -> None:
    r = AuditResult(status=AuditStatus.SKIPPED, detail="pip-audit not installed")
    assert "SKIPPED" in r.summary()
    assert "pip-audit not installed" in r.summary()


def test_audit_result_summary_error() -> None:
    r = AuditResult(status=AuditStatus.ERROR, detail="connection refused")
    assert "ERROR" in r.summary()
    assert "connection refused" in r.summary()


def test_audit_result_is_fresh() -> None:
    recent = AuditResult(
        status=AuditStatus.OK,
        checked_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    assert recent.is_fresh

    old = AuditResult(
        status=AuditStatus.OK,
        checked_at=datetime.now(timezone.utc) - timedelta(hours=25),
    )
    assert not old.is_fresh


def test_audit_result_singular_vulnerability() -> None:
    r = AuditResult(status=AuditStatus.VULNERABLE, n_vulnerabilities=1)
    assert "vulnerability" in r.summary()
    assert "vulnerabilities" not in r.summary()


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------


def test_cache_round_trip(tmp_path: Path) -> None:
    cache_file = tmp_path / "audit-cache.json"
    original = AuditResult(status=AuditStatus.OK, n_vulnerabilities=0)
    _save_cache(cache_file, original)

    loaded = _load_cache(cache_file)
    assert loaded is not None
    assert loaded.status == AuditStatus.OK
    assert loaded.n_vulnerabilities == 0


def test_cache_missing_returns_none(tmp_path: Path) -> None:
    assert _load_cache(tmp_path / "nonexistent.json") is None


def test_cache_corrupt_returns_none(tmp_path: Path) -> None:
    bad = tmp_path / "audit-cache.json"
    bad.write_text("not-valid-json{{{")
    assert _load_cache(bad) is None


# ---------------------------------------------------------------------------
# run_audit — uses fresh cache (no subprocess)
# ---------------------------------------------------------------------------


def test_run_audit_uses_fresh_cache(tmp_path: Path) -> None:
    cache_file = tmp_path / "audit-cache.json"
    fresh = AuditResult(
        status=AuditStatus.OK,
        checked_at=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    _save_cache(cache_file, fresh)

    with patch("presidio_vol_assign.security._run_pip_audit") as mock_audit:
        result = run_audit(cache_dir=tmp_path)

    mock_audit.assert_not_called()
    assert result.status == AuditStatus.OK


def test_run_audit_refreshes_stale_cache(tmp_path: Path) -> None:
    cache_file = tmp_path / "audit-cache.json"
    stale = AuditResult(
        status=AuditStatus.OK,
        checked_at=datetime.now(timezone.utc) - timedelta(hours=25),
    )
    _save_cache(cache_file, stale)

    fresh_result = AuditResult(status=AuditStatus.OK)
    with patch("presidio_vol_assign.security._run_pip_audit", return_value=fresh_result):
        result = run_audit(cache_dir=tmp_path)

    assert result.status == AuditStatus.OK


def test_run_audit_no_cache_calls_pip_audit(tmp_path: Path) -> None:
    ok = AuditResult(status=AuditStatus.OK)
    with patch("presidio_vol_assign.security._run_pip_audit", return_value=ok) as mock_audit:
        result = run_audit(cache_dir=tmp_path)

    mock_audit.assert_called_once()
    assert result.status == AuditStatus.OK


# ---------------------------------------------------------------------------
# _run_pip_audit — subprocess paths
# ---------------------------------------------------------------------------


def test_run_pip_audit_file_not_found() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = _run_pip_audit()
    assert result.status == AuditStatus.SKIPPED
    assert "pip-audit not installed" in result.detail


def test_run_pip_audit_timeout() -> None:
    with patch(
        "subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="pip_audit", timeout=60)
    ):
        result = _run_pip_audit()
    assert result.status == AuditStatus.ERROR
    assert "timed out" in result.detail


def test_run_pip_audit_bad_return_code() -> None:
    proc = MagicMock()
    proc.returncode = 2
    proc.stderr = "something went wrong"
    with patch("subprocess.run", return_value=proc):
        result = _run_pip_audit()
    assert result.status == AuditStatus.ERROR
    assert "2" in result.detail


def test_run_pip_audit_invalid_json() -> None:
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = "not-json{"
    with patch("subprocess.run", return_value=proc):
        result = _run_pip_audit()
    assert result.status == AuditStatus.ERROR
    assert "unparseable" in result.detail


def test_run_pip_audit_vulnerable() -> None:
    proc = MagicMock()
    proc.returncode = 1
    proc.stdout = json.dumps(
        {
            "dependencies": [
                {"name": "pkg", "vulns": [{"id": "CVE-1234"}]},
                {"name": "safe", "vulns": []},
            ]
        }
    )
    with patch("subprocess.run", return_value=proc):
        result = _run_pip_audit()
    assert result.status == AuditStatus.VULNERABLE
    assert result.n_vulnerabilities == 1


def test_run_pip_audit_ok() -> None:
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = json.dumps({"dependencies": [{"name": "pkg", "vulns": []}]})
    with patch("subprocess.run", return_value=proc):
        result = _run_pip_audit()
    assert result.status == AuditStatus.OK


# ---------------------------------------------------------------------------
# _save_cache OSError
# ---------------------------------------------------------------------------


def test_save_cache_oserror_is_silent(tmp_path: Path) -> None:
    cache_file = tmp_path / "audit-cache.json"
    result = AuditResult(status=AuditStatus.OK)
    with patch("pathlib.Path.write_text", side_effect=OSError("disk full")):
        # Must not raise
        _save_cache(cache_file, result)


# ---------------------------------------------------------------------------
# StructuredLogger
# ---------------------------------------------------------------------------


def test_logger_error_method(tmp_path: Path) -> None:
    log = tmp_path / "pva.log"
    logger = StructuredLogger(log)
    logger.error("something broke", code=42)
    entry = json.loads(log.read_text().strip())
    assert entry["level"] == "ERROR"
    assert entry["event"] == "something broke"
    assert entry["code"] == 42


def test_logger_writes_json_lines(tmp_path: Path) -> None:
    log = tmp_path / "pva.log"
    logger = StructuredLogger(log)
    logger.info("startup", n_volunteers=5)
    logger.warning("audit warning", n_vulnerabilities=1)

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2

    entry0 = json.loads(lines[0])
    assert entry0["level"] == "INFO"
    assert entry0["event"] == "startup"
    assert entry0["n_volunteers"] == 5
    assert "ts" in entry0
    assert "version" in entry0

    entry1 = json.loads(lines[1])
    assert entry1["level"] == "WARNING"
    assert entry1["n_vulnerabilities"] == 1


def test_logger_creates_parent_dirs(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "dir" / "pva.log"
    logger = StructuredLogger(log)
    logger.info("test")
    assert log.exists()


def test_logger_appends(tmp_path: Path) -> None:
    log = tmp_path / "pva.log"
    logger = StructuredLogger(log)
    logger.info("first")
    logger.info("second")
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2


def test_get_logger_default_path() -> None:
    logger = get_logger()
    assert logger._path == Path("pva.log")


def test_get_logger_custom_path(tmp_path: Path) -> None:
    log = tmp_path / "custom.log"
    logger = get_logger(log)
    assert logger._path == log


# ---------------------------------------------------------------------------
# log_startup
# ---------------------------------------------------------------------------


def test_log_startup_ok(tmp_path: Path) -> None:
    logger = StructuredLogger(tmp_path / "pva.log")
    audit = AuditResult(status=AuditStatus.OK)
    result = log_startup(logger, audit=audit)

    assert result.status == AuditStatus.OK
    lines = (tmp_path / "pva.log").read_text().strip().splitlines()
    assert len(lines) == 1  # only INFO startup, no warning
    entry = json.loads(lines[0])
    assert entry["audit_status"] == "ok"


def test_log_startup_vulnerable_emits_warning(tmp_path: Path) -> None:
    logger = StructuredLogger(tmp_path / "pva.log")
    audit = AuditResult(status=AuditStatus.VULNERABLE, n_vulnerabilities=2)
    log_startup(logger, audit=audit)

    lines = (tmp_path / "pva.log").read_text().strip().splitlines()
    assert len(lines) == 2  # startup INFO + WARNING
    levels = [json.loads(line)["level"] for line in lines]
    assert "WARNING" in levels


def test_log_startup_calls_run_audit_when_no_audit_given(tmp_path: Path) -> None:
    logger = StructuredLogger(tmp_path / "pva.log")
    ok = AuditResult(status=AuditStatus.OK)
    with patch("presidio_vol_assign.security.run_audit", return_value=ok) as mock_run:
        log_startup(logger)
    mock_run.assert_called_once()
