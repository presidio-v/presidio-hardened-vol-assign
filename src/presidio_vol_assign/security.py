"""Security module for presidio-hardened-vol-assign.

Responsibilities:
    - Structured JSON-lines logger (no PII — volunteer IDs and aggregates only)
    - Dependency CVE audit via pip-audit with a 24-hour on-disk cache
    - Startup security banner emitted at every CLI invocation

Public API:
    get_logger(log_path) -> StructuredLogger
    run_audit(cache_dir)  -> AuditResult
    log_startup(logger)   -> AuditResult
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from presidio_vol_assign import __version__

# ---------------------------------------------------------------------------
# Audit result
# ---------------------------------------------------------------------------

_AUDIT_CACHE_TTL_SECONDS = 86_400  # 24 hours
_DEFAULT_CACHE_DIR = Path.home() / ".pva"


class AuditStatus(str, Enum):
    OK = "ok"
    VULNERABLE = "vulnerable"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class AuditResult:
    status: AuditStatus
    n_vulnerabilities: int = 0
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    detail: str = ""

    @property
    def is_fresh(self) -> bool:
        age = (datetime.now(timezone.utc) - self.checked_at).total_seconds()
        return age < _AUDIT_CACHE_TTL_SECONDS

    def summary(self) -> str:
        ts = self.checked_at.strftime("%Y-%m-%d %H:%M UTC")
        if self.status == AuditStatus.OK:
            return f"OK (last checked: {ts}, 0 vulnerabilities)"
        if self.status == AuditStatus.VULNERABLE:
            return (
                f"WARNING — {self.n_vulnerabilities} known "
                f"vulnerabilit{'y' if self.n_vulnerabilities == 1 else 'ies'} "
                f"(last checked: {ts}; run pip-audit for details)"
            )
        if self.status == AuditStatus.SKIPPED:
            return f"SKIPPED ({self.detail})"
        return f"ERROR ({self.detail})"


# ---------------------------------------------------------------------------
# Dependency audit
# ---------------------------------------------------------------------------


def run_audit(cache_dir: Path = _DEFAULT_CACHE_DIR) -> AuditResult:
    """Run pip-audit and return an AuditResult.

    Results are cached to ``cache_dir/audit-cache.json`` for 24 hours so that
    normal CLI invocations do not incur a network round-trip every time.
    pip-audit is a dev dependency; when it is absent the result is SKIPPED.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "audit-cache.json"

    cached = _load_cache(cache_file)
    if cached is not None and cached.is_fresh:
        return cached

    result = _run_pip_audit()
    _save_cache(cache_file, result)
    return result


def _run_pip_audit() -> AuditResult:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--format", "json", "--progress-spinner", "off"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        return AuditResult(status=AuditStatus.SKIPPED, detail="pip-audit not installed")
    except subprocess.TimeoutExpired:
        return AuditResult(status=AuditStatus.ERROR, detail="pip-audit timed out after 60s")
    except Exception as exc:  # noqa: BLE001
        return AuditResult(status=AuditStatus.ERROR, detail=str(exc))

    if proc.returncode not in (0, 1):
        # exit code 1 = vulnerabilities found; anything else is an error
        return AuditResult(
            status=AuditStatus.ERROR,
            detail=f"pip-audit exited {proc.returncode}: {proc.stderr.strip()[:200]}",
        )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return AuditResult(status=AuditStatus.ERROR, detail="pip-audit returned unparseable output")

    # pip-audit JSON schema: {"dependencies": [{"vulns": [...]}]}
    n_vulns = sum(len(dep.get("vulns", [])) for dep in data.get("dependencies", []))
    if n_vulns > 0:
        return AuditResult(status=AuditStatus.VULNERABLE, n_vulnerabilities=n_vulns)
    return AuditResult(status=AuditStatus.OK)


def _load_cache(cache_file: Path) -> AuditResult | None:
    if not cache_file.exists():
        return None
    try:
        raw = json.loads(cache_file.read_text())
        return AuditResult(
            status=AuditStatus(raw["status"]),
            n_vulnerabilities=raw.get("n_vulnerabilities", 0),
            checked_at=datetime.fromisoformat(raw["checked_at"]),
            detail=raw.get("detail", ""),
        )
    except Exception:  # noqa: BLE001
        return None  # corrupt cache → re-run


def _save_cache(cache_file: Path, result: AuditResult) -> None:
    try:
        cache_file.write_text(
            json.dumps(
                {
                    "status": result.status.value,
                    "n_vulnerabilities": result.n_vulnerabilities,
                    "checked_at": result.checked_at.isoformat(),
                    "detail": result.detail,
                }
            )
        )
    except OSError:
        pass  # cache write failure is non-fatal


# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------


class StructuredLogger:
    """Append-only JSON-lines logger.

    Each log entry is one JSON object per line:
        {"ts": "<ISO8601>", "level": "INFO", "event": "...", "version": "0.1.0", ...}

    PII rule: never pass volunteer names, addresses, or contact details as
    keyword arguments. Use volunteer_id (opaque identifier) and numeric aggregates only.
    """

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, level: str, event: str, **extra: object) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "version": __version__,
            "event": event,
            **extra,
        }
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def info(self, event: str, **extra: object) -> None:
        self._write("INFO", event, **extra)

    def warning(self, event: str, **extra: object) -> None:
        self._write("WARNING", event, **extra)

    def error(self, event: str, **extra: object) -> None:
        self._write("ERROR", event, **extra)


def get_logger(log_path: Path | None = None) -> StructuredLogger:
    """Return a StructuredLogger writing to *log_path* (default: ./pva.log)."""
    return StructuredLogger(log_path or Path("pva.log"))


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------


def log_startup(logger: StructuredLogger, audit: AuditResult | None = None) -> AuditResult:
    """Emit the startup security banner and return the AuditResult.

    If *audit* is supplied (e.g. a pre-fetched result) it is used directly;
    otherwise ``run_audit()`` is called.  This allows the CLI to pass a cached
    result without triggering a second subprocess call.
    """
    if audit is None:
        audit = run_audit()

    logger.info(
        "presidio-hardened-vol-assign loaded",
        audit_status=audit.status.value,
        n_vulnerabilities=audit.n_vulnerabilities,
    )

    if audit.status == AuditStatus.VULNERABLE:
        logger.warning(
            "dependency audit found vulnerabilities",
            n_vulnerabilities=audit.n_vulnerabilities,
        )

    return audit
