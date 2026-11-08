# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| 0.1.x   | No        |

## Reporting a Vulnerability

Please report security vulnerabilities by opening a private GitHub Security Advisory
(via the "Security" tab → "Report a vulnerability") rather than a public issue.

Include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive an acknowledgement within 5 business days. We aim to release a patch
within 30 days of a confirmed vulnerability.

## Security Design

`presidio-hardened-vol-assign` applies the following hardening measures:

- **CSV input sanitization** — all input files are validated for schema, types, and value ranges before processing
- **CSV output safety** — record IDs written to result CSVs are neutralised against spreadsheet formula injection (a leading `'` is added to cells beginning with `= + - @`)
- **Path traversal guard** — `--output` paths are resolved to absolute form; `..` traversal is rejected
- **Secure logging** — record IDs only (volunteer/person); no names, addresses, or other PII in log output
- **Dependency audit** — `pip-audit` runs at startup and in CI; unpatched CVEs trigger a warning
- **No secrets via CLI** — API keys and credentials must be supplied via environment variables, never as CLI flags

## Known Limitations

- On Python 3.10, matplotlib's transitive `pillow` stays at 12.3.0, which carries
  35 advisories (CVE-2026-96636..96778, CVE-2026-112390..112901,
  GHSA-78g4-xqwv-q3g8). The fix, pillow 13.0.0, requires Python 3.11, so the
  `pillow>=13.0.0` floor applies on 3.11+ only. vol-assign uses pillow only to
  write PNG charts (`savefig`) and never decodes untrusted images, but 3.10
  users should upgrade Python. The 3.10 CI audit ignores exactly these IDs
  (`.github/pip-audit-py310-ignore.txt`); the 3.11 and 3.12 audits ignore nothing.

## Software Development Lifecycle

This repository is developed under the Presidio hardened-family SDLC. The public report
— scope, standards mapping, threat-model gates, and supply-chain controls — is at
<https://github.com/presidio-v/presidio-hardened-docs/blob/main/sdlc/sdlc-report.md>.
