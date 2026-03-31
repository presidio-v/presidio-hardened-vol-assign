# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Please report security vulnerabilities by opening a **private** GitHub Security Advisory
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
- **Path traversal guard** — `--output` paths are resolved to absolute form; `..` traversal is rejected
- **Secure logging** — volunteer IDs only; no names, addresses, or other PII in log output
- **Dependency audit** — `pip-audit` runs at startup and in CI; unpatched CVEs trigger a warning
- **No secrets via CLI** — API keys and credentials must be supplied via environment variables, never as CLI flags
