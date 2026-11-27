"""Compare a fresh run of the Paper A correction with the released results.

Paper A: Rabiei, Arias-Aranda, Stantchev, Appl. Sci. 2026, 16, 7581.
Released code and results: tag v0.5.1 of presidio-v/presidio-hardened-vol-assign.

Checks, after `sh experiments/run_correction.sh <archive>` has run in a clone of v0.5.1:
  1. The original objective, rerun with v0.5.1, reproduces the archived v0.2.0 manifest.
  2. Every result file of the correction that the tag ships is reproduced by the rerun.
Timing columns (CPU and wall time) depend on the machine and are reported, not compared.

Standard library only. Usage:
  python verify_correction.py --clone <v0.5.1 clone> \
      --archive <unzipped archived manifests> [--quick]
Writes verification-report.txt into the clone and exits 1 if any value differs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import platform
import subprocess
import sys
from pathlib import Path

TAG = "v0.5.1"
TIMING = {"cpu_time_sec", "wall_time_sec", "cpu_time_sec_measured", "cpu_avg"}
# Tables built from the corrected runs carry the CPU time measured on this machine.
LOCAL_CPU_TABLES = {"corrected_table3.csv", "corrected_pvalues.csv"}
CORR = "experiments/results/correction"


def read_csv(text: str) -> tuple[list[str], list[list[str]]]:
    rows = list(csv.reader(io.StringIO(text)))
    return rows[0], rows[1:]


def compare(name: str, ref_text: str, new_text: str, skip: set[str]) -> tuple[bool, str]:
    ref_head, ref_rows = read_csv(ref_text)
    new_head, new_rows = read_csv(new_text)
    if ref_head != new_head:
        return False, f"{name}: header differs"
    if len(ref_rows) != len(new_rows):
        return (
            False,
            f"{name}: {len(ref_rows)} reference rows, {len(new_rows)} new rows",
        )
    cols = [i for i, h in enumerate(ref_head) if h not in skip]
    n_diff, max_abs, first = 0, 0.0, ""
    for r, (a, b) in enumerate(zip(ref_rows, new_rows), start=2):
        for i in cols:
            if a[i] == b[i]:
                continue
            n_diff += 1
            try:
                max_abs = max(max_abs, abs(float(a[i]) - float(b[i])))
            except ValueError:
                max_abs = float("inf")
            if not first:
                first = f" first at line {r}, column {ref_head[i]}: {a[i]!r} vs {b[i]!r}"
    compared = len(ref_rows) * len(cols)
    skipped = sorted(set(ref_head) & skip)
    note = f" (not compared: {', '.join(skipped)})" if skipped else ""
    if n_diff == 0:
        return True, f"{name}: IDENTICAL, {compared} values{note}"
    return (
        False,
        f"{name}: {n_diff} of {compared} values differ, max |diff| {max_abs:.3g};{first}{note}",
    )


def git(clone: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(clone), *args], check=True, capture_output=True, text=True
    ).stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", type=Path, required=True)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="only the H1/H2/H4 manifests (see VERIFY.md, step 4b)",
    )
    a = ap.parse_args()
    clone, archive = a.clone.resolve(), a.archive.resolve()

    lines = [
        "Paper A correction: verification against tag " + TAG,
        f"Commit: {git(clone, 'rev-parse', 'HEAD').strip()}",
        f"Tag {TAG} points to: {git(clone, 'rev-parse', TAG + '^{commit}').strip()}",
        f"Python {platform.python_version()} on {platform.platform()}",
        "",
        "1. Original objective, rerun with v0.5.1, against the archived v0.2.0 manifest",
    ]
    ok = True
    ref = (archive / "h1_h2_h4" / "manifest.csv").read_text()
    new = (clone / CORR / "repro_directed/h1_h2_h4/manifest.csv").read_text()
    good, msg = compare("repro_directed/h1_h2_h4/manifest.csv", ref, new, TIMING)
    ok &= good
    lines += [
        "   " + msg,
        "",
        f"2. Correction results shipped with {TAG}, against the rerun",
    ]

    shipped = [p for p in git(clone, "ls-files", CORR).split() if p.endswith(".csv")]
    if a.quick:
        shipped = [p for p in shipped if p.endswith("h1_h2_h4/manifest.csv")]
        lines.append("   (quick check: H1/H2/H4 manifests only)")
    for path in sorted(shipped):
        name = path.removeprefix(CORR + "/")
        skip = set(TIMING)
        if Path(path).name in LOCAL_CPU_TABLES:
            skip.add("cpu_time_sec")
        new_file = clone / path
        if not new_file.exists():
            ok = False
            lines.append(f"   {name}: MISSING from the rerun")
            continue
        good, msg = compare(name, git(clone, "show", f"{TAG}:{path}"), new_file.read_text(), skip)
        ok &= good
        lines.append("   " + msg)

    lines += ["", "SHA-256 of the rerun result files"]
    for path in sorted(shipped):
        f = clone / path
        if f.exists():
            lines.append(f"   {hashlib.sha256(f.read_bytes()).hexdigest()}  {path}")
    lines += ["", "RESULT: " + ("ALL VALUES REPRODUCED" if ok else "DIFFERENCES FOUND")]

    report = "\n".join(lines) + "\n"
    (clone / "verification-report.txt").write_text(report)
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
