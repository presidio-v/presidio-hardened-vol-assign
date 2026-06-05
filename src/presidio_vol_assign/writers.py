"""Output writers for presidio-hardened-vol-assign.

Two CSV files and one JSON file are written per solver run:

    pareto_<solver>_<ts>.csv     — one row per Pareto-front solution (z1, z2)
    assignments_<solver>_<ts>.csv — one row per assignment within each solution
    metrics_<solver>_<ts>.json   — NNS, MID, SM, HV, cpu_time_sec

The pareto CSV is self-contained for ``pva metrics``: it includes a ``solver``
column so metrics can be re-computed without the assignments file.

Public API:
    write_pareto_csv(front, output_dir)       -> Path
    write_assignments_csv(front, output_dir)  -> Path
    write_metrics_json(metrics, output_dir)   -> Path
    load_pareto_csv(path)                     -> ParetoFront
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from presidio_vol_assign.models import (
    Metrics,
    ParetoFront,
    Solution,
    SolverType,
)

if TYPE_CHECKING:
    from presidio_vol_assign.domains.base import Domain

_Z_COL_RE = re.compile(r"z\d+$")

# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_pareto_csv(front: ParetoFront, output_dir: Path) -> Path:
    """Write Pareto-front objective values to CSV. Returns the file path.

    Emits one ``z<k>`` column per objective, so the schema adapts to 2- or
    3-objective fronts automatically.
    """
    ts = _timestamp()
    path = output_dir / f"pareto_{front.solver.value}_{ts}.csv"

    rows = []
    for i, sol in enumerate(front.solutions):
        row = {"solver": front.solver.value, "solution_id": i}
        for k, value in enumerate(sol.objectives, start=1):
            row[f"z{k}"] = round(value, 6)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_assignments_csv(front: ParetoFront, output_dir: Path, domain: Domain) -> Path:
    """Write per-assignment details to CSV, using the domain's column schema.

    Returns the file path.
    """
    ts = _timestamp()
    path = output_dir / f"assignments_{front.solver.value}_{ts}.csv"

    rows = []
    for i, sol in enumerate(front.solutions):
        for asgn in sol.assignments:
            rows.append({"solution_id": i, **domain.assignment_row(asgn)})
    columns = ["solution_id", *domain.assignment_fieldnames]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def write_metrics_json(m: Metrics, output_dir: Path) -> Path:
    """Write metrics to JSON. Returns the file path."""
    ts = _timestamp()
    path = output_dir / f"metrics_{m.solver.value}_{ts}.json"

    data = {
        "solver": m.solver.value,
        "nns": m.nns,
        "mid": round(m.mid, 6),
        "sm": round(m.sm, 6),
        "hv": round(m.hv, 6),
        "cpu_time_sec": round(m.cpu_time_sec, 3),
    }
    if m.rep is not None:
        data["rep"] = round(m.rep, 6)
    path.write_text(json.dumps(data, indent=2))
    return path


# ---------------------------------------------------------------------------
# Reader (for pva metrics)
# ---------------------------------------------------------------------------


def load_pareto_csv(path: Path) -> ParetoFront:
    """Load a pareto CSV back into a ParetoFront (objectives only; no assignments).

    Detects ``z1, z2, ... zk`` objective columns automatically, so both 2- and
    3-objective fronts round-trip. The solver column determines the SolverType.

    Raises:
        ValueError: If the file is missing required columns or has an unknown solver.
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Pareto CSV not found: {path}")

    df = pd.read_csv(path)
    _require_cols(df, {"solver", "solution_id"}, str(path))

    z_cols = sorted(
        (c for c in df.columns if _Z_COL_RE.fullmatch(c)),
        key=lambda c: int(c[1:]),
    )
    if not z_cols:
        raise ValueError(
            f"{path}: no objective columns found (expected at least 'z1'). "
            f"Found: {sorted(df.columns)}"
        )

    try:
        solver = SolverType(str(df["solver"].iloc[0]).strip())
    except ValueError:
        raise ValueError(
            f"Unknown solver value {df['solver'].iloc[0]!r} in {path}. "
            f"Expected one of: {[s.value for s in SolverType]}"
        )

    solutions = [
        Solution(assignments=[], objectives=tuple(float(row[c]) for c in z_cols))
        for _, row in df.iterrows()
    ]
    return ParetoFront(solver=solver, solutions=solutions)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _require_cols(df: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source}: missing required columns {sorted(missing)}. Found: {sorted(df.columns)}"
        )
