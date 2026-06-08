"""Output writers for the allocation module.

Three files are written per solver run:

    pareto_alloc_<solver>_<obj>obj_<ts>.csv      one row per Pareto-front solution
    allocations_alloc_<solver>_<obj>obj_<ts>.csv one row per (person → center) allocation
    metrics_alloc_<solver>_<obj>obj_<ts>.json    NNS, MID, SM, HV, cpu_time_sec

The pareto CSV is self-contained for downstream metrics recomputation: it
includes solver, objectives_count, and the per-objective columns. The
3-objective and 4-objective formulations write different column sets and
the reader infers the formulation from the CSV header.

Public API:
    write_allocation_pareto_csv(front, output_dir)       -> Path
    write_allocation_csv(front, output_dir)              -> Path
    write_allocation_metrics_json(metrics, output_dir)   -> Path
    load_allocation_pareto_csv(path)                     -> AllocationParetoFront
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from presidio_vol_assign.allocation.models import (
    Allocation,
    AllocationMetrics,
    AllocationParetoFront,
    AllocationSolution,
    AllocationSolverType,
)

# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_allocation_pareto_csv(front: AllocationParetoFront, output_dir: Path) -> Path:
    """Write Pareto-front objective values to CSV. Returns the file path."""
    ts = _timestamp()
    path = output_dir / f"pareto_alloc_{front.solver.value}_{front.objectives_count}obj_{ts}.csv"

    rows = []
    for i, sol in enumerate(front.solutions):
        row = {
            "solver": front.solver.value,
            "objectives_count": front.objectives_count,
            "solution_id": i,
            "mn_ulpp": round(sol.mn_ulpp, 6),
            "mn_cail": round(sol.mn_cail, 6),
        }
        if front.objectives_count == 4:
            row["mn_trd"] = round(sol.mn_trd, 6)
            row["mn_rpd"] = round(sol.mn_rpd, 6)
        else:
            row["mn_til"] = round(sol.mn_til, 6)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_allocation_csv(front: AllocationParetoFront, output_dir: Path) -> Path:
    """Write per-allocation details to CSV. Returns the file path."""
    ts = _timestamp()
    path = (
        output_dir / f"allocations_alloc_{front.solver.value}_{front.objectives_count}obj_{ts}.csv"
    )

    rows = []
    for i, sol in enumerate(front.solutions):
        for alloc in sol.allocations:
            row = {
                "solution_id": i,
                "person_id": alloc.person_id,
                "center_id": alloc.center_id,
                "ulpp": round(alloc.ulpp, 6),
                "cail_contrib": round(alloc.cail_contrib, 6),
            }
            if front.objectives_count == 4:
                row["trd"] = round(alloc.trd, 6)
                row["rpd"] = round(alloc.rpd, 6)
            else:
                row["til"] = round(alloc.til, 6)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_allocation_metrics_json(m: AllocationMetrics, output_dir: Path) -> Path:
    """Write metrics to JSON. Returns the file path."""
    ts = _timestamp()
    path = output_dir / f"metrics_alloc_{m.solver.value}_{m.objectives_count}obj_{ts}.json"

    data = {
        "solver": m.solver.value,
        "objectives_count": m.objectives_count,
        "nns": m.nns,
        "mid": round(m.mid, 6),
        "sm": round(m.sm, 6),
        "hv": round(m.hv, 6),
        "cpu_time_sec": round(m.cpu_time_sec, 3),
    }
    path.write_text(json.dumps(data, indent=2))
    return path


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def load_allocation_pareto_csv(path: Path) -> AllocationParetoFront:
    """Load a pareto CSV back into an AllocationParetoFront.

    Allocations are not recovered (only objective tuples are needed for
    `compute_allocation_metrics`); each solution holds an empty allocations
    list. The solver and objectives_count are read from the CSV.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: On missing required columns or unknown solver value.
    """
    if not path.exists():
        raise FileNotFoundError(f"Pareto CSV not found: {path}")

    df = pd.read_csv(path)
    base_cols = {"solver", "objectives_count", "solution_id", "mn_ulpp", "mn_cail"}
    _require_cols(df, base_cols, str(path))
    n_obj = int(df["objectives_count"].iloc[0])
    if n_obj == 4:
        _require_cols(df, {"mn_trd", "mn_rpd"}, str(path))
    else:
        _require_cols(df, {"mn_til"}, str(path))

    try:
        solver = AllocationSolverType(str(df["solver"].iloc[0]).strip())
    except ValueError:
        raise ValueError(
            f"Unknown solver value {df['solver'].iloc[0]!r} in {path}. "
            f"Expected one of: {[s.value for s in AllocationSolverType]}"
        )

    solutions: list[AllocationSolution] = []
    for _, row in df.iterrows():
        if n_obj == 4:
            sol = AllocationSolution(
                allocations=[Allocation(person_id="", center_id="")],
                objectives_count=4,
                mn_ulpp=float(row["mn_ulpp"]),
                mn_trd=float(row["mn_trd"]),
                mn_rpd=float(row["mn_rpd"]),
                mn_cail=float(row["mn_cail"]),
            )
        else:
            sol = AllocationSolution(
                allocations=[Allocation(person_id="", center_id="")],
                objectives_count=3,
                mn_ulpp=float(row["mn_ulpp"]),
                mn_til=float(row["mn_til"]),
                mn_cail=float(row["mn_cail"]),
            )
        solutions.append(sol)

    return AllocationParetoFront(
        solver=solver,
        objectives_count=n_obj,
        solutions=solutions,
    )


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
