"""Benchmark harness — reproducible multi-instance evaluation.

Generates the paper's problem instances deterministically, runs the solver(s) on
each, and aggregates the Pareto metrics into a Table-3-style summary
(mean +/- std for NNS, MID, SM, HV, CPU time) with an optional bit-for-bit
reproducibility (REP) column.

Instance sizes (per the paper):
    humanitarian: small = 5 centres / 150 people,   large = 10 / 300
    ed-staffing:  small = 5 EDs    / 75 volunteers, large = 10 / 150

Instance generation is fully seeded: instance ``i`` of a size class uses
``base_seed + size_offset + i`` so a benchmark run is reproducible end to end.
Generation happens in memory (no CSV round-trip).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from presidio_vol_assign.domains import get_domain
from presidio_vol_assign.engine import run
from presidio_vol_assign.metrics import compute_metrics, fronts_signature
from presidio_vol_assign.models import (
    Center,
    HumanitarianProblem,
    Person,
    ProblemInstance,
    RunConfig,
    SkillType,
    Vacancy,
    Volunteer,
)

# size -> (n_secondary_units, n_primary_units); meaning is model-specific.
_HUMANITARIAN_SIZES = {"small": (5, 150), "large": (10, 300)}  # (centres, people)
_ED_SIZES = {"small": (5, 75), "large": (10, 150)}  # (EDs, volunteers)

# Deterministic per-size seed offsets, so small/large instance streams never overlap.
_SIZE_OFFSET = {"small": 0, "large": 100_000}


@dataclass
class BenchmarkRow:
    """Aggregated metrics for one (model, size, solver) cell of the summary."""

    model: str
    size: str
    solver: str
    n_instances: int
    nns_mean: float
    nns_std: float
    mid_mean: float
    mid_std: float
    sm_mean: float
    sm_std: float
    hv_mean: float
    hv_std: float
    cpu_mean: float
    cpu_std: float
    rep: float | None = None
    # Per-instance HV values feeding the Wilcoxon rank-sum tests; kept in memory
    # for the stats layer and stripped from the written Table-3 summary.
    hv_samples: list[float] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Instance generators (deterministic)
# ---------------------------------------------------------------------------


def _generate_humanitarian(n_centers: int, n_people: int, seed: int) -> HumanitarianProblem:
    rng = np.random.default_rng(seed)

    people = []
    for i in range(n_people):
        distances = {
            f"C{j + 1}": float(round(rng.uniform(1.0, 100.0), 2)) for j in range(n_centers)
        }
        people.append(
            Person(
                person_id=f"P{i + 1}",
                vulnerability=float(round(rng.uniform(0.0, 10.0), 2)),
                mobility=float(round(rng.uniform(0.0, 10.0), 2)),
                group_size=int(rng.integers(1, 4)),  # 1-3
                distances=distances,
            )
        )

    demand = sum(p.group_size for p in people)
    # Total capacity ~1.15x demand, distributed across centres with mild variation.
    base_cap = math.ceil(1.15 * demand / n_centers)
    centers = [
        Center(
            center_id=f"C{j + 1}",
            capacity=base_cap + int(rng.integers(0, max(1, base_cap // 4 + 1))),
            service_level=float(round(rng.uniform(0.0, 10.0), 2)),
            road_accessibility=float(round(rng.uniform(0.0, 10.0), 2)),
        )
        for j in range(n_centers)
    ]
    return HumanitarianProblem(people=people, centers=centers)


def _generate_ed(n_eds: int, n_volunteers: int, seed: int) -> ProblemInstance:
    rng = np.random.default_rng(seed)

    vacancies = []
    for j in range(n_eds):
        vac_type = SkillType.TRIAGE if j % 2 == 0 else SkillType.ER_NURSE
        vacancies.append(
            Vacancy(
                ed_id=f"ED{j + 1}",
                vacancy_type=vac_type,
                num_patients=int(rng.integers(0, 101)),
                emergency_level=float(round(rng.uniform(0.0, 10.0), 2)),
            )
        )

    volunteers = []
    for i in range(n_volunteers):
        # Alternate skill types so each type comfortably covers its vacancies.
        skill = SkillType.TRIAGE if i % 2 == 0 else SkillType.ER_NURSE
        distances = {f"ED{j + 1}": float(round(rng.uniform(0.0, 100.0), 2)) for j in range(n_eds)}
        volunteers.append(
            Volunteer(
                volunteer_id=f"V{i + 1}",
                skill_type=skill,
                skill_level=float(round(rng.uniform(0.0, 10.0), 2)),
                distances=distances,
                difficulty_tolerance=float(round(rng.uniform(0.0, 10.0), 2)),
            )
        )
    return ProblemInstance(volunteers=volunteers, vacancies=vacancies)


_GENERATORS: dict[str, Callable[[int, int, int], object]] = {
    "humanitarian": _generate_humanitarian,
    "ed-staffing": _generate_ed,
}
_SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "humanitarian": _HUMANITARIAN_SIZES,
    "ed-staffing": _ED_SIZES,
}


def generate_instance(model: str, size: str, seed: int) -> object:
    """Deterministically build one problem instance for *model* / *size*."""
    if model not in _GENERATORS:
        raise ValueError(f"unknown model {model!r}; available: {sorted(_GENERATORS)!r}")
    if size not in _SIZES[model]:
        raise ValueError(f"unknown size {size!r}; available: {sorted(_SIZES[model])!r}")
    a, b = _SIZES[model][size]
    return _GENERATORS[model](a, b, seed)


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


def run_benchmark(
    model: str,
    sizes: list[str],
    n_instances: int,
    config: RunConfig,
    *,
    base_seed: int = 42,
    check_repro: bool = False,
    include_baseline: bool = False,
    include_exact: bool = False,
) -> list[BenchmarkRow]:
    """Run the benchmark and return one aggregated row per (size, solver).

    Each instance is solved with ``config`` (its seed drives the solver). When
    ``check_repro`` is set, every instance is solved a second time and the
    fraction of bit-for-bit identical results is reported as REP. When
    ``include_baseline`` / ``include_exact`` are set, the deterministic greedy
    and/or exact weighted-sum comparators are additionally run on every instance
    and reported as ``greedy`` / ``exact`` solver rows, enabling a
    framework-vs-baseline comparison (and the Wilcoxon HV tests).
    """
    domain = get_domain(model)
    extra_configs = []
    if include_baseline:
        extra_configs.append(replace(config, solver="greedy"))
    if include_exact:
        extra_configs.append(replace(config, solver="exact"))
    rows: list[BenchmarkRow] = []

    for size in sizes:
        # solver value -> metric name -> list across instances
        per_solver: dict[str, dict[str, list[float]]] = {}
        repro_flags: list[float] = []

        for i in range(n_instances):
            problem = generate_instance(model, size, base_seed + _SIZE_OFFSET[size] + i)
            fronts = run(problem, config, domain)
            for extra_cfg in extra_configs:
                fronts = [*fronts, *run(problem, extra_cfg, domain)]

            for front in fronts:
                m = compute_metrics(front)
                bucket = per_solver.setdefault(
                    front.solver.value, {k: [] for k in ("nns", "mid", "sm", "hv", "cpu")}
                )
                bucket["nns"].append(m.nns)
                bucket["mid"].append(m.mid)
                bucket["sm"].append(m.sm)
                bucket["hv"].append(m.hv)
                bucket["cpu"].append(m.cpu_time_sec)

            if check_repro:
                fronts2 = run(problem, config, domain)
                identical = fronts_signature(fronts) == fronts_signature(fronts2)
                repro_flags.append(1.0 if identical else 0.0)

        rep = float(np.mean(repro_flags)) if check_repro and repro_flags else None
        for solver_value, metrics in per_solver.items():
            rows.append(
                BenchmarkRow(
                    model=model,
                    size=size,
                    solver=solver_value,
                    n_instances=n_instances,
                    nns_mean=float(np.mean(metrics["nns"])),
                    nns_std=float(np.std(metrics["nns"])),
                    mid_mean=float(np.mean(metrics["mid"])),
                    mid_std=float(np.std(metrics["mid"])),
                    sm_mean=float(np.mean(metrics["sm"])),
                    sm_std=float(np.std(metrics["sm"])),
                    hv_mean=float(np.mean(metrics["hv"])),
                    hv_std=float(np.std(metrics["hv"])),
                    cpu_mean=float(np.mean(metrics["cpu"])),
                    cpu_std=float(np.std(metrics["cpu"])),
                    rep=rep,
                    hv_samples=list(metrics["hv"]),
                )
            )
    return rows


def resolve_sizes(size: str) -> list[str]:
    """Map the CLI --size value to a concrete list of size classes."""
    if size == "both":
        return ["small", "large"]
    if size in ("small", "large"):
        return [size]
    raise ValueError(f"size must be one of 'small', 'large', 'both'; got {size!r}")


def write_benchmark_summary(rows: list[BenchmarkRow], output_dir: Path) -> tuple[Path, Path]:
    """Write the aggregated benchmark summary to CSV + JSON. Returns both paths."""
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    # Drop the in-memory per-instance sample arrays from the Table-3 summary.
    records = [{k: v for k, v in asdict(r).items() if not k.endswith("_samples")} for r in rows]
    csv_path = output_dir / f"benchmark_{ts}.csv"
    json_path = output_dir / f"benchmark_{ts}.json"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(records, indent=2))
    return csv_path, json_path
