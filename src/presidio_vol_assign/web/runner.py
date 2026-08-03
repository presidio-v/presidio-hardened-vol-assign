"""Bounded, process-isolated execution of demo solver runs.

The solver is CPU-bound Python, so it runs in a worker process rather than on
the event loop or in a thread. Workers are pre-warmed (the heavy scikit-fuzzy /
DEAP imports dominate a cold run) and every run is bounded twice: by the caps in
:data:`LIMITS`, which keep the work small enough to finish in seconds, and by a
wall-clock timeout that kills the worker if a run somehow escapes those caps.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("presidio_vol_assign.web")


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Limits:
    """Hard server-side bounds on a demo run.

    These are enforced after the per-knob clamping in
    :meth:`~presidio_vol_assign.web.scenarios.Scenario.resolve_knobs`, so a
    hand-crafted request cannot ask for more work than the sliders allow.
    """

    max_units: int = 300
    max_sites: int = 20
    max_pop_size: int = 200
    max_generations: int = 200
    max_solutions_returned: int = 120
    timeout_sec: float = 45.0
    max_workers: int = 2


LIMITS = Limits()

ALLOWED_SOLVERS = ("nsga2", "nrga", "both")
"""Solvers the demo exposes. ``all`` and the baselines stay CLI-only."""


class RunRejected(ValueError):
    """The request is outside the demo's caps or otherwise unusable."""


class RunTimeout(RuntimeError):
    """The run exceeded :attr:`Limits.timeout_sec` and its worker was killed."""


@dataclass
class RunRequest:
    """A validated demo run.

    Attributes:
        scenario_id: Which preset to run.
        knobs: Already-clamped slider values.
        solver: One of :data:`ALLOWED_SOLVERS`.
        seed: Reproducibility seed; the demo always sets one.
        pop_size: GA population size.
        generations: GA generation count.
        want_evidence: Whether to also emit a signed evidence record. Ignored
            when the server has no signing key configured.
    """

    scenario_id: str
    knobs: dict[str, float] = field(default_factory=dict)
    solver: str = "nsga2"
    seed: int = 42
    pop_size: int = 100
    generations: int = 120
    want_evidence: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "knobs": dict(self.knobs),
            "solver": self.solver,
            "seed": self.seed,
            "pop_size": self.pop_size,
            "generations": self.generations,
            "want_evidence": self.want_evidence,
        }


def build_request(payload: dict[str, Any]) -> RunRequest:
    """Validate a raw request payload into a capped :class:`RunRequest`.

    Raises:
        RunRejected: On an unknown scenario or a value outside the caps.
    """
    from presidio_vol_assign.web.scenarios import SCENARIOS_BY_ID

    scenario_id = str(payload.get("scenario", ""))
    scenario = SCENARIOS_BY_ID.get(scenario_id)
    if scenario is None:
        raise RunRejected(f"unknown scenario {scenario_id!r}")

    solver = str(payload.get("solver", "nsga2"))
    if solver not in ALLOWED_SOLVERS:
        raise RunRejected(f"solver must be one of {list(ALLOWED_SOLVERS)}, got {solver!r}")

    knobs = scenario.resolve_knobs(payload.get("knobs"))
    _enforce_size_caps(scenario, knobs)

    return RunRequest(
        scenario_id=scenario_id,
        knobs=knobs,
        solver=solver,
        seed=_bounded_int(payload.get("seed"), default=42, low=0, high=2**31 - 1, name="seed"),
        pop_size=_bounded_int(
            payload.get("pop_size"), default=100, low=10, high=LIMITS.max_pop_size, name="pop_size"
        ),
        generations=_bounded_int(
            payload.get("generations"),
            default=120,
            low=5,
            high=LIMITS.max_generations,
            name="generations",
        ),
        want_evidence=bool(payload.get("evidence", False)),
    )


def _bounded_int(value: Any, *, default: int, low: int, high: int, name: str) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RunRejected(f"{name} must be an integer") from exc
    if not low <= parsed <= high:
        raise RunRejected(f"{name} must be between {low} and {high}, got {parsed}")
    return parsed


def _enforce_size_caps(scenario: Any, knobs: dict[str, float]) -> None:
    """Reject instances larger than the demo allows, whatever the knob bounds say."""
    units = int(knobs.get("n_people", knobs.get("n_volunteers", 0)))
    sites = int(knobs.get("n_centers", knobs.get("n_eds", 0)))
    if units > LIMITS.max_units:
        raise RunRejected(f"at most {LIMITS.max_units} units per run (asked for {units})")
    if sites > LIMITS.max_sites:
        raise RunRejected(f"at most {LIMITS.max_sites} sites per run (asked for {sites})")


# ---------------------------------------------------------------------------
# Worker body — must be importable at module level so it can be pickled
# ---------------------------------------------------------------------------


def _warm_worker() -> None:
    """Pre-import the heavy solver stack so the first real run is not cold."""
    import presidio_vol_assign.domains.ed_staffing  # noqa: F401
    import presidio_vol_assign.domains.humanitarian  # noqa: F401
    import presidio_vol_assign.engine  # noqa: F401


def _build_domain(scenario: Any, knobs: dict[str, float]) -> Any:
    from presidio_vol_assign.domains.ed_staffing import EDStaffingDomain
    from presidio_vol_assign.domains.humanitarian import HumanitarianDomain

    if scenario.model == "ed-staffing":
        return EDStaffingDomain()
    if scenario.hard_capacity:
        return HumanitarianDomain(
            hard_capacity=True,
            max_distance=float(knobs.get("max_distance", 30.0)),
        )
    return HumanitarianDomain()


_INSTANCE_CACHE: dict[str, tuple[Any, Any, Any]] = {}
_INSTANCE_CACHE_SIZE = 4
"""Per-worker memo of (instance, domain, fis_cache).

Generating the instance is cheap; pre-computing the fuzzy scores for every
(unit, site) pair is not, and it dominates a demo run. Because an instance is a
pure function of (scenario, knobs, seed), changing only the algorithm or the
generation count can reuse the whole precompute.
"""


def _instance_key(request: RunRequest) -> str:
    knobs = ",".join(f"{k}={request.knobs[k]:.6g}" for k in sorted(request.knobs))
    return f"{request.scenario_id}|{request.seed}|{knobs}"


def _prepare(request: RunRequest) -> tuple[Any, Any, Any]:
    """Return (instance, domain, fis_cache), reusing a memoised precompute."""
    from presidio_vol_assign.web.scenarios import generate_instance, get_scenario

    key = _instance_key(request)
    cached = _INSTANCE_CACHE.get(key)
    if cached is not None:
        return cached

    scenario = get_scenario(request.scenario_id)
    instance = generate_instance(scenario, request.knobs, request.seed)
    domain = _build_domain(scenario, request.knobs)
    # precompute() also primes solver-side state on the domain, so the two are
    # cached together and never recombined across different knob settings.
    fis_cache = domain.precompute(instance.problem)

    if len(_INSTANCE_CACHE) >= _INSTANCE_CACHE_SIZE:
        _INSTANCE_CACHE.pop(next(iter(_INSTANCE_CACHE)))
    _INSTANCE_CACHE[key] = (instance, domain, fis_cache)
    return instance, domain, fis_cache


def _solve(request_dict: dict[str, Any]) -> dict[str, Any]:
    """Run one demo request end to end. Executes inside a worker process."""
    from presidio_vol_assign.engine import run as run_solvers
    from presidio_vol_assign.metrics import compute_metrics
    from presidio_vol_assign.models import RunConfig
    from presidio_vol_assign.web.scenarios import get_scenario

    request = RunRequest(**request_dict)
    scenario = get_scenario(request.scenario_id)
    instance, domain, fis_cache = _prepare(request)

    config = RunConfig(
        solver=request.solver,
        pop_size=request.pop_size,
        generations=request.generations,
        seed=request.seed,
        output_dir=".",  # unused: the demo never writes result files
    )
    fronts = run_solvers(instance.problem, config, domain, cache=fis_cache)

    site_ids = [s["id"] for s in instance.site_points]
    site_index = {sid: i for i, sid in enumerate(site_ids)}
    unit_ids = [u["id"] for u in instance.unit_points]
    unit_index = {uid: i for i, uid in enumerate(unit_ids)}

    results = []
    for front in fronts:
        metrics = compute_metrics(front)
        solutions = _encode_solutions(front, scenario, unit_index, site_index)
        results.append(
            {
                "solver": front.solver.value,
                "metrics": {
                    "nns": metrics.nns,
                    "hv": round(metrics.hv, 6),
                    "sm": round(metrics.sm, 6),
                    "mid": round(metrics.mid, 6),
                    "cpuTimeSec": round(metrics.cpu_time_sec, 3),
                },
                "solutions": solutions,
            }
        )

    payload: dict[str, Any] = {
        "scenario": scenario.id,
        "model": scenario.model,
        "hardCapacity": scenario.hard_capacity,
        "seed": request.seed,
        "solver": request.solver,
        "popSize": request.pop_size,
        "generations": request.generations,
        "knobs": request.knobs,
        "objectives": [o.as_dict() for o in scenario.objectives],
        "units": instance.unit_points,
        "sites": instance.site_points,
        "summary": instance.summary,
        "results": results,
        "cliHint": scenario.cli_hint,
    }

    if request.want_evidence:
        payload["evidence"] = _maybe_emit_evidence(request, scenario, instance, fronts, domain)

    return payload


def _encode_solutions(
    front: Any,
    scenario: Any,
    unit_index: dict[str, int],
    site_index: dict[str, int],
) -> list[dict[str, Any]]:
    """Encode a front compactly: objectives plus one site index per unit.

    Sending an allocation vector rather than a list of assignment objects keeps
    the payload small enough that the browser can hold the whole front and
    re-render instantly as the trade-off slider moves.
    """
    id_field = "person_id" if scenario.model == "humanitarian" else "volunteer_id"
    site_field = "center_id" if scenario.model == "humanitarian" else "ed_id"

    solutions = front.solutions
    if len(solutions) > LIMITS.max_solutions_returned:
        # Subsample evenly along the front so its shape is preserved.
        stride = len(solutions) / LIMITS.max_solutions_returned
        solutions = [solutions[int(i * stride)] for i in range(LIMITS.max_solutions_returned)]

    encoded = []
    for sol in solutions:
        alloc = [-1] * len(unit_index)
        for assignment in sol.assignments:
            ui = unit_index.get(getattr(assignment, id_field))
            si = site_index.get(getattr(assignment, site_field))
            if ui is not None and si is not None:
                alloc[ui] = si
        encoded.append(
            {
                "objectives": [round(float(v), 6) for v in sol.objectives],
                "alloc": alloc,
            }
        )
    return encoded


# ---------------------------------------------------------------------------
# Evidence (opt-in, requires a signing key in the server environment)
# ---------------------------------------------------------------------------


def evidence_available() -> bool:
    """Whether the server has signing key material configured."""
    return bool(os.environ.get("PVA_EVIDENCE_KEY") or os.environ.get("PVA_EVIDENCE_ED25519_KEY"))


def _write_instance_csvs(scenario: Any, instance: Any, out_dir: Path) -> list[Path]:
    """Materialise the generated instance as the CSVs the evidence record hashes."""
    problem = instance.problem
    if scenario.model == "humanitarian":
        center_ids = [c.center_id for c in problem.centers]
        centers_path = out_dir / "centers.csv"
        with centers_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["center_id", "capacity", "service_level", "road_accessibility"])
            for c in problem.centers:
                w.writerow([c.center_id, c.capacity, c.service_level, c.road_accessibility])
        people_path = out_dir / "people.csv"
        with people_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["person_id", "vulnerability", "mobility", "group_size"]
                + [f"distance_center_{cid}" for cid in center_ids]
            )
            for p in problem.people:
                w.writerow(
                    [p.person_id, p.vulnerability, p.mobility, p.group_size]
                    + [p.distances[cid] for cid in center_ids]
                )
        return [people_path, centers_path]

    ed_ids = sorted({v.ed_id for v in problem.vacancies})
    eds_path = out_dir / "eds.csv"
    with eds_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ed_id", "vacancy_type", "num_patients", "emergency_level"])
        for v in problem.vacancies:
            w.writerow([v.ed_id, v.vacancy_type.value, v.num_patients, v.emergency_level])
    vols_path = out_dir / "volunteers.csv"
    with vols_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["volunteer_id", "skill_type", "skill_level"]
            + [f"distance_ed_{eid}" for eid in ed_ids]
            + ["difficulty_tolerance"]
        )
        for vol in problem.volunteers:
            w.writerow(
                [vol.volunteer_id, vol.skill_type.value, vol.skill_level]
                + [vol.distances[eid] for eid in ed_ids]
                + [vol.difficulty_tolerance]
            )
    return [vols_path, eds_path]


def _maybe_emit_evidence(
    request: RunRequest,
    scenario: Any,
    instance: Any,
    fronts: list[Any],
    domain: Any,
) -> dict[str, Any]:
    """Emit a signed evidence record for the first front, or explain why not.

    Reuses the CLI's evidence path verbatim — including hashing the input CSVs —
    by materialising the generated instance into a temporary directory that is
    removed before returning. Nothing survives the request.
    """
    from presidio_vol_assign.evidence_cli import SigningKeyError, emit_evidence, resolve_signing_key
    from presidio_vol_assign.metrics import compute_metrics
    from presidio_vol_assign.writers import write_assignments_csv

    try:
        signer, alg, key = resolve_signing_key()
    except SigningKeyError as exc:
        return {"available": False, "reason": str(exc)}

    tmp_dir = Path(tempfile.mkdtemp(prefix="pva-web-evidence-"))
    try:
        input_paths = _write_instance_csvs(scenario, instance, tmp_dir)
        front = fronts[0]
        assignments_path = write_assignments_csv(front, tmp_dir, domain)
        record_path = emit_evidence(
            front=front,
            metrics=compute_metrics(front),
            model=scenario.model,
            solver=front.solver.value,
            seed=request.seed,
            pop_size=request.pop_size,
            generations=request.generations,
            input_csv_paths=input_paths,
            objective_labels=domain.objective_names,
            assignments_csv_path=assignments_path,
            output_dir=tmp_dir,
            signer=signer,
            alg=alg,
            key=key,
        )
        record = json.loads(record_path.read_text())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"available": True, "record": record}


# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------


class SolverPool:
    """A small pool of pre-warmed worker processes with a hard per-run timeout.

    ``ProcessPoolExecutor`` cannot cancel a task that has already started, so a
    timed-out run is handled by tearing the whole pool down (killing its worker
    processes) and building a fresh one. At the demo's caps a timeout means a
    bug rather than ordinary load, so the bluntness is acceptable and the next
    request simply pays a cold start.
    """

    def __init__(self, limits: Limits = LIMITS) -> None:
        self._limits = limits
        self._executor: ProcessPoolExecutor | None = None

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._limits.max_workers,
                initializer=_warm_worker,
            )
        return self._executor

    def start(self) -> None:
        """Create the pool up front so the first request is not a cold start."""
        self._ensure_executor()

    def shutdown(self) -> None:
        """Shut the pool down, waiting for in-flight runs to finish."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _kill(self) -> None:
        """Terminate every worker and drop the pool."""
        executor, self._executor = self._executor, None
        if executor is None:
            return
        # ProcessPoolExecutor exposes no public way to kill a running task, so
        # the worker processes are terminated directly before shutting down.
        for proc in list(getattr(executor, "_processes", {}).values()):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001 - best effort; the pool is discarded anyway
                log.warning("could not kill solver worker %s", getattr(proc, "pid", "?"))
        try:
            executor.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            log.warning("solver pool shutdown raised while recovering from a timeout")

    def run(self, request: RunRequest) -> dict[str, Any]:
        """Execute *request* in a worker.

        Raises:
            RunTimeout: If the run exceeds the configured wall-clock timeout.
        """
        executor = self._ensure_executor()
        future = executor.submit(_solve, request.as_dict())
        try:
            return future.result(timeout=self._limits.timeout_sec)
        except FutureTimeoutError as exc:
            self._kill()
            raise RunTimeout(
                f"run exceeded {self._limits.timeout_sec:.0f}s and was cancelled"
            ) from exc
