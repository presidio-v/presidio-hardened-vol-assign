"""Data models for the allocation module.

All models are dataclasses — type-safe containers with no business logic.
Validation lives in `allocation.validation`; algorithm logic in
`allocation.fis` and `allocation.solvers`.

Notation matches Rabiei, Arias-Aranda, Stantchev (ATRes 2026, in press)
Table 1 throughout. Score mappings for ordinal status fields match the
paper exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Enumerations — status fields with discrete score mappings (ATRes Table 1)
# ---------------------------------------------------------------------------


class DisabilityStatus(str, Enum):
    """ATRes DS_j: disability status of an affected person."""

    NONE = "none"
    MINOR = "minor"
    SEVERE = "severe"

    @property
    def score(self) -> float:
        return {"none": 0.0, "minor": 0.5, "severe": 1.0}[self.value]


class InjuryLevel(str, Enum):
    """ATRes IL_j: injury severity classification."""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SERIOUS = "serious"
    LIFE_THREATENING = "life_threatening"

    @property
    def score(self) -> float:
        return {
            "none": 0.0,
            "minor": 0.25,
            "moderate": 0.5,
            "serious": 0.75,
            "life_threatening": 1.0,
        }[self.value]


class LivingStatus(str, Enum):
    """ATRes LS_j: living-arrangement classification."""

    WITH_SUPPORT = "with_support"
    ALONE = "alone"

    @property
    def score(self) -> float:
        return {"with_support": 0.0, "alone": 1.0}[self.value]


class RoadCondition(str, Enum):
    """ATRes RCS_{j,i}: road condition score for a (person, center) route."""

    CLEAR = "clear"
    PARTIALLY_BLOCKED = "partially_blocked"
    BLOCKED = "blocked"

    @property
    def score(self) -> float:
        return {"clear": 0.1, "partially_blocked": 0.5, "blocked": 1.0}[self.value]


class HazardLevel(str, Enum):
    """ATRes PHS_{j,i}: possible-hazard score for a (person, center) route."""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    EXTREME = "extreme"

    @property
    def score(self) -> float:
        return {
            "none": 0.05,
            "minor": 0.25,
            "moderate": 0.5,
            "significant": 0.75,
            "extreme": 1.0,
        }[self.value]


class AllocationSolverType(str, Enum):
    """Multi-objective evolutionary algorithms supported by `pva allocate`."""

    NSGA2 = "nsga2"
    NRGA = "nrga"
    NSGA3 = "nsga3"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


@dataclass
class Person:
    """One affected person eligible for allocation to a relief center.

    Notation follows ATRes Table 1. The vulnerability score VS_j is derived
    from (age, disability_status, injury_level, living_status) per Eq. (2)
    using the configured weights in `AllocationConfig.weights`; it is not
    stored here.

    Attributes:
        person_id: Unique identifier.
        age: Person's age in years (used to derive Age Score AS per Eq.
            in §3.2.1: AS=(25-age)/25 if age<25; 0 if 25≤age≤60;
            (age-60)/30 if 60<age≤90; 1 if age>90).
        disability_status: Discrete DS_j status.
        injury_level: Discrete IL_j status.
        living_status: Discrete LS_j status.
        infrastructure_damage_level: ATRes IDL_j ∈ [0, 100] — damage at the
            person's location.
        resource_time_remaining: ATRes RTR_j in hours — estimated time
            before the person's essential resources run out.
    """

    person_id: str
    age: float
    disability_status: DisabilityStatus
    injury_level: InjuryLevel
    living_status: LivingStatus
    infrastructure_damage_level: float
    resource_time_remaining: float

    @property
    def age_score(self) -> float:
        """ATRes AS_j — piecewise age score in [0, 1]."""
        a = float(self.age)
        if a < 25:
            return (25.0 - a) / 25.0
        if a <= 60:
            return 0.0
        if a <= 90:
            return (a - 60.0) / 30.0
        return 1.0


@dataclass
class ReliefCenter:
    """One relief center available for allocation.

    Attributes:
        center_id: Unique identifier.
        center_occupancy_rate: ATRes COR_i ∈ [0, 100] — percentage of center
            capacity already in use.
        resource_depletion_rate: ATRes RDR_i ∈ [0, 100] — percentage of
            resources consumed per hour at the center.
    """

    center_id: str
    center_occupancy_rate: float
    resource_depletion_rate: float


@dataclass
class TravelInfo:
    """Per (person, center) routing information.

    The roadworthiness score RWS_{j,i} is derived from (road_condition,
    possible_hazard) per Eq. (5) using weights in `AllocationConfig`; not
    stored here.

    Attributes:
        person_id: Origin person.
        center_id: Destination center.
        travel_duration: ATRes TD_{j,i} in minutes.
        road_condition: Discrete RCS_{j,i}.
        possible_hazard: Discrete PHS_{j,i}.
    """

    person_id: str
    center_id: str
    travel_duration: float
    road_condition: RoadCondition
    possible_hazard: HazardLevel


@dataclass
class AllocationProblem:
    """A complete allocation problem instance.

    Invariants (enforced in validation):
        - n_dir < len(people)   (Eq. 15: cannot serve everyone)
        - n_dir > 0
        - len(centers) > 0
        - travel has exactly one entry per (person, center) pair
    """

    people: list[Person]
    centers: list[ReliefCenter]
    travel: dict[tuple[str, str], TravelInfo]
    n_dir: int

    @property
    def n_people(self) -> int:
        return len(self.people)

    @property
    def n_centers(self) -> int:
        return len(self.centers)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Weights:
    """Parametric weights for derived scores.

    Defaults match ATRes baseline (equal weighting within each group).

    Attributes:
        was, wds, wil, wls: Weights for vulnerability score VS (Eq. 2) —
            applied to age_score, disability, injury, living-status.
        wrc, wph: Weights for roadworthiness score RWS (Eq. 5) —
            applied to road_condition, possible_hazard.
    """

    was: float = 1.0
    wds: float = 1.0
    wil: float = 1.0
    wls: float = 1.0
    wrc: float = 1.0
    wph: float = 1.0


@dataclass
class AllocationConfig:
    """Solver hyper-parameters and run settings for `pva allocate`.

    Attributes:
        solver: Which solver to run (or use a list at the CLI layer for "all").
        objectives: 3 (ATRes original — fused TIL) or 4 (MDPI extension —
            split TRD + RPD). Default 4.
        weights: Parametric weights for VS and RWS computation.
        pop_size: GA population size.
        generations: Number of generations.
        seed: Optional random seed (None = non-deterministic).
        nsga3_divisions: Das-Dennis reference-point divisions for NSGA-III
            (default p=4, yielding 35 reference points in 4D simplex).
        output_dir: Directory where result files are written.
    """

    solver: AllocationSolverType
    objectives: int = 4
    weights: Weights = field(default_factory=Weights)
    pop_size: int = 100
    generations: int = 200
    seed: int | None = None
    nsga3_divisions: int = 4
    output_dir: str = "./results"


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


@dataclass
class Allocation:
    """One person → center allocation within a solution.

    The objective contributions stored here are *per-allocation* values
    used to reconstruct the solution-level mean objectives.

    For 4-objective runs: ulpp, trd, rpd, cail_contrib.
    For 3-objective runs: ulpp, til, cail_contrib.  (trd and rpd are 0.)
    """

    person_id: str
    center_id: str
    ulpp: float = 0.0
    til: float = 0.0  # used only when objectives == 3
    trd: float = 0.0  # used only when objectives == 4
    rpd: float = 0.0  # used only when objectives == 4
    cail_contrib: float = 0.0


@dataclass
class AllocationSolution:
    """One Pareto-optimal allocation: complete mapping + mean objective values.

    Attributes:
        allocations: One Allocation per directed person (length = n_dir).
        objectives_count: 3 or 4 — drives which mean fields are populated.
        mn_ulpp: Mean Unfairness in People Prioritization (Eq. 3, minimise).
        mn_til: Mean Transport Infeasibility Level (Eq. 7) — populated only
            when objectives_count == 3.
        mn_trd: Mean Transport Robustness Deficit — populated only when
            objectives_count == 4.
        mn_rpd: Mean Rapidity Deficit — populated only when
            objectives_count == 4.
        mn_cail: Mean Center Allocation Imbalance Level (Eq. 10, minimise).
    """

    allocations: list[Allocation]
    objectives_count: int
    mn_ulpp: float
    mn_cail: float
    mn_til: float = 0.0
    mn_trd: float = 0.0
    mn_rpd: float = 0.0

    @property
    def n_allocations(self) -> int:
        return len(self.allocations)

    @property
    def fitness(self) -> tuple[float, ...]:
        """Return the objective tuple in the order used by the solver."""
        if self.objectives_count == 4:
            return (self.mn_ulpp, self.mn_trd, self.mn_rpd, self.mn_cail)
        return (self.mn_ulpp, self.mn_til, self.mn_cail)


@dataclass
class AllocationParetoFront:
    """Non-dominated solutions produced by one solver run.

    Attributes:
        solver: The solver that produced this front.
        objectives_count: 3 or 4 — must match every solution's
            `objectives_count`.
        solutions: Non-dominated AllocationSolution objects.
        cpu_time_sec: Wall-clock solver time.
    """

    solver: AllocationSolverType
    objectives_count: int
    solutions: list[AllocationSolution] = field(default_factory=list)
    cpu_time_sec: float = 0.0

    @property
    def nns(self) -> int:
        """Number of Non-dominated Solutions."""
        return len(self.solutions)


@dataclass
class AllocationMetrics:
    """Quality metrics for an allocation Pareto front.

    Attributes:
        solver: The solver that produced the front.
        objectives_count: 3 or 4 — drives HV reference-point dimensionality.
        nns: Number of Non-dominated Solutions.
        mid: Mean Ideal Distance — Euclidean distance from each solution to
            the ideal point at the origin in objective space.
        sm: Spacing Metric — variability of distances between consecutive
            front points.
        hv: Hypervolume — objective-space volume dominated by the front
            (computed via pymoo for ≥3D; sweep-line for 2D legacy paths).
        cpu_time_sec: Wall-clock solver time.
    """

    solver: AllocationSolverType
    objectives_count: int
    nns: int
    mid: float
    sm: float
    hv: float
    cpu_time_sec: float
