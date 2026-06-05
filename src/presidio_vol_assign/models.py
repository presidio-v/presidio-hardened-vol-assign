"""Core data models for presidio-hardened-vol-assign.

All models are plain dataclasses — type-safe containers with no business logic.
Validation lives in validation.py; algorithm logic in fis.py and solvers.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SkillType(str, Enum):
    """Volunteer/vacancy role type."""

    TRIAGE = "triage"
    ER_NURSE = "er_nurse"


class SolverType(str, Enum):
    """Available metaheuristic solvers."""

    NSGA2 = "nsga2"
    NRGA = "nrga"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


@dataclass
class Volunteer:
    """One spontaneous volunteer available for assignment.

    Attributes:
        volunteer_id: Unique string identifier.
        skill_type: Role the volunteer can fill (triage or ER nurse).
        skill_level: Proficiency score in [0, 10].
        distances: Mapping from ED id to distance in km [0, 100].
        difficulty_tolerance: Volunteer's self-reported tolerance for difficult
            situations, in [0, 10]. Used by FIS3 as a preference input.
    """

    volunteer_id: str
    skill_type: SkillType
    skill_level: float
    distances: dict[str, float]
    difficulty_tolerance: float

    def distance_to(self, ed_id: str) -> float:
        """Return distance in km to a specific ED."""
        return self.distances[ed_id]


@dataclass
class Vacancy:
    """One open role at an Emergency Department.

    Attributes:
        ed_id: Unique ED identifier.
        vacancy_type: Role to be filled (triage or ER nurse).
        num_patients: Current patient count at the ED [0, 100].
        emergency_level: Urgency score for this ED [0, 10].
    """

    ed_id: str
    vacancy_type: SkillType
    num_patients: int
    emergency_level: float


@dataclass
class ProblemInstance:
    """A complete volunteer assignment problem instance.

    Invariant (checked in validation.py):
        len(volunteers) >= len(vacancies)
        All volunteer skill_type values can cover at least their matched vacancy type.
    """

    volunteers: list[Volunteer]
    vacancies: list[Vacancy]

    @property
    def n_volunteers(self) -> int:
        return len(self.volunteers)

    @property
    def n_vacancies(self) -> int:
        return len(self.vacancies)


@dataclass
class RunConfig:
    """Solver hyper-parameters and run settings.

    Attributes:
        solver: Which solver(s) to run.
        pop_size: GA population size.
        generations: Number of generations to evolve.
        seed: Optional random seed for reproducibility (None = non-deterministic).
        output_dir: Directory where result files are written.
    """

    solver: SolverType | str  # "both" is accepted at CLI level, expanded before passing here
    pop_size: int = 100
    generations: int = 200
    seed: int | None = None
    output_dir: str = "./results"


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


@dataclass
class Assignment:
    """A single volunteer-to-vacancy assignment within a solution.

    Attributes:
        volunteer_id: The assigned volunteer.
        ed_id: The target Emergency Department.
        vacancy_type: The role being filled.
        fis1_score: FIS1 output for this pairing (importance of unmet triage need).
        fis2_score: FIS2 output for this pairing (importance of unmet ER-nurse need).
        fis3_score: FIS3 output for this pairing (preference dissatisfaction).
    """

    volunteer_id: str
    ed_id: str
    vacancy_type: SkillType
    fis1_score: float = 0.0
    fis2_score: float = 0.0
    fis3_score: float = 0.0


@dataclass
class Solution:
    """One Pareto-optimal solution: a complete assignment mapping + objective values.

    The canonical objective representation is the ``objectives`` tuple, which has
    one entry per objective and so supports both the 2-objective ED-staffing
    model and the 3-objective humanitarian model. ``z1`` / ``z2`` are retained as
    backward-compatible views on the first two objectives.

    Construct either way:
        Solution(assignments=..., z1=0.3, z2=0.4)            # 2-objective
        Solution(assignments=..., objectives=(0.3, 0.4, 0.5))  # N-objective

    Attributes:
        assignments: One Assignment per filled vacancy / allocated unit.
        z1: First objective value (minimise).
        z2: Second objective value (minimise).
        objectives: Full objective vector (length = number of objectives).
    """

    assignments: list[Assignment]
    z1: float = 0.0
    z2: float = 0.0
    objectives: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.objectives:
            # 2-objective construction path: derive the vector from z1/z2.
            self.objectives = (self.z1, self.z2)
        else:
            # N-objective path: expose the first two objectives as z1/z2.
            self.z1 = self.objectives[0]
            if len(self.objectives) > 1:
                self.z2 = self.objectives[1]

    @property
    def n_assignments(self) -> int:
        return len(self.assignments)

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)


@dataclass
class ParetoFront:
    """Collection of non-dominated solutions produced by one solver run.

    Attributes:
        solver: The solver that produced this front.
        solutions: Non-dominated Solution objects (sorted by z1 ascending).
        cpu_time_sec: Wall-clock time for the solver run.
    """

    solver: SolverType
    solutions: list[Solution] = field(default_factory=list)
    cpu_time_sec: float = 0.0

    @property
    def nns(self) -> int:
        """Number of Non-dominated Solutions."""
        return len(self.solutions)


@dataclass
class Metrics:
    """Quality metrics for a Pareto front, matching paper Table 3.

    Attributes:
        solver: The solver that produced the front.
        nns: Number of Non-dominated Solutions.
        mid: Mean Ideal Distance — mean Euclidean distance from each solution
             to the ideal point (0, 0) in objective space.
        sm: Spacing Metric — standard deviation of distances between consecutive
            solutions on the front (lower = more evenly spread).
        hv: Hypervolume — volume of objective space dominated by the front
            (higher = better coverage).
        cpu_time_sec: Wall-clock solver time.
        rep: Reproducibility score in [0, 1] — 1.0 when repeated seeded runs
            produce bit-for-bit identical fronts. None when not assessed.
    """

    solver: SolverType
    nns: int
    mid: float
    sm: float
    hv: float
    cpu_time_sec: float
    rep: float | None = None


@dataclass
class ReproReport:
    """Outcome of a bit-for-bit reproducibility check across repeated runs.

    Attributes:
        n_runs: How many times the same seeded configuration was executed.
        rep: 1.0 if every run produced an identical front signature, else 0.0.
        signature: SHA-256 signature of the first run's combined fronts.
        identical: Convenience flag, ``rep == 1.0``.
    """

    n_runs: int
    rep: float
    signature: str
    identical: bool


# ---------------------------------------------------------------------------
# Humanitarian allocation model (v0.2.0) — affected people -> relief centres
# ---------------------------------------------------------------------------


@dataclass
class Person:
    """One affected person (or household) awaiting allocation to a relief centre.

    Attributes:
        person_id: Unique string identifier (no PII).
        vulnerability: Priority/need score in [0, 10] (FIS-A input).
        mobility: Personal transport-access score in [0, 10] (FIS-B input;
            0 = immobile, 10 = fully mobile).
        group_size: People moved together as a unit, int [1, 20]; the load this
            person contributes to a centre (FIS-C utilisation).
        distances: Mapping from centre id to distance in km [0, 100].
    """

    person_id: str
    vulnerability: float
    mobility: float
    group_size: int
    distances: dict[str, float]

    def distance_to(self, center_id: str) -> float:
        """Return distance in km to a specific relief centre."""
        return self.distances[center_id]


@dataclass
class Center:
    """One relief centre that affected people can be allocated to.

    Attributes:
        center_id: Unique centre identifier.
        capacity: Nominal capacity in people, int [1, 5000] (FIS-C input).
        service_level: Resource/quality level in [0, 10] (FIS-A input).
        road_accessibility: Route condition / access score in [0, 10] (FIS-B input).
    """

    center_id: str
    capacity: int
    service_level: float
    road_accessibility: float


@dataclass
class HumanitarianProblem:
    """A complete humanitarian-allocation problem instance.

    Invariant (checked in validation.py):
        sum(centre capacities) >= sum(person group sizes)  — enough room overall.
    """

    people: list[Person]
    centers: list[Center]

    @property
    def n_people(self) -> int:
        return len(self.people)

    @property
    def n_centers(self) -> int:
        return len(self.centers)


@dataclass
class CenterAssignment:
    """One person-to-centre allocation within a humanitarian solution.

    Attributes:
        person_id: The allocated person.
        center_id: The target relief centre.
        fairness: FIS-A output for this pairing (unfairness of prioritisation).
        transport: FIS-B output for this pairing (transportation infeasibility).
        overcrowding: FIS-C output for the assigned centre (balance/overcrowding).
    """

    person_id: str
    center_id: str
    fairness: float = 0.0
    transport: float = 0.0
    overcrowding: float = 0.0
