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

    Attributes:
        assignments: One Assignment per vacancy (complete coverage).
        z1: Mean importance of unmet needs across all vacancies (objective 1, minimise).
        z2: Mean degree of unsatisfied volunteer preferences (objective 2, minimise).
    """

    assignments: list[Assignment]
    z1: float
    z2: float

    @property
    def n_assignments(self) -> int:
        return len(self.assignments)


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
    """

    solver: SolverType
    nns: int
    mid: float
    sm: float
    hv: float
    cpu_time_sec: float
