"""Demo scenarios: preset problem shapes plus in-memory synthetic instances.

The CLI reads CSVs from disk; the demo server never touches the filesystem. It
builds :class:`ProblemInstance` / :class:`HumanitarianProblem` objects directly
from a seeded generator, so a public instance holds no user data and every run
is reproducible from ``(scenario, knobs, seed)`` alone.

People, centres, volunteers and EDs are placed on a square affected-area grid
and distances are Euclidean, matching ``examples/generate_examples.py``. The
coordinates are kept alongside the problem so the browser can draw the instance
on a map rather than only reporting objective numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from presidio_vol_assign.models import (
    Center,
    HumanitarianProblem,
    Person,
    ProblemInstance,
    SkillType,
    Vacancy,
    Volunteer,
)

AREA_KM = 70.0
"""Side length of the square affected area, in km (as in the worked example)."""


# ---------------------------------------------------------------------------
# Knob + scenario descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Knob:
    """One user-facing slider.

    Attributes:
        key: Identifier sent back in the run request.
        label: Plain-language slider label.
        minimum / maximum / step: Slider bounds, enforced again server-side.
        default: Initial value.
        help: One-line explanation aimed at a non-specialist.
        integer: Whether the value is rounded to an int before use.
    """

    key: str
    label: str
    minimum: float
    maximum: float
    step: float
    default: float
    help: str
    integer: bool = True

    def clamp(self, value: float | None) -> float:
        """Return *value* coerced into the declared range (default if None)."""
        raw = self.default if value is None else float(value)
        raw = min(max(raw, self.minimum), self.maximum)
        return float(round(raw)) if self.integer else raw

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "min": self.minimum,
            "max": self.maximum,
            "step": self.step,
            "default": self.default,
            "help": self.help,
        }


@dataclass(frozen=True)
class Objective:
    """A plain-language name for one solver objective.

    ``key`` is the paper's symbol (``z1``…); ``label`` is what a layperson sees.
    All objectives are minimised, so ``lower_is_better`` is always true — it is
    stated explicitly because the GUI says so on screen.
    """

    key: str
    label: str
    help: str
    lower_is_better: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "help": self.help,
            "lowerIsBetter": self.lower_is_better,
        }


@dataclass(frozen=True)
class Scenario:
    """One preset the GUI offers as a card.

    Attributes:
        id: Stable identifier used in API requests.
        title: Card heading.
        subtitle: One-line framing for a non-specialist.
        description: Short paragraph shown once the card is selected.
        model: Which solver model backs it (``ed-staffing`` / ``humanitarian``).
        hard_capacity: Humanitarian hard-constraint (repair) mode.
        unit_label / site_label: Plural nouns for the two sides of the problem.
        objectives: Plain-language objective descriptors, in solver order.
        knobs: Sliders shown for this scenario.
        cli_hint: The equivalent `pva` invocation, shown so the GUI stays an
            honest front-end for the CLI rather than a separate implementation.
    """

    id: str
    title: str
    subtitle: str
    description: str
    model: Literal["ed-staffing", "humanitarian"]
    hard_capacity: bool
    unit_label: str
    site_label: str
    objectives: tuple[Objective, ...]
    knobs: tuple[Knob, ...]
    cli_hint: str

    def knob(self, key: str) -> Knob:
        for k in self.knobs:
            if k.key == key:
                return k
        raise KeyError(key)

    def resolve_knobs(self, raw: dict[str, float] | None) -> dict[str, float]:
        """Clamp every declared knob against user input; ignore unknown keys."""
        supplied = raw or {}
        return {k.key: k.clamp(supplied.get(k.key)) for k in self.knobs}

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "subtitle": self.subtitle,
            "description": self.description,
            "model": self.model,
            "hardCapacity": self.hard_capacity,
            "unitLabel": self.unit_label,
            "siteLabel": self.site_label,
            "objectives": [o.as_dict() for o in self.objectives],
            "knobs": [k.as_dict() for k in self.knobs],
            "cliHint": self.cli_hint,
        }


# ---------------------------------------------------------------------------
# The three presets
# ---------------------------------------------------------------------------

_PEOPLE_KNOBS = (
    Knob(
        key="n_people",
        label="People needing shelter",
        minimum=10,
        maximum=300,
        step=10,
        default=150,
        help="How many affected people (or households) have to be placed.",
    ),
    Knob(
        key="n_centers",
        label="Relief centres open",
        minimum=2,
        maximum=12,
        step=1,
        default=5,
        help="How many centres are receiving people.",
    ),
    Knob(
        key="capacity_slack",
        label="Spare capacity",
        minimum=1.0,
        maximum=2.0,
        step=0.05,
        default=1.2,
        help="Total capacity as a multiple of demand. 1.0 means no slack at all.",
        integer=False,
    ),
    Knob(
        key="vulnerability",
        label="Average vulnerability",
        minimum=2.0,
        maximum=8.0,
        step=0.5,
        default=5.0,
        help="Higher means more of the population is high-priority.",
        integer=False,
    ),
)

_HUMANITARIAN_OBJECTIVES = (
    Objective(
        "z1",
        "Unfairness to the most vulnerable",
        "How badly the allocation serves the people who need help most.",
    ),
    Objective(
        "z2",
        "Travel burden",
        "How hard the journey to the assigned centre is, given distance and mobility.",
    ),
    Objective(
        "z3",
        "Centre overcrowding",
        "How far centres are pushed past comfortable occupancy.",
    ),
)

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="volunteers",
        title="Volunteers → Emergency Departments",
        subtitle="Who should staff which hospital after a disaster?",
        description=(
            "Spontaneous volunteers arrive after a disaster and have to be matched to "
            "open triage and ER-nurse roles across several Emergency Departments. The "
            "solver balances sending skilled people where the need is greatest against "
            "asking them to travel far or work beyond their stated tolerance."
        ),
        model="ed-staffing",
        hard_capacity=False,
        unit_label="volunteers",
        site_label="emergency departments",
        objectives=(
            Objective(
                "z1",
                "Unmet clinical need",
                "How much critical staffing need is left uncovered.",
            ),
            Objective(
                "z2",
                "Strain on volunteers",
                "How far volunteers travel and how far past their comfort they are pushed.",
            ),
        ),
        knobs=(
            Knob(
                key="n_volunteers",
                label="Volunteers available",
                minimum=6,
                maximum=200,
                step=2,
                default=60,
                help="How many people have come forward to help.",
            ),
            Knob(
                key="n_vacancies",
                label="Roles to fill",
                minimum=2,
                maximum=40,
                step=1,
                default=12,
                help="Open triage and ER-nurse posts across all departments.",
            ),
            Knob(
                key="n_eds",
                label="Emergency departments",
                minimum=1,
                maximum=8,
                step=1,
                default=3,
                help="How many hospitals are receiving volunteers.",
            ),
            Knob(
                key="emergency_level",
                label="Average urgency",
                minimum=2.0,
                maximum=9.0,
                step=0.5,
                default=6.0,
                help="How stretched the departments are on average.",
                integer=False,
            ),
        ),
        cli_hint="pva assign --model ed-staffing --volunteers volunteers.csv --eds eds.csv",
    ),
    Scenario(
        id="relief-centres",
        title="People in need → relief centres",
        subtitle="Where should each affected household be sent?",
        description=(
            "Affected people are allocated to relief centres. Capacity is treated as a "
            "soft target: any allocation is allowed, and crowding a centre past its "
            "capacity is penalised rather than forbidden. This is the model as published "
            "in the four-objective fuzzy framework paper."
        ),
        model="humanitarian",
        hard_capacity=False,
        unit_label="people",
        site_label="relief centres",
        objectives=_HUMANITARIAN_OBJECTIVES,
        knobs=_PEOPLE_KNOBS,
        cli_hint="pva allocate-people --people people.csv --centers centers.csv",
    ),
    Scenario(
        id="last-mile",
        title="Last mile under hard capacity limits",
        subtitle="Same problem — but no centre may overflow.",
        description=(
            "The same allocation, with capacity enforced as a hard constraint. A "
            "deterministic repair step guarantees no centre exceeds its capacity, and "
            "people with low mobility are not sent beyond a maximum distance. Compare "
            "the result with the previous scenario to see what those guarantees cost."
        ),
        model="humanitarian",
        hard_capacity=True,
        unit_label="people",
        site_label="relief centres",
        objectives=_HUMANITARIAN_OBJECTIVES,
        knobs=_PEOPLE_KNOBS
        + (
            Knob(
                key="max_distance",
                label="Max distance for low-mobility people (km)",
                minimum=10,
                maximum=70,
                step=5,
                default=30,
                help="People who cannot travel easily are never sent further than this.",
            ),
        ),
        cli_hint=(
            "pva allocate-people --people people.csv --centers centers.csv "
            "--hard-capacity --max-distance 30"
        ),
    ),
)

SCENARIOS_BY_ID = {s.id: s for s in SCENARIOS}


def get_scenario(scenario_id: str) -> Scenario:
    """Look up a scenario by id.

    Raises:
        KeyError: If *scenario_id* is not a known preset.
    """
    return SCENARIOS_BY_ID[scenario_id]


# ---------------------------------------------------------------------------
# Synthetic instance generation
# ---------------------------------------------------------------------------


@dataclass
class GeneratedInstance:
    """A synthetic problem plus the geometry needed to draw it.

    Attributes:
        problem: The solver-ready problem object.
        unit_points: One ``{id, x, y, label, weight}`` per allocatable unit.
        site_points: One ``{id, x, y, label, capacity}`` per destination site.
        summary: Short human-readable facts about the instance.
    """

    problem: Any
    unit_points: list[dict[str, Any]] = field(default_factory=list)
    site_points: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _euclidean(unit_xy: np.ndarray, site_xy: np.ndarray) -> np.ndarray:
    """Pairwise distances in km, clipped into the model's valid [1, 100] range."""
    deltas = unit_xy[:, None, :] - site_xy[None, :, :]
    return np.clip(np.sqrt((deltas**2).sum(axis=-1)), 1.0, 100.0)


def _generate_humanitarian(knobs: dict[str, float], seed: int) -> GeneratedInstance:
    """Build a people-to-centres instance on the affected-area grid."""
    rng = np.random.default_rng(seed)
    n_people = int(knobs["n_people"])
    n_centers = int(knobs["n_centers"])

    center_xy = rng.uniform(0, AREA_KM, size=(n_centers, 2))
    people_xy = rng.uniform(0, AREA_KM, size=(n_people, 2))
    center_ids = [f"C{j + 1}" for j in range(n_centers)]

    vulnerability = np.clip(rng.normal(knobs["vulnerability"], 2.5, n_people), 0, 10)
    mobility = np.clip(rng.normal(5.5, 2.5, n_people), 0, 10)
    group_size = rng.choice([1, 1, 1, 2, 2, 3, 4, 5], size=n_people)
    distance = _euclidean(people_xy, center_xy)

    demand = int(group_size.sum())
    # Capacity is spread over centres with a little jitter, then floor-corrected
    # so the total always clears demand — the model rejects infeasible instances.
    base_cap = math.ceil(knobs["capacity_slack"] * demand / n_centers)
    capacity = base_cap + rng.integers(0, base_cap // 4 + 1, size=n_centers)
    shortfall = demand - int(capacity.sum())
    if shortfall > 0:
        capacity[0] += shortfall

    service_level = np.clip(rng.normal(6.5, 2.0, n_centers), 0, 10)
    road = np.clip(rng.normal(6.0, 2.0, n_centers), 0, 10)

    centers = [
        Center(
            center_id=center_ids[j],
            capacity=int(capacity[j]),
            service_level=round(float(service_level[j]), 1),
            road_accessibility=round(float(road[j]), 1),
        )
        for j in range(n_centers)
    ]
    people = [
        Person(
            person_id=f"P{i + 1}",
            vulnerability=round(float(vulnerability[i]), 1),
            mobility=round(float(mobility[i]), 1),
            group_size=int(group_size[i]),
            distances={center_ids[j]: round(float(distance[i, j]), 1) for j in range(n_centers)},
        )
        for i in range(n_people)
    ]

    return GeneratedInstance(
        problem=HumanitarianProblem(people=people, centers=centers),
        unit_points=[
            {
                "id": people[i].person_id,
                "x": round(float(people_xy[i, 0]), 2),
                "y": round(float(people_xy[i, 1]), 2),
                "label": f"{people[i].person_id} · group of {people[i].group_size}",
                "weight": people[i].group_size,
                "priority": people[i].vulnerability,
            }
            for i in range(n_people)
        ],
        site_points=[
            {
                "id": centers[j].center_id,
                "x": round(float(center_xy[j, 0]), 2),
                "y": round(float(center_xy[j, 1]), 2),
                "label": f"{centers[j].center_id} · capacity {centers[j].capacity}",
                "capacity": centers[j].capacity,
            }
            for j in range(n_centers)
        ],
        summary={
            "units": n_people,
            "sites": n_centers,
            "demand": demand,
            "capacity": int(sum(c.capacity for c in centers)),
        },
    )


def _generate_ed_staffing(knobs: dict[str, float], seed: int) -> GeneratedInstance:
    """Build a volunteers-to-EDs instance on the affected-area grid.

    Vacancies are split between the two roles, then volunteers are generated with
    at least as many of each skill type as there are vacancies of that type, so
    the instance always satisfies the model's per-type feasibility constraint.
    """
    rng = np.random.default_rng(seed)
    n_volunteers = int(knobs["n_volunteers"])
    n_vacancies = int(knobs["n_vacancies"])
    n_eds = int(knobs["n_eds"])
    # The GUI clamps each knob independently, so the combination still has to be
    # reconciled here: the model needs at least one volunteer per vacancy.
    n_vacancies = min(n_vacancies, n_volunteers)

    ed_xy = rng.uniform(0, AREA_KM, size=(n_eds, 2))
    vol_xy = rng.uniform(0, AREA_KM, size=(n_volunteers, 2))
    ed_ids = [f"ED{j + 1}" for j in range(n_eds)]
    distance = _euclidean(vol_xy, ed_xy)

    n_triage_vac = max(1, n_vacancies // 2) if n_vacancies > 1 else n_vacancies
    n_nurse_vac = n_vacancies - n_triage_vac
    vacancy_types = [SkillType.TRIAGE] * n_triage_vac + [SkillType.ER_NURSE] * n_nurse_vac
    vacancy_eds = [ed_ids[k % n_eds] for k in range(n_vacancies)]

    num_patients = rng.integers(10, 90, size=n_vacancies)
    emergency = np.clip(rng.normal(knobs["emergency_level"], 1.5, n_vacancies), 0, 10)
    vacancies = [
        Vacancy(
            ed_id=vacancy_eds[k],
            vacancy_type=vacancy_types[k],
            num_patients=int(num_patients[k]),
            emergency_level=round(float(emergency[k]), 1),
        )
        for k in range(n_vacancies)
    ]

    # Guarantee per-type coverage first, then fill the remainder at random.
    # Values are carried as plain strings: numpy truncates enum members when it
    # coerces them into a fixed-width array.
    skill_values = [SkillType.TRIAGE.value] * n_triage_vac
    skill_values += [SkillType.ER_NURSE.value] * n_nurse_vac
    remaining = n_volunteers - len(skill_values)
    if remaining > 0:
        skill_values += [
            str(v)
            for v in rng.choice([SkillType.TRIAGE.value, SkillType.ER_NURSE.value], size=remaining)
        ]
    rng.shuffle(skill_values)
    skill_types = [SkillType(v) for v in skill_values]

    skill_level = np.clip(rng.normal(6.5, 2.0, n_volunteers), 0, 10)
    tolerance = np.clip(rng.normal(6.0, 2.0, n_volunteers), 0, 10)
    volunteers = [
        Volunteer(
            volunteer_id=f"V{i + 1}",
            skill_type=skill_types[i],
            skill_level=round(float(skill_level[i]), 1),
            distances={ed_ids[j]: round(float(distance[i, j]), 1) for j in range(n_eds)},
            difficulty_tolerance=round(float(tolerance[i]), 1),
        )
        for i in range(n_volunteers)
    ]

    ed_load = {eid: 0 for eid in ed_ids}
    for vac in vacancies:
        ed_load[vac.ed_id] += 1

    return GeneratedInstance(
        problem=ProblemInstance(volunteers=volunteers, vacancies=vacancies),
        unit_points=[
            {
                "id": volunteers[i].volunteer_id,
                "x": round(float(vol_xy[i, 0]), 2),
                "y": round(float(vol_xy[i, 1]), 2),
                "label": (
                    f"{volunteers[i].volunteer_id} · "
                    f"{volunteers[i].skill_type.value} · skill {volunteers[i].skill_level}"
                ),
                "weight": 1,
                "priority": volunteers[i].skill_level,
            }
            for i in range(n_volunteers)
        ],
        site_points=[
            {
                "id": ed_ids[j],
                "x": round(float(ed_xy[j, 0]), 2),
                "y": round(float(ed_xy[j, 1]), 2),
                "label": f"{ed_ids[j]} · {ed_load[ed_ids[j]]} open role(s)",
                "capacity": ed_load[ed_ids[j]],
            }
            for j in range(n_eds)
        ],
        summary={
            "units": n_volunteers,
            "sites": n_eds,
            "vacancies": n_vacancies,
            "triageVacancies": n_triage_vac,
            "nurseVacancies": n_nurse_vac,
        },
    )


def generate_instance(scenario: Scenario, knobs: dict[str, float], seed: int) -> GeneratedInstance:
    """Build a synthetic instance for *scenario* from clamped *knobs* and *seed*."""
    if scenario.model == "humanitarian":
        return _generate_humanitarian(knobs, seed)
    return _generate_ed_staffing(knobs, seed)
