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
        model: Which solver model backs it (``ed-staffing`` / ``humanitarian`` /
            ``allocation``). ``allocation`` is the published relief-allocation
            model (:mod:`presidio_vol_assign.allocation`) audited in Paper B.
        hard_capacity: Humanitarian hard-constraint (repair) mode; for
            ``allocation`` it selects the repaired model (load-coupled CAIL,
            hard capacity in the exact reference).
        unit_label / site_label: Plural nouns for the two sides of the problem.
        objectives: Plain-language objective descriptors, in solver order.
        knobs: Sliders shown for this scenario.
        cli_hint: The equivalent `pva` invocation, shown so the GUI stays an
            honest front-end for the CLI rather than a separate implementation.
            Empty when no CLI command exists (the allocation scenarios run the
            library directly); the page then says so instead.
    """

    id: str
    title: str
    subtitle: str
    description: str
    model: Literal["ed-staffing", "humanitarian", "allocation"]
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

_ALLOCATION_KNOBS = (
    Knob(
        key="n_people",
        label="People in the affected area",
        minimum=30,
        maximum=300,
        step=30,
        default=150,
        help="A third of them can be directed to a centre in this round.",
    ),
    Knob(
        key="n_centers",
        label="Relief centres open",
        minimum=2,
        maximum=10,
        step=1,
        default=5,
        help="How many centres can receive people.",
    ),
)

_ALLOCATION_OBJECTIVES = (
    Objective(
        "Mn_ULPP",
        "Unfairness in who is helped first",
        "How far the people directed now fall short of being the most urgent cases.",
    ),
    Objective(
        "Mn_TIL",
        "Transport infeasibility",
        "How long and how unsafe the journeys to the assigned centres are.",
    ),
    Objective(
        "Mn_CAIL",
        "Centre imbalance",
        "How strained the receiving centres are by occupancy and resource depletion.",
    ),
)

KAPPA_F_KNOB = Knob(
    key="kappa_f",
    label="Free places per person to place",
    minimum=1.1,
    maximum=2.0,
    step=0.1,
    default=1.5,
    help="Total free places across all centres as a multiple of the people being directed.",
    integer=False,
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
    Scenario(
        id="relief-published",
        title="Relief allocation: published model",
        subtitle="The evolutionary search next to the exact answer.",
        description=(
            "This is the relief-allocation model exactly as published. Its objectives "
            "decompose person by person, so a simple exact rule computes the best "
            "trade-offs instantly; compare the evolutionary fronts with the exact one."
        ),
        model="allocation",
        hard_capacity=False,
        unit_label="people",
        site_label="relief centres",
        objectives=_ALLOCATION_OBJECTIVES,
        knobs=_ALLOCATION_KNOBS,
        cli_hint="",
    ),
    Scenario(
        id="relief-repaired",
        title="Relief allocation: repaired model (load-aware, hard capacity)",
        subtitle="Centre strain now grows with the people sent there.",
        description=(
            "The repaired model lets centre imbalance rise with the load each "
            "allocation creates, and caps every centre at its free places. The exact "
            "reference is a mixed-integer program that respects those caps; the "
            "evolutionary search has no capacity handling, so its options may "
            "overfill a centre."
        ),
        model="allocation",
        hard_capacity=True,
        unit_label="people",
        site_label="relief centres",
        objectives=_ALLOCATION_OBJECTIVES,
        knobs=_ALLOCATION_KNOBS + (KAPPA_F_KNOB,),
        cli_hint="",
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
        extras: Solver-side data that is not part of the problem object, e.g.
            the centre capacities of the repaired allocation model.
    """

    problem: Any
    unit_points: list[dict[str, Any]] = field(default_factory=list)
    site_points: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


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


# Categorical distributions of the published experiment generator
# (pva-paperB ``experiments/generate_instances.py``), mirrored so the demo
# instances have the same demographic and route character as the paper's.
_DISABILITY_P = (0.70, 0.20, 0.10)
_INJURY_P = (0.40, 0.25, 0.20, 0.10, 0.05)
_LIVING_P = (0.65, 0.35)
_RCS_P = (0.45, 0.40, 0.15)
_PHS_P = (0.30, 0.30, 0.25, 0.10, 0.05)

ROAD_DETOUR = 1.3
"""Road distance per straight-line km (a common circuity factor)."""

ROAD_SPEED_KMH = 60.0
"""Average road speed. The published generator also converts km at 60 km/h."""

TRAVEL_MINUTES_RANGE = (2.0, 180.0)
"""Valid TD range: the allocation validator accepts [0, 180] and the published
generator clips into [2, 180]."""


def _generate_allocation(
    knobs: dict[str, float], seed: int, *, repaired: bool
) -> GeneratedInstance:
    """Build a relief-allocation instance (published model) on the area grid.

    Person and centre attributes follow the published generator's
    distributions; travel duration comes from the Euclidean distance on the
    grid (with a detour factor) rather than an independent draw, so the map
    and the objective values tell the same story. ``n_dir`` is a third of the
    people, as in the published instance sizes (150/50, 225/75, 300/100).

    For the repaired model the centres also get a capacity, sized so their
    free places sum to ``kappa_f * n_dir``; whole free places per centre are
    the hard limits the exact reference must respect.
    """
    from presidio_vol_assign.allocation.fis import compute_vs
    from presidio_vol_assign.allocation.load_coupling import (
        capacities_from_free_factor,
        hard_load_limits,
    )
    from presidio_vol_assign.allocation.models import (
        AllocationProblem,
        DisabilityStatus,
        HazardLevel,
        InjuryLevel,
        LivingStatus,
        ReliefCenter,
        RoadCondition,
        TravelInfo,
        Weights,
    )
    from presidio_vol_assign.allocation.models import Person as AllocPerson

    rng = np.random.default_rng(seed)
    n_people = int(knobs["n_people"])
    n_centers = int(knobs["n_centers"])
    n_dir = max(1, n_people // 3)

    center_xy = rng.uniform(0, AREA_KM, size=(n_centers, 2))
    people_xy = rng.uniform(0, AREA_KM, size=(n_people, 2))

    age = np.round(rng.uniform(5, 90, size=n_people), 1)
    disability = rng.choice(len(DisabilityStatus), size=n_people, p=_DISABILITY_P)
    injury = rng.choice(len(InjuryLevel), size=n_people, p=_INJURY_P)
    living = rng.choice(len(LivingStatus), size=n_people, p=_LIVING_P)
    idl = np.round(rng.beta(2.5, 2.0, size=n_people) * 100, 2)
    rtr = np.round(rng.uniform(1.0, 48.0, size=n_people), 2)

    cor = np.round(rng.uniform(20, 90, size=n_centers), 2)
    rdr = np.round(rng.uniform(10, 80, size=n_centers), 2)

    deltas = people_xy[:, None, :] - center_xy[None, :, :]
    km = np.sqrt((deltas**2).sum(axis=-1)) * ROAD_DETOUR
    minutes = np.clip(km / ROAD_SPEED_KMH * 60.0, *TRAVEL_MINUTES_RANGE)
    rcs = rng.choice(len(RoadCondition), size=(n_people, n_centers), p=_RCS_P)
    phs = rng.choice(len(HazardLevel), size=(n_people, n_centers), p=_PHS_P)

    disability_levels = list(DisabilityStatus)
    injury_levels = list(InjuryLevel)
    living_levels = list(LivingStatus)
    road_levels = list(RoadCondition)
    hazard_levels = list(HazardLevel)

    people = [
        AllocPerson(
            person_id=f"P{i + 1}",
            age=float(age[i]),
            disability_status=disability_levels[int(disability[i])],
            injury_level=injury_levels[int(injury[i])],
            living_status=living_levels[int(living[i])],
            infrastructure_damage_level=float(idl[i]),
            resource_time_remaining=float(rtr[i]),
        )
        for i in range(n_people)
    ]
    centers = [
        ReliefCenter(
            center_id=f"C{j + 1}",
            center_occupancy_rate=float(cor[j]),
            resource_depletion_rate=float(rdr[j]),
        )
        for j in range(n_centers)
    ]
    travel = {
        (people[i].person_id, centers[j].center_id): TravelInfo(
            person_id=people[i].person_id,
            center_id=centers[j].center_id,
            travel_duration=round(float(minutes[i, j]), 2),
            road_condition=road_levels[int(rcs[i, j])],
            possible_hazard=hazard_levels[int(phs[i, j])],
        )
        for i in range(n_people)
        for j in range(n_centers)
    }
    problem = AllocationProblem(people=people, centers=centers, travel=travel, n_dir=n_dir)

    extras: dict[str, Any] = {}
    free_places: list[int | None] = [None] * n_centers
    summary: dict[str, Any] = {"units": n_people, "sites": n_centers, "directed": n_dir}
    if repaired:
        kappa_f = float(knobs["kappa_f"])
        capacity = capacities_from_free_factor(problem, kappa_f)
        limits = hard_load_limits(problem, capacity)
        # Flooring each centre's free places can leave fewer than n_dir whole
        # places in total; grow the factor until the hard limits are feasible
        # so the exact reference never receives an infeasible instance.
        effective = kappa_f
        while int(limits.sum()) < n_dir:
            effective *= 1.05
            capacity = capacities_from_free_factor(problem, effective)
            limits = hard_load_limits(problem, capacity)
        extras = {"capacity": capacity, "limits": limits, "kappa_f": effective}
        free_places = [int(v) for v in limits]
        summary.update(
            {
                "kappaF": round(kappa_f, 3),
                "kappaFEffective": round(effective, 3),
                "freePlaces": int(limits.sum()),
                "centreSize": round(float(capacity[0]), 2),
            }
        )

    vs_weights = Weights()
    unit_points = [
        {
            "id": people[i].person_id,
            "x": round(float(people_xy[i, 0]), 2),
            "y": round(float(people_xy[i, 1]), 2),
            "label": (
                f"{people[i].person_id} · age {people[i].age:g} · "
                f"injury {people[i].injury_level.value.replace('_', ' ')}"
            ),
            "weight": 1,
            "priority": round(compute_vs(people[i], vs_weights) * 10, 2),
        }
        for i in range(n_people)
    ]
    site_points = []
    for j in range(n_centers):
        label = f"{centers[j].center_id} · {centers[j].center_occupancy_rate:g}% occupied"
        if free_places[j] is not None:
            label += f" · {free_places[j]} free places"
        site_points.append(
            {
                "id": centers[j].center_id,
                "x": round(float(center_xy[j, 0]), 2),
                "y": round(float(center_xy[j, 1]), 2),
                "label": label,
                # None tells the page the published model has no capacity at all.
                "capacity": free_places[j],
            }
        )

    return GeneratedInstance(
        problem=problem,
        unit_points=unit_points,
        site_points=site_points,
        summary=summary,
        extras=extras,
    )


def generate_instance(scenario: Scenario, knobs: dict[str, float], seed: int) -> GeneratedInstance:
    """Build a synthetic instance for *scenario* from clamped *knobs* and *seed*."""
    if scenario.model == "allocation":
        return _generate_allocation(knobs, seed, repaired=scenario.hard_capacity)
    if scenario.model == "humanitarian":
        return _generate_humanitarian(knobs, seed)
    return _generate_ed_staffing(knobs, seed)
