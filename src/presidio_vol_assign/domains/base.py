"""The ``Domain`` adapter interface.

A domain supplies everything that varies between problem families while the
shared engine (``engine.py``) owns the generic NSGA-II / NRGA machinery:
population management, (mu + lambda) survival, non-dominated sorting, and
Pareto-front extraction.

Concrete domains implement the abstract hooks below. The metadata class
attributes describe the objective space (used by the metrics layer for the
ideal point, the hypervolume reference point, and the CSV column names) and the
DEAP ``creator`` type names the engine should register for this domain's
chromosome encoding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from presidio_vol_assign.models import Solution


class Domain(ABC):
    """Adapter binding one problem family to the shared evolutionary engine."""

    # --- identity / objective-space metadata (override in subclasses) ---
    name: str = ""
    objective_names: tuple[str, ...] = ()
    reference_point: tuple[float, ...] = ()  # hypervolume reference (worst) point
    ideal_point: tuple[float, ...] = ()  # MID ideal (best) point
    weights: tuple[float, ...] = ()  # DEAP fitness weights (-1 = minimise)

    # DEAP creator attribute names for this domain's fitness / individual types.
    fitness_attr: str = ""
    individual_attr: str = ""

    # CLI input-file roles, in the order ``load`` expects them
    # (e.g. ("volunteers", "eds") or ("people", "centers")).
    required_inputs: tuple[str, str] = ()

    # Column order for the per-assignment CSV this domain writes.
    assignment_fieldnames: tuple[str, ...] = ()

    @property
    def n_objectives(self) -> int:
        return len(self.objective_names)

    # ------------------------------------------------------------------
    # I/O hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, primary: Path, secondary: Path) -> Any:
        """Parse and validate the two input CSVs into a problem instance.

        ``primary`` / ``secondary`` correspond to ``required_inputs[0]`` and
        ``required_inputs[1]`` respectively.
        """

    @abstractmethod
    def assignment_row(self, assignment: Any) -> dict[str, Any]:
        """Serialise one assignment to a CSV row dict keyed by
        ``assignment_fieldnames`` (excluding ``solution_id``, which the writer
        prepends)."""

    # ------------------------------------------------------------------
    # Evolutionary hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def precompute(self, problem: Any) -> Any:
        """Build any reusable per-instance cache (e.g. FIS scores).

        Returned object is treated opaquely by the engine and handed back to
        ``evaluate`` / ``to_solution``.
        """

    def perturb(self, cache: Any, factor: float) -> Any:
        """Return a copy of *cache* with FIS output scores scaled by ``1 + factor``.

        Used by the sensitivity analysis to probe robustness to FIS rule-base
        specification uncertainty. ``factor`` is a signed fraction (e.g. -0.1 for
        -10 %); scaled scores are clipped to [0, 1]. The default returns the cache
        unchanged; domains with FIS scores override it.
        """
        return cache

    @abstractmethod
    def init_individual(self, problem: Any, individual_cls: type) -> list:
        """Create one random feasible individual wrapped in ``individual_cls``."""

    @abstractmethod
    def mate(self, ind1: list, ind2: list) -> tuple[list, list]:
        """Apply crossover in place and return the two children."""

    @abstractmethod
    def mutate(self, ind: list) -> tuple[list]:
        """Apply mutation in place and return the (mutated) individual."""

    @abstractmethod
    def evaluate(self, individual: list, cache: Any, problem: Any) -> tuple[float, ...]:
        """Return the objective tuple for ``individual`` (length n_objectives)."""

    @abstractmethod
    def to_solution(self, individual: list, cache: Any, problem: Any) -> Solution:
        """Reconstruct a full :class:`Solution` from an evaluated individual.

        Reads the already-computed objective values from
        ``individual.fitness.values``.
        """
