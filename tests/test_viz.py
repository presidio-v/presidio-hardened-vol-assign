"""Tests for the Pareto-front plotting module."""

from __future__ import annotations

from pathlib import Path

import pytest

from presidio_vol_assign.models import ParetoFront, Solution, SolverType
from presidio_vol_assign.viz import plot_fronts


def _front2d(solver: SolverType) -> ParetoFront:
    sols = [
        Solution(assignments=[], objectives=(0.2, 0.8)),
        Solution(assignments=[], objectives=(0.6, 0.4)),
    ]
    return ParetoFront(solver=solver, solutions=sols)


def _front3d(solver: SolverType) -> ParetoFront:
    sols = [
        Solution(assignments=[], objectives=(0.2, 0.8, 0.3)),
        Solution(assignments=[], objectives=(0.6, 0.4, 0.5)),
    ]
    return ParetoFront(solver=solver, solutions=sols)


def test_plot_2d_creates_png(tmp_path: Path) -> None:
    out = tmp_path / "front.png"
    result = plot_fronts([_front2d(SolverType.NSGA2)], out, title="2D")
    assert result == out
    assert out.exists() and out.stat().st_size > 0


def test_plot_2d_overlays_two_solvers(tmp_path: Path) -> None:
    out = tmp_path / "overlay.png"
    plot_fronts([_front2d(SolverType.NSGA2), _front2d(SolverType.NRGA)], out)
    assert out.exists()


def test_plot_3d_creates_svg(tmp_path: Path) -> None:
    out = tmp_path / "front.svg"
    plot_fronts([_front3d(SolverType.NSGA2), _front3d(SolverType.NRGA)], out)
    assert out.exists() and out.stat().st_size > 0


def test_empty_fronts_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no fronts"):
        plot_fronts([], tmp_path / "x.png")


def test_inconsistent_dimensions_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        plot_fronts([_front2d(SolverType.NSGA2), _front3d(SolverType.NRGA)], tmp_path / "x.png")


def test_creates_parent_directory(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "deep" / "front.png"
    plot_fronts([_front2d(SolverType.NSGA2)], out)
    assert out.exists()
