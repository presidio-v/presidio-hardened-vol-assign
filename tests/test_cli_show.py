"""Integration tests for the `pva show` command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from presidio_vol_assign.cli import app

runner = CliRunner()


def _write_pareto(path: Path, *objective_rows: tuple[float, ...]) -> None:
    dim = len(objective_rows[0])
    header = ["solver", "solution_id"] + [f"z{k}" for k in range(1, dim + 1)]
    lines = [",".join(header)]
    for i, row in enumerate(objective_rows):
        lines.append(",".join(["nsga2", str(i)] + [str(v) for v in row]))
    path.write_text("\n".join(lines) + "\n")


def test_show_2d_writes_png(tmp_path: Path) -> None:
    pareto = tmp_path / "pareto.csv"
    _write_pareto(pareto, (0.2, 0.8), (0.5, 0.5), (0.8, 0.2))
    out = tmp_path / "fig.png"
    result = runner.invoke(app, ["show", "--pareto", str(pareto), "--output", str(out)])
    assert result.exit_code == 0, result.output + (result.stderr or "")
    assert out.exists()
    # SDLC #5: a security-event log is emitted next to the figure
    log = out.parent / "pva.log"
    assert log.exists()
    assert "loaded" in log.read_text()


def test_show_3d_writes_png(tmp_path: Path) -> None:
    pareto = tmp_path / "pareto3.csv"
    _write_pareto(pareto, (0.2, 0.8, 0.3), (0.5, 0.5, 0.5))
    out = tmp_path / "fig3.png"
    result = runner.invoke(app, ["show", "--pareto", str(pareto), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_show_overlay_two_csvs(tmp_path: Path) -> None:
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    _write_pareto(p1, (0.2, 0.8), (0.5, 0.5))
    _write_pareto(p2, (0.3, 0.7), (0.6, 0.4))
    out = tmp_path / "overlay.png"
    result = runner.invoke(
        app, ["show", "--pareto", str(p1), "--pareto", str(p2), "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_show_missing_file_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["show", "--pareto", str(tmp_path / "nope.csv"), "--output", str(tmp_path / "x.png")]
    )
    assert result.exit_code == 1
