"""presidio-hardened-vol-assign CLI entry point."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from presidio_vol_assign import __version__
from presidio_vol_assign.metrics import compute_metrics
from presidio_vol_assign.models import RunConfig
from presidio_vol_assign.security import AuditStatus, get_logger, log_startup, run_audit
from presidio_vol_assign.solvers import solve
from presidio_vol_assign.validation import guard_output_path, load_problem, validate_run_config
from presidio_vol_assign.writers import (
    load_pareto_csv,
    write_assignments_csv,
    write_metrics_json,
    write_pareto_csv,
)

app = typer.Typer(
    name="pva",
    help="Multi-objective volunteer assignment for post-disaster ED staffing.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# pva assign
# ---------------------------------------------------------------------------


@app.command()
def assign(
    volunteers: Path = typer.Option(..., "--volunteers", help="Volunteer roster CSV."),
    eds: Path = typer.Option(..., "--eds", help="Emergency Department vacancies CSV."),
    solver: str = typer.Option(
        "both", "--solver", show_default=True, help="Solver: nsga2, nrga, or both."
    ),
    seed: int = typer.Option(None, "--seed", help="Random seed for reproducibility."),
    pop_size: int = typer.Option(100, "--pop-size", show_default=True, help="GA population size."),
    generations: int = typer.Option(
        200, "--generations", show_default=True, help="Number of generations."
    ),
    output: Path = typer.Option(
        Path("./results"), "--output", show_default=True, help="Output directory."
    ),
) -> None:
    """Run volunteer assignment optimisation and write Pareto front + metrics."""
    # ---- Security ----
    try:
        out_dir = guard_output_path(output)
    except ValueError as exc:
        err_console.print(f"[red]Security:[/red] {exc}")
        raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(out_dir / "pva.log")
    audit = run_audit()
    log_startup(logger, audit=audit)

    if audit.status == AuditStatus.VULNERABLE:
        err_console.print(f"[yellow]Warning:[/yellow] dependency audit: {audit.summary()}")

    # ---- Validate inputs ----
    try:
        problem = load_problem(volunteers, eds)
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"[red]Input error:[/red] {exc}")
        raise typer.Exit(code=1)

    config = RunConfig(
        solver=solver,
        pop_size=pop_size,
        generations=generations,
        seed=seed,
        output_dir=str(out_dir),
    )
    try:
        validate_run_config(config)
    except ValueError as exc:
        err_console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(
        f"Problem: [bold]{problem.n_volunteers}[/bold] volunteers, "
        f"[bold]{problem.n_vacancies}[/bold] vacancies  "
        f"| solver: [bold]{solver}[/bold]  "
        f"| pop: {pop_size}  gen: {generations}"
    )

    # ---- Solve ----
    with console.status("[bold green]Running solver(s)…[/bold green]"):
        fronts = solve(problem, config)

    # ---- Write outputs + print summary ----
    for front in fronts:
        pareto_path = write_pareto_csv(front, out_dir)
        assign_path = write_assignments_csv(front, out_dir)
        m = compute_metrics(front)
        metrics_path = write_metrics_json(m, out_dir)

        logger.info(
            "solver completed",
            solver=front.solver.value,
            nns=m.nns,
            cpu_time_sec=round(m.cpu_time_sec, 3),
        )

        _print_run_summary(front.solver.value.upper(), m)
        console.print(f"  Pareto CSV  → [cyan]{pareto_path}[/cyan]")
        console.print(f"  Assignments → [cyan]{assign_path}[/cyan]")
        console.print(f"  Metrics     → [cyan]{metrics_path}[/cyan]")


# ---------------------------------------------------------------------------
# pva metrics
# ---------------------------------------------------------------------------


@app.command()
def metrics(
    pareto: Path = typer.Option(..., "--pareto", help="Path to a Pareto front CSV."),
) -> None:
    """Compute NNS, MID, SM, and HV for a Pareto front CSV."""
    try:
        front = load_pareto_csv(pareto)
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    m = compute_metrics(front)
    _print_run_summary(front.solver.value.upper(), m)


# ---------------------------------------------------------------------------
# pva version
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print version and dependency check status."""
    console.print(f"presidio-hardened-vol-assign {__version__}")
    audit = run_audit()
    logger = get_logger()
    log_startup(logger, audit=audit)
    colour = "green" if audit.status == AuditStatus.OK else "yellow"
    console.print(f"Dependency audit: [{colour}]{audit.summary()}[/{colour}]")


# ---------------------------------------------------------------------------
# Shared display helper
# ---------------------------------------------------------------------------


def _print_run_summary(solver_label: str, m: Metrics) -> None:  # noqa: F821

    table = Table(title=f"Results — {solver_label}", show_header=True, header_style="bold")
    table.add_column("Metric", style="dim", width=14)
    table.add_column("Value", justify="right")

    table.add_row("NNS", str(m.nns))
    table.add_row("MID", f"{m.mid:.4f}")
    table.add_row("SM", f"{m.sm:.4f}")
    table.add_row("HV", f"{m.hv:.4f}")
    table.add_row("CPU time", f"{m.cpu_time_sec:.2f}s")

    console.print(table)
