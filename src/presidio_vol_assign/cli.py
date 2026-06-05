"""presidio-hardened-vol-assign CLI entry point."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from presidio_vol_assign import __version__
from presidio_vol_assign.domains import get_domain
from presidio_vol_assign.engine import run as run_solvers
from presidio_vol_assign.metrics import compute_metrics
from presidio_vol_assign.models import RunConfig
from presidio_vol_assign.security import AuditStatus, get_logger, log_startup, run_audit
from presidio_vol_assign.validation import guard_output_path, validate_run_config
from presidio_vol_assign.writers import (
    load_pareto_csv,
    write_assignments_csv,
    write_metrics_json,
    write_pareto_csv,
)

app = typer.Typer(
    name="pva",
    help=(
        "Multi-objective post-disaster assignment. Two models: ed-staffing "
        "(volunteer -> ED, 2 objectives) and humanitarian (people -> relief "
        "centres, 3 objectives)."
    ),
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# pva assign
# ---------------------------------------------------------------------------


@app.command()
def assign(
    model: str = typer.Option(
        "ed-staffing",
        "--model",
        show_default=True,
        help="Problem model: ed-staffing (2 objectives) or humanitarian (3 objectives).",
    ),
    volunteers: Path = typer.Option(
        None, "--volunteers", help="[ed-staffing] Volunteer roster CSV."
    ),
    eds: Path = typer.Option(
        None, "--eds", help="[ed-staffing] Emergency Department vacancies CSV."
    ),
    people: Path = typer.Option(None, "--people", help="[humanitarian] Affected-people CSV."),
    centers: Path = typer.Option(None, "--centers", help="[humanitarian] Relief-centres CSV."),
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
    """Run assignment optimisation and write Pareto front + metrics."""
    # ---- Resolve model ----
    try:
        domain = get_domain(model)
    except ValueError as exc:
        err_console.print(f"[red]Model error:[/red] {exc}")
        raise typer.Exit(code=1)

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

    # ---- Select + validate inputs for this model ----
    provided = {"volunteers": volunteers, "eds": eds, "people": people, "centers": centers}
    primary_name, secondary_name = domain.required_inputs
    primary, secondary = provided[primary_name], provided[secondary_name]
    if primary is None or secondary is None:
        err_console.print(
            f"[red]Input error:[/red] model {model!r} requires "
            f"--{primary_name} and --{secondary_name}."
        )
        raise typer.Exit(code=1)

    try:
        problem = domain.load(primary, secondary)
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
        f"Problem: [bold]{model}[/bold] | {_problem_size(problem)}  "
        f"| solver: [bold]{solver}[/bold]  "
        f"| pop: {pop_size}  gen: {generations}"
    )

    # ---- Solve ----
    with console.status("[bold green]Running solver(s)…[/bold green]"):
        fronts = run_solvers(problem, config, domain)

    # ---- Write outputs + print summary ----
    for front in fronts:
        pareto_path = write_pareto_csv(front, out_dir)
        assign_path = write_assignments_csv(front, out_dir, domain)
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
# pva benchmark
# ---------------------------------------------------------------------------


@app.command()
def benchmark(
    model: str = typer.Option(
        "humanitarian", "--model", show_default=True, help="Model to benchmark."
    ),
    size: str = typer.Option(
        "both", "--size", show_default=True, help="Instance size: small, large, or both."
    ),
    instances: int = typer.Option(
        10, "--instances", show_default=True, help="Random instances per size class."
    ),
    solver: str = typer.Option("both", "--solver", show_default=True, help="Solver(s) to run."),
    seed: int = typer.Option(42, "--seed", show_default=True, help="Base seed (instances+solver)."),
    pop_size: int = typer.Option(100, "--pop-size", show_default=True, help="GA population size."),
    generations: int = typer.Option(
        200, "--generations", show_default=True, help="Number of generations."
    ),
    check_repro: bool = typer.Option(
        False, "--check-repro", help="Re-run each instance and report bit-for-bit REP."
    ),
    output: Path = typer.Option(
        Path("./results"), "--output", show_default=True, help="Output directory."
    ),
) -> None:
    """Generate the paper's instances, run the solver(s), and summarise metrics."""
    from presidio_vol_assign.benchmark import (
        resolve_sizes,
        run_benchmark,
        write_benchmark_summary,
    )

    try:
        get_domain(model)  # validate model early
        sizes = resolve_sizes(size)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        out_dir = guard_output_path(output)
    except ValueError as exc:
        err_console.print(f"[red]Security:[/red] {exc}")
        raise typer.Exit(code=1)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        f"Benchmark: [bold]{model}[/bold] | sizes: {', '.join(sizes)} "
        f"| instances/size: {instances} | solver: {solver} | pop: {pop_size} gen: {generations}"
        + ("  (+reproducibility check)" if check_repro else "")
    )

    with console.status("[bold green]Running benchmark…[/bold green]"):
        rows = run_benchmark(
            model, sizes, instances, config, base_seed=seed, check_repro=check_repro
        )

    _print_benchmark_table(model, rows, check_repro)
    csv_path, json_path = write_benchmark_summary(rows, out_dir)
    console.print(f"  Summary CSV  → [cyan]{csv_path}[/cyan]")
    console.print(f"  Summary JSON → [cyan]{json_path}[/cyan]")


# ---------------------------------------------------------------------------
# pva show
# ---------------------------------------------------------------------------


@app.command()
def show(
    pareto: list[Path] = typer.Option(
        ..., "--pareto", help="Pareto CSV(s) to plot; repeat to overlay solvers."
    ),
    output: Path = typer.Option(
        Path("./results/pareto.png"),
        "--output",
        show_default=True,
        help="Image path (.png or .svg by extension).",
    ),
    title: str = typer.Option(None, "--title", help="Optional figure title."),
) -> None:
    """Render publication-quality Pareto-front figures from one or more CSVs.

    Two-objective fronts produce a Z1-Z2 scatter; three-objective fronts produce
    the three pairwise projections plus a 3-D scatter, with solvers overlaid.
    """
    try:
        out_path = guard_output_path(output)
    except ValueError as exc:
        err_console.print(f"[red]Security:[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        from presidio_vol_assign.viz import plot_fronts
    except ImportError:
        err_console.print(
            "[red]Error:[/red] plotting requires matplotlib. "
            "Install it with: pip install 'presidio-hardened-vol-assign[viz]'"
        )
        raise typer.Exit(code=1)

    try:
        fronts = [load_pareto_csv(p) for p in pareto]
        saved = plot_fronts(fronts, out_path, title=title)
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"  Figure → [cyan]{saved}[/cyan]")


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


def _print_benchmark_table(model: str, rows: list, check_repro: bool) -> None:
    """Render the Table-3-style mean±std benchmark summary."""
    table = Table(title=f"Benchmark summary — {model}", show_header=True, header_style="bold")
    table.add_column("Size", style="dim")
    table.add_column("Solver", style="dim")
    table.add_column("N", justify="right")
    for col in ("NNS", "MID", "SM", "HV", "CPU (s)"):
        table.add_column(col, justify="right")
    if check_repro:
        table.add_column("REP", justify="right")

    for r in rows:
        cells = [
            r.size,
            r.solver,
            str(r.n_instances),
            f"{r.nns_mean:.1f} ± {r.nns_std:.1f}",
            f"{r.mid_mean:.3f} ± {r.mid_std:.3f}",
            f"{r.sm_mean:.4f} ± {r.sm_std:.4f}",
            f"{r.hv_mean:.3f} ± {r.hv_std:.3f}",
            f"{r.cpu_mean:.2f} ± {r.cpu_std:.2f}",
        ]
        if check_repro:
            cells.append("—" if r.rep is None else f"{r.rep:.2f}")
        table.add_row(*cells)
    console.print(table)


def _problem_size(problem: object) -> str:
    """Human-readable size description, model-agnostic."""
    if hasattr(problem, "n_volunteers"):
        return f"{problem.n_volunteers} volunteers, {problem.n_vacancies} vacancies"
    if hasattr(problem, "n_people"):
        return f"{problem.n_people} people, {problem.n_centers} centres"
    return "unknown size"


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
