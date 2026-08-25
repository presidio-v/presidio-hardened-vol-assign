"""presidio-hardened-vol-assign CLI entry point."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from presidio_vol_assign import __version__
from presidio_vol_assign.domains import HumanitarianDomain, get_domain
from presidio_vol_assign.engine import run as run_solvers
from presidio_vol_assign.evidence import (
    EvidenceError,
    SigningKeyError,
    load_trust_store,
    verify_record,
)
from presidio_vol_assign.evidence_cli import emit_evidence, resolve_signing_key
from presidio_vol_assign.metrics import compute_metrics
from presidio_vol_assign.models import RunConfig
from presidio_vol_assign.security import (
    AuditResult,
    AuditStatus,
    StructuredLogger,
    get_logger,
    log_startup,
    run_audit,
)
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
# Shared security preamble (Presidio extensions #4 on-run CVE check, #5 event log)
# ---------------------------------------------------------------------------


def _run_security_preamble(log_dir: Path | None = None) -> tuple[StructuredLogger, AuditResult]:
    """Run the mandatory per-invocation security checks for any command.

    Emits the "loaded" security-event banner to ``pva.log`` (in *log_dir* when the
    command has an output directory, otherwise the working directory), runs the
    on-run dependency audit, and surfaces a warning when the audit found
    vulnerabilities. Returns the logger and audit result so the caller can log
    further events.
    """
    logger = get_logger((log_dir / "pva.log") if log_dir is not None else None)
    audit = run_audit()
    log_startup(logger, audit=audit)
    if audit.status == AuditStatus.VULNERABLE:
        err_console.print(f"[yellow]Warning:[/yellow] dependency audit: {audit.summary()}")
    return logger, audit


# ---------------------------------------------------------------------------
# pva assign
# ---------------------------------------------------------------------------


_SOLVER_HELP = "Solver: nsga2, nrga, nrga-ranked, both (nsga2+nrga), or all."


def _build_hard_humanitarian(
    model: str, max_distance: float | None, mobility_threshold: float
) -> HumanitarianDomain:
    """Construct the hard-capacity humanitarian domain, rejecting other models."""
    if model != "humanitarian":
        raise ValueError("--hard-capacity is only available for --model humanitarian")
    return HumanitarianDomain(
        hard_capacity=True,
        max_distance=max_distance,
        mobility_threshold=mobility_threshold,
    )


def _execute_assignment(
    domain,
    primary: Path | None,
    secondary: Path | None,
    *,
    solver: str,
    seed: int | None,
    pop_size: int,
    generations: int,
    output: Path,
    model_label: str,
    emit_evidence_flag: bool = False,
) -> None:
    """Shared body for ``assign`` / ``allocate-people``: validate, solve, write."""
    try:
        out_dir = guard_output_path(output)
    except ValueError as exc:
        err_console.print(f"[red]Security:[/red] {exc}")
        raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    logger, _audit = _run_security_preamble(out_dir)

    # Fail-closed key resolution BEFORE any solving, so we never do the work and
    # then discover we cannot sign. Default OFF: byte-identical behaviour when unset.
    evidence_key: tuple[str, str, bytes] | None = None
    if emit_evidence_flag:
        try:
            evidence_key = resolve_signing_key()
        except SigningKeyError as exc:
            err_console.print(f"[red]Evidence:[/red] {exc}")
            raise typer.Exit(code=1)

    primary_name, secondary_name = domain.required_inputs
    if primary is None or secondary is None:
        err_console.print(
            f"[red]Input error:[/red] model {model_label!r} requires "
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
        f"Problem: [bold]{model_label}[/bold] | {_problem_size(problem)}  "
        f"| solver: [bold]{solver}[/bold]  "
        f"| pop: {pop_size}  gen: {generations}"
    )

    with console.status("[bold green]Running solver(s)…[/bold green]"):
        fronts = run_solvers(problem, config, domain)

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

        if evidence_key is not None:
            signer, alg, key = evidence_key
            evidence_path = emit_evidence(
                front=front,
                metrics=m,
                model=model_label,
                solver=front.solver.value,
                seed=seed,
                pop_size=pop_size,
                generations=generations,
                input_csv_paths=[primary, secondary],
                objective_labels=domain.objective_names,
                assignments_csv_path=assign_path,
                output_dir=out_dir,
                signer=signer,
                alg=alg,
                key=key,
            )
            logger.info(
                "evidence emitted",
                solver=front.solver.value,
                alg=alg,
                signer=signer,
            )
            console.print(f"  Evidence    → [cyan]{evidence_path}[/cyan]")


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
    solver: str = typer.Option("both", "--solver", show_default=True, help=_SOLVER_HELP),
    seed: int = typer.Option(None, "--seed", help="Random seed for reproducibility."),
    pop_size: int = typer.Option(100, "--pop-size", show_default=True, help="GA population size."),
    generations: int = typer.Option(
        200, "--generations", show_default=True, help="Number of generations."
    ),
    hard_capacity: bool = typer.Option(
        False,
        "--hard-capacity",
        help="[humanitarian] Enforce centre capacity as a hard constraint (repair).",
    ),
    max_distance: float = typer.Option(
        None,
        "--max-distance",
        help="[humanitarian, hard mode] Max km a low-mobility person may be sent.",
    ),
    mobility_threshold: float = typer.Option(
        3.0,
        "--mobility-threshold",
        show_default=True,
        help="[humanitarian, hard mode] Mobility below this is transport-limited.",
    ),
    output: Path = typer.Option(
        Path("./results"), "--output", show_default=True, help="Output directory."
    ),
    emit_evidence_flag: bool = typer.Option(
        False,
        "--emit-evidence",
        help=(
            "Emit a signed, content-addressed evidence record per solver run "
            "(evidence_<solver>_<ts>.json). Requires a signing key in the "
            "environment (PVA_EVIDENCE_KEY or PVA_EVIDENCE_ED25519_KEY); "
            "fail-closed when absent. Default off; behaviour unchanged when unset."
        ),
    ),
) -> None:
    """Run assignment optimisation and write Pareto front + metrics."""
    try:
        domain = get_domain(model)
        model_label = model
        if hard_capacity:
            domain = _build_hard_humanitarian(model, max_distance, mobility_threshold)
            model_label = f"{model} (hard-capacity)"
    except ValueError as exc:
        err_console.print(f"[red]Model error:[/red] {exc}")
        raise typer.Exit(code=1)

    provided = {"volunteers": volunteers, "eds": eds, "people": people, "centers": centers}
    primary_name, secondary_name = domain.required_inputs
    _execute_assignment(
        domain,
        provided[primary_name],
        provided[secondary_name],
        solver=solver,
        seed=seed,
        pop_size=pop_size,
        generations=generations,
        output=output,
        model_label=model_label,
        emit_evidence_flag=emit_evidence_flag,
    )


@app.command(name="allocate-people")
def allocate_people(
    people: Path = typer.Option(None, "--people", help="Affected-people CSV."),
    centers: Path = typer.Option(None, "--centers", help="Relief-centres CSV."),
    solver: str = typer.Option("both", "--solver", show_default=True, help=_SOLVER_HELP),
    seed: int = typer.Option(None, "--seed", help="Random seed for reproducibility."),
    pop_size: int = typer.Option(100, "--pop-size", show_default=True, help="GA population size."),
    generations: int = typer.Option(
        200, "--generations", show_default=True, help="Number of generations."
    ),
    hard_capacity: bool = typer.Option(
        False, "--hard-capacity", help="Enforce centre capacity as a hard constraint (repair)."
    ),
    max_distance: float = typer.Option(
        None, "--max-distance", help="[hard mode] Max km a low-mobility person may be sent."
    ),
    mobility_threshold: float = typer.Option(
        3.0,
        "--mobility-threshold",
        show_default=True,
        help="[hard mode] Mobility below this is transport-limited.",
    ),
    output: Path = typer.Option(
        Path("./results"), "--output", show_default=True, help="Output directory."
    ),
    emit_evidence_flag: bool = typer.Option(
        False,
        "--emit-evidence",
        help=(
            "Emit a signed evidence record per solver run (see `pva assign "
            "--help`). Requires PVA_EVIDENCE_KEY or PVA_EVIDENCE_ED25519_KEY; "
            "fail-closed when absent. Default off."
        ),
    ),
) -> None:
    """Allocate affected people to relief centres (humanitarian model).

    Convenience alias for ``assign --model humanitarian``.
    """
    if hard_capacity:
        domain = HumanitarianDomain(
            hard_capacity=True, max_distance=max_distance, mobility_threshold=mobility_threshold
        )
        model_label = "humanitarian (hard-capacity)"
    else:
        domain = get_domain("humanitarian")
        model_label = "humanitarian"
    _execute_assignment(
        domain,
        people,
        centers,
        solver=solver,
        seed=seed,
        pop_size=pop_size,
        generations=generations,
        output=output,
        model_label=model_label,
        emit_evidence_flag=emit_evidence_flag,
    )


# ---------------------------------------------------------------------------
# pva metrics
# ---------------------------------------------------------------------------


@app.command()
def metrics(
    pareto: Path = typer.Option(..., "--pareto", help="Path to a Pareto front CSV."),
) -> None:
    """Compute NNS, MID, SM, and HV for a Pareto front CSV."""
    logger, _audit = _run_security_preamble()

    try:
        front = load_pareto_csv(pareto)
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    m = compute_metrics(front)
    logger.info("metrics computed", solver=front.solver.value, nns=m.nns)
    _print_run_summary(front.solver.value.upper(), m)


# ---------------------------------------------------------------------------
# pva verify-evidence
# ---------------------------------------------------------------------------


@app.command(name="verify-evidence")
def verify_evidence(
    evidence: Path = typer.Option(..., "--evidence", help="Path to an evidence JSON record."),
    trust: Path = typer.Option(
        ..., "--trust", help="Path to a trust store JSON: {signer: {alg, public_key|secret}}."
    ),
) -> None:
    """Verify a signed evidence record offline (fail-closed). Exits 0 on success, 1 otherwise.

    Distinct failure reasons: schema mismatch, float leak, hash mismatch, unknown
    signer, bad signature. No network, no state — the record and the trust store
    are sufficient.
    """
    try:
        raw = evidence.read_text(encoding="utf-8")
    except OSError as exc:
        err_console.print(f"[red]Verify:[/red] cannot read evidence file: {exc}")
        raise typer.Exit(code=1)

    import json as _json

    try:
        record = _json.loads(raw)
    except _json.JSONDecodeError as exc:
        err_console.print(f"[red]Verify:[/red] evidence file is not valid JSON: {exc}")
        raise typer.Exit(code=1)

    try:
        trust_store = load_trust_store(trust)
    except (FileNotFoundError, EvidenceError) as exc:
        err_console.print(f"[red]Verify:[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        verify_record(record, trust_store)
    except EvidenceError as exc:
        reason = type(exc).__name__
        err_console.print(f"[red]Verify FAILED[/red] ({reason}): {exc}")
        raise typer.Exit(code=1)

    signer = record.get("signer")
    console.print(
        f"[green]Verify OK[/green] — signer={signer!r} alg={record.get('alg')!r} "
        f"content_hash={record.get('content_hash')}"
    )
    raise typer.Exit(code=0)


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
    baseline: bool = typer.Option(
        False,
        "--baseline",
        help="Also run the greedy baseline comparator and add it as a 'greedy' row.",
    ),
    exact: bool = typer.Option(
        False,
        "--exact",
        help="Also run the exact weighted-sum comparator and add it as an 'exact' row.",
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
    from presidio_vol_assign.stats import wilcoxon_hv_tests, write_hv_tests_csv

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
    logger, _audit = _run_security_preamble(out_dir)

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
        + ("  (+greedy baseline)" if baseline else "")
        + ("  (+exact baseline)" if exact else "")
        + ("  (+reproducibility check)" if check_repro else "")
    )

    with console.status("[bold green]Running benchmark…[/bold green]"):
        rows = run_benchmark(
            model,
            sizes,
            instances,
            config,
            base_seed=seed,
            check_repro=check_repro,
            include_baseline=baseline,
            include_exact=exact,
        )

    _print_benchmark_table(model, rows, check_repro)
    csv_path, json_path = write_benchmark_summary(rows, out_dir)
    logger.info(
        "benchmark completed",
        model=model,
        sizes=",".join(sizes),
        instances=instances,
        n_rows=len(rows),
    )
    console.print(f"  Summary CSV  → [cyan]{csv_path}[/cyan]")
    console.print(f"  Summary JSON → [cyan]{json_path}[/cyan]")

    # Wilcoxon rank-sum HV significance tests (skipped silently when too few
    # instances or only one solver per size — wilcoxon_hv_tests returns []).
    comparisons = wilcoxon_hv_tests(rows)
    if comparisons:
        _print_stats_table(comparisons)
        stats_path = write_hv_tests_csv(comparisons, out_dir)
        logger.info("benchmark stats computed", n_comparisons=len(comparisons))
        console.print(f"  HV stats CSV → [cyan]{stats_path}[/cyan]")


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

    logger, _audit = _run_security_preamble(out_path.parent)

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

    logger.info("figure rendered", n_fronts=len(pareto), output=str(saved))
    console.print(f"  Figure → [cyan]{saved}[/cyan]")


# ---------------------------------------------------------------------------
# pva sensitivity
# ---------------------------------------------------------------------------


@app.command()
def sensitivity(
    model: str = typer.Option(
        "humanitarian", "--model", show_default=True, help="Problem model to analyse."
    ),
    volunteers: Path = typer.Option(None, "--volunteers", help="[ed-staffing] Volunteer CSV."),
    eds: Path = typer.Option(None, "--eds", help="[ed-staffing] ED vacancies CSV."),
    people: Path = typer.Option(None, "--people", help="[humanitarian] Affected-people CSV."),
    centers: Path = typer.Option(None, "--centers", help="[humanitarian] Relief-centres CSV."),
    factors: str = typer.Option(
        "-0.2,-0.1,0,0.1,0.2",
        "--factors",
        show_default=True,
        help="Comma-separated FIS-output perturbations (signed fractions).",
    ),
    solver: str = typer.Option("both", "--solver", show_default=True, help=_SOLVER_HELP),
    seed: int = typer.Option(42, "--seed", show_default=True, help="Random seed."),
    pop_size: int = typer.Option(100, "--pop-size", show_default=True, help="GA population size."),
    generations: int = typer.Option(
        200, "--generations", show_default=True, help="Number of generations."
    ),
    output: Path = typer.Option(
        Path("./results"), "--output", show_default=True, help="Output directory."
    ),
) -> None:
    """Sweep FIS rule-base perturbations and report how the Pareto metrics shift."""
    from presidio_vol_assign.sensitivity import (
        parse_factors,
        run_sensitivity,
        write_sensitivity_csv,
    )

    try:
        domain = get_domain(model)
        factor_values = parse_factors(factors)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        out_dir = guard_output_path(output)
    except ValueError as exc:
        err_console.print(f"[red]Security:[/red] {exc}")
        raise typer.Exit(code=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger, _audit = _run_security_preamble(out_dir)

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

    factor_str = ", ".join(f"{f:+g}" for f in factor_values)
    console.print(
        f"Sensitivity: [bold]{model}[/bold] | factors: {factor_str} "
        f"| solver: {solver} | pop: {pop_size} gen: {generations}"
    )

    with console.status("[bold green]Running sensitivity sweep…[/bold green]"):
        rows = run_sensitivity(domain, problem, config, factor_values)

    _print_sensitivity_table(model, rows)
    csv_path = write_sensitivity_csv(rows, out_dir)
    logger.info("sensitivity completed", model=model, n_rows=len(rows))
    console.print(f"  Sensitivity CSV → [cyan]{csv_path}[/cyan]")


# ---------------------------------------------------------------------------
# pva ablation
# ---------------------------------------------------------------------------


@app.command()
def ablation(
    model: str = typer.Option(
        "humanitarian", "--model", show_default=True, help="Problem model to analyse."
    ),
    volunteers: Path = typer.Option(None, "--volunteers", help="[ed-staffing] Volunteer CSV."),
    eds: Path = typer.Option(None, "--eds", help="[ed-staffing] ED vacancies CSV."),
    people: Path = typer.Option(None, "--people", help="[humanitarian] Affected-people CSV."),
    centers: Path = typer.Option(None, "--centers", help="[humanitarian] Relief-centres CSV."),
    solver: str = typer.Option("nsga2", "--solver", show_default=True, help=_SOLVER_HELP),
    seed: int = typer.Option(42, "--seed", show_default=True, help="Random seed."),
    pop_size: int = typer.Option(100, "--pop-size", show_default=True, help="GA population size."),
    generations: int = typer.Option(
        200, "--generations", show_default=True, help="Number of generations."
    ),
    output: Path = typer.Option(
        Path("./results"), "--output", show_default=True, help="Output directory."
    ),
) -> None:
    """Leave-one-objective-out ablation: show how much each objective contributes.

    Re-solves the problem with each objective dropped in turn and reports how the
    dropped objective and the overall hypervolume degrade — empirical evidence
    that each qualitative indicator is non-redundant.
    """
    from presidio_vol_assign.ablation import run_ablation, write_ablation_csv

    try:
        domain = get_domain(model)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        out_dir = guard_output_path(output)
    except ValueError as exc:
        err_console.print(f"[red]Security:[/red] {exc}")
        raise typer.Exit(code=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger, _audit = _run_security_preamble(out_dir)

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
        f"Ablation: [bold]{model}[/bold] | solver: {solver} | pop: {pop_size} gen: {generations}"
    )

    try:
        with console.status("[bold green]Running objective ablation…[/bold green]"):
            rows = run_ablation(domain, problem, config)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    _print_ablation_table(model, rows)
    csv_path = write_ablation_csv(rows, out_dir)
    logger.info("ablation completed", model=model, n_rows=len(rows))
    console.print(f"  Ablation CSV → [cyan]{csv_path}[/cyan]")


# ---------------------------------------------------------------------------
# pva version
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address. Use 0.0.0.0 in a container."),
    port: int = typer.Option(8000, help="Port to listen on."),
    cors_origin: list[str] = typer.Option(
        [],
        "--cors-origin",
        help="Extra origin allowed to call the API (repeatable). Not needed for the bundled UI.",
    ),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev only)."),
) -> None:
    """Serve the interactive demo GUI at http://<host>:<port>/.

    Every run is generated from (scenario, sliders, seed) — no data is uploaded
    or stored. Requires the ``web`` extra:

        pip install "presidio-hardened-vol-assign[web]"
    """
    try:
        import uvicorn

        from presidio_vol_assign.web.app import create_app
    except ImportError as exc:
        err_console.print(
            f"[red]Missing dependency:[/red] {exc.name or exc}. The demo server needs the "
            "'web' extra:\n\n    pip install 'presidio-hardened-vol-assign[web]'\n"
        )
        raise typer.Exit(code=1) from exc

    logger = get_logger()
    audit = run_audit()
    log_startup(logger, audit=audit)
    if audit.status != AuditStatus.OK:
        err_console.print(f"[yellow]Dependency audit:[/yellow] {audit.summary()}")

    if host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            f"[yellow]Binding to {host}[/yellow] — the demo has no authentication. "
            "Put it behind a reverse proxy with TLS before exposing it publicly."
        )

    console.print(f"Demo GUI on [cyan]http://{host}:{port}/[/cyan]  (Ctrl-C to stop)")

    if reload:
        # The reloader needs an import string rather than an app instance.
        import os

        os.environ["PVA_WEB_CORS_ORIGINS"] = ",".join(cors_origin)
        uvicorn.run(
            "presidio_vol_assign.web.app:app_from_env",
            host=host,
            port=port,
            reload=True,
            factory=True,
        )
        return

    uvicorn.run(create_app(cors_allow_origins=tuple(cors_origin)), host=host, port=port)


@app.command(name="build-demo")
def build_demo(
    output: Path = typer.Option(
        Path("./dist-demo"), "--output", show_default=True, help="Target directory."
    ),
    grid: str = typer.Option(
        "compact", "--grid", help="Grid density: 'compact' (default) or 'full'."
    ),
    workers: int = typer.Option(
        None, "--workers", help="Solver processes (default: one per core)."
    ),
    shard: str = typer.Option(
        None,
        "--shard",
        help="Solve only part of the grid, as 'i/n' (e.g. 2/4). Overlay every "
        "shard's output to get a complete site.",
    ),
) -> None:
    """Build the demo GUI as a static site, pre-solving a grid of settings.

    The result is a directory of static files — no server required — suitable
    for plain static hosting. Visitors step through the pre-solved settings
    instead of choosing arbitrary values; everything downstream of the solve
    (trade-off slider, map, CSV export) still runs live in the browser.
    """
    if grid not in ("compact", "full"):
        err_console.print(f"[red]Error:[/red] --grid must be 'compact' or 'full', got {grid!r}")
        raise typer.Exit(code=1)

    try:
        out_dir = guard_output_path(output)
    except ValueError as exc:
        err_console.print(f"[red]Security:[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        from presidio_vol_assign.web.static_build import build, parse_shard, plan, shard_points
    except ImportError as exc:  # pragma: no cover - defensive; no web deps needed here
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        shard_spec = parse_shard(shard)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    logger, _audit = _run_security_preamble(out_dir.parent)
    all_points, _values = plan(grid)
    points = shard_points(all_points, shard_spec)
    scope = f" (shard {shard} of {len(all_points)})" if shard_spec else ""
    console.print(
        f"Pre-solving [bold]{len(points)}[/bold] runs ({grid} grid){scope} → [cyan]{out_dir}[/cyan]"
    )

    with typer.progressbar(length=len(points), label="Solving") as bar:
        seen = {"n": 0}

        def advance(done: int, _total: int) -> None:
            bar.update(done - seen["n"])
            seen["n"] = done

        summary = build(
            out_dir, grid_name=grid, workers=workers, progress=advance, shard=shard_spec
        )

    logger.info("static demo built", runs=summary["runs"], output=summary["output"])
    console.print(
        f"  {summary['runs']} runs · {summary['bytes'] / 1_048_576:.1f} MiB → "
        f"[cyan]{summary['output']}[/cyan]"
    )


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
    """Render the Table-3-style mean±std benchmark summary.

    HV (the primary convergence+diversity indicator) leads; MID is reported last
    as a diagnostic only — it rewards proximity to the ideal point and so is not
    a sound stand-alone quality measure for a Pareto front.
    """
    table = Table(title=f"Benchmark summary — {model}", show_header=True, header_style="bold")
    table.add_column("Size", style="dim")
    table.add_column("Solver", style="dim")
    table.add_column("N", justify="right")
    for col in ("HV (primary)", "NNS", "SM", "CPU (s)", "MID (diag.)"):
        table.add_column(col, justify="right")
    if check_repro:
        table.add_column("REP", justify="right")

    for r in rows:
        cells = [
            r.size,
            r.solver,
            str(r.n_instances),
            f"{r.hv_mean:.3f} ± {r.hv_std:.3f}",
            f"{r.nns_mean:.1f} ± {r.nns_std:.1f}",
            f"{r.sm_mean:.4f} ± {r.sm_std:.4f}",
            f"{r.cpu_mean:.2f} ± {r.cpu_std:.2f}",
            f"{r.mid_mean:.3f} ± {r.mid_std:.3f}",
        ]
        if check_repro:
            cells.append("—" if r.rep is None else f"{r.rep:.2f}")
        table.add_row(*cells)
    console.print(table)


def _print_stats_table(comparisons: list) -> None:
    """Render the Wilcoxon rank-sum HV significance comparisons."""
    table = Table(
        title="HV significance — Wilcoxon rank-sum (vs. reference solver)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Size", style="dim")
    table.add_column("Solver", style="dim")
    table.add_column("Reference", style="dim")
    for col in ("median HV", "ref median HV", "p-value"):
        table.add_column(col, justify="right")
    table.add_column("Better")
    table.add_column("Sig. (α=.05)", justify="center")

    for c in comparisons:
        table.add_row(
            c.size,
            c.solver,
            c.reference,
            f"{c.median:.3f}",
            f"{c.reference_median:.3f}",
            f"{c.p_value:.4f}",
            c.better,
            "✓" if c.significant else "—",
        )
    console.print(table)


def _print_sensitivity_table(model: str, rows: list) -> None:
    """Render the FIS-perturbation sensitivity sweep."""
    table = Table(title=f"Sensitivity sweep — {model}", show_header=True, header_style="bold")
    table.add_column("Perturb", justify="right", style="dim")
    table.add_column("Solver", style="dim")
    for col in ("NNS", "MID", "SM", "HV", "CPU (s)"):
        table.add_column(col, justify="right")
    for r in rows:
        table.add_row(
            f"{r.factor:+.0%}",
            r.solver,
            str(r.nns),
            f"{r.mid:.4f}",
            f"{r.sm:.4f}",
            f"{r.hv:.4f}",
            f"{r.cpu_time_sec:.2f}",
        )
    console.print(table)


def _print_ablation_table(model: str, rows: list) -> None:
    """Render the leave-one-objective-out ablation summary."""
    table = Table(
        title=f"Objective ablation — {model} (larger Δ = more non-redundant)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Dropped", style="dim")
    table.add_column("NNS (full→abl.)", justify="right")
    table.add_column("mean (full)", justify="right")
    table.add_column("mean (ablated)", justify="right")
    table.add_column("Δ dropped", justify="right")
    table.add_column("HV (full)", justify="right")
    table.add_column("HV (ablated)", justify="right")
    table.add_column("Δ HV", justify="right")
    for r in rows:
        table.add_row(
            r.dropped,
            f"{r.nns_full}→{r.nns_ablated}",
            f"{r.mean_dropped_full:.4f}",
            f"{r.mean_dropped_ablated:.4f}",
            f"{r.delta_dropped:+.4f}",
            f"{r.hv_full:.4f}",
            f"{r.hv_ablated:.4f}",
            f"{r.delta_hv:+.4f}",
        )
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
    table.add_column("Metric", style="dim", width=16)
    table.add_column("Value", justify="right")

    # HV first: it is the primary indicator (captures convergence + diversity).
    table.add_row("HV (primary)", f"{m.hv:.4f}")
    table.add_row("NNS", str(m.nns))
    table.add_row("SM", f"{m.sm:.4f}")
    table.add_row("CPU time", f"{m.cpu_time_sec:.2f}s")
    # MID last and flagged diagnostic: it favours solutions near the ideal point,
    # so it is not a sound stand-alone quality measure for a Pareto front.
    table.add_row("MID (diag.)", f"{m.mid:.4f}")

    console.print(table)
