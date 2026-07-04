# Paper B — Phase 2 scoping (design kickoff)

Turns the design brief into a locked scope. Two items need Daniel/GE sign-off before we
build experiments (§5); everything else is actionable now.

## 1. Framing bridge — recommendation: B1 (refined)

The SI is about **digital AI systems under *economic* turbulence and uncertainty**; our
substrate is post-disaster allocation. Honest bridge:

> Position the tool as a **digital AI decision system for allocating scarce resources under
> acute turbulence and uncertainty.** Post-disaster relief allocation is the *acute
> economic-shock* instance — allocating scarce resources (capacity, transport, prioritisation)
> when a sudden shock has made demand exceed supply and the input data itself is degraded.
> The "AI system" is the fuzzy-inference + multi-objective engine; the "response to
> turbulence" is that it (a) encodes expert judgement to tolerate imprecise/uncertain inputs,
> (b) yields auditable trade-off decisions, and (c) is reproducible under scrutiny.

This keeps the contribution on *systems* ground (reliability, trust, operability of a decision
system under disruption), not on OR/4R theory (that is Paper A). It fits *Systems*' remit
(intelligent planning, complex social systems, decision systems). A humanitarian case is
defensible **if** framed as acute resource-economics under shock — flagged for a one-line GE
check (§5).

Alternative **B2** (resilience of the AI system itself) is viable but narrower; B1 gives a
cleaner economic-turbulence hook.

## 2. Research questions (proposed lock)

- **RQ1 (primary).** When the decision system's inputs are degraded by turbulence
  (noisy / missing / biased data), how stable are its allocation decisions — and does
  fuzzy expert-knowledge encoding degrade *more gracefully* than a crisp optimisation
  baseline?
- **RQ2.** Is the system's decision **reproducible and auditable** across hardware, seeds,
  and software environments, and why does that matter for trust in AI-assisted decisions
  under turbulence?
- **RQ3.** What is the system's **operability envelope** (decision latency, throughput,
  resource use) at decision-relevant sizes — i.e., is it deployable under pressure?

## 3. Distinguishing analyses (NEW — none of these are in Paper A)

- **Input-turbulence stress tests.** Perturb the FIS inputs and demand data (noise levels,
  missingness, bias) and measure decision stability vs a crisp baseline.
  *Distinct from Paper A H3, which perturbed elicitation **weights**; here we perturb the
  **input data** — a genuinely different axis.*
- **Cross-environment reproducibility benchmark.** REP signature across OS / Python / seed;
  ties the DEAP→pymoo hypervolume-fallback work to an operational-reliability story.
- **Operability envelope.** Latency / throughput / memory at sizes; quantify "runs on a laptop
  under pressure."
- **Auditability / governance.** What the system exposes for audit (run manifests, front
  signatures, dependency audit, SECURITY policy) as digital-system trustworthiness.
- *(Optional, pending §5)* a second, synthetic **economically-framed** allocation instance.

## 4. Distinctness statement vs Paper A (anti-salami / dual-submission guard)

| | Paper A (Applied Sciences) | Paper B (Systems) |
|---|---|---|
| Journal / SI | Applied Sciences · SCR resilience | Systems · digital AI under turbulence |
| Object | the model/theory (4R, many-objective) | the **system** (reliability, trust, operability) |
| Core | 4-objective, NSGA-III | **3-objective**, NSGA-II vs NRGA |
| Headline | Robustness/Rapidity visibility gain; NSGA-III speed | graceful degradation under input turbulence; reproducibility/auditability |
| Experiments | Pareto/HV/elicitation-weight sensitivity | input-turbulence, reproducibility-across-env, operability |

Shared: the tool + a minimal model recap (cross-cited to Paper A, not reproduced). Different
RQ, methods emphasis, results, and framing. Keep this statement in our records for any
same-group scrutiny.

## 5. Decisions — RESOLVED (2026-07-04)

1. Framing: **B1 confirmed.**
2. Economic instance: **relief only** — frame post-disaster relief allocation as the acute
   economic-shock case; do **not** add a separate synthetic economic instance (Daniel).
3. Scope fit: **humanitarian primary case is acceptable** (Daniel, as SI invitee). No separate
   GE scope-check email; instead carry a one-sentence scope-fit note in Paper B's cover letter.
4. Corresponding author: **Vladimir Stantchev** (same as Paper A).
5. Article (not Review).

**Belinda Zhao APC-discount enquiry:** email sent 2026-06-30; **no response yet** (pending).

## 6. Next steps once §5 clears

1. Draft `explore/problem-statement.md` + `explore/hypothesis-rq.md` (lock RQ1–RQ3).
2. Design the input-turbulence + reproducibility + operability experiments; decide what new
   code the tool needs (a turbulence/degradation harness distinct from the existing
   sensitivity module).
3. Scaffold `author/` in the MDPI *Systems* template (reuse the `Definitions/` mechanics from
   Paper A; journal option `systems`).
4. Run experiments → figures → draft → internal adversarial review → submit ≤ mid-Nov.
