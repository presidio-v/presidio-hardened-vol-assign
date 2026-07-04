# Paper B — research questions and hypotheses

Three RQs, each with a testable hypothesis, a baseline/comparator, and metrics. All are
**new** relative to Paper A (distinctness table in `scoping.md` §4).

## RQ1 — graceful degradation under input turbulence (primary)

**Question.** As the system's inputs degrade (noise / missingness / bias), how stable are its
allocation decisions, and does the fuzzy-MOEA design degrade more gracefully than a crisp
optimisation baseline?

**H-B1.** Under increasing input degradation, the fuzzy-MOEA system's decision quality declines
*more slowly* (smaller degradation slope) and *more predictably* (lower variance) than a crisp
weighted-sum / greedy baseline on the same inputs.

- **Turbulence knobs:** additive noise on FIS inputs (σ sweep), random missingness on demand
  fields (rate sweep), and systematic bias (shift). One-factor-at-a-time + a combined stress
  level.
- **Baseline:** a crisp allocator on the same data — deterministic weighted-sum greedy (and, if
  cheap, an exact per-scalarisation comparator).
- **Decision-stability metrics:** change in the realised objective vector vs the clean-input
  decision; allocation churn (fraction of people re-routed); rank stability of centre loads.
  Report degradation *slope* and *variance* vs turbulence level, fuzzy vs crisp.
- **Distinct from Paper A H3:** that perturbed elicitation **weights**; this perturbs the
  **input data**. Different mechanism, different claim.

## RQ2 — reproducibility and auditability as trust properties

**Question.** Is the system's decision reproducible across hardware, seeds, and software
environments, and what does it expose for audit?

**H-B2.** For a fixed seed and inputs, the system produces bit-for-bit identical allocation
fronts across environments (REP = 1.0), and its run artefacts (manifests, front signatures,
dependency audit) constitute a sufficient audit trail.

- **Reproducibility benchmark:** REP signature across ≥2 OS × ≥2 Python × seeds; report REP and
  any divergence source. Ties in the DEAP→pymoo hypervolume-fallback fix as an
  operational-reliability story (a real environment-portability failure that was closed).
- **Auditability:** enumerate what the system exposes (run manifest, SHA-256 front signature,
  `pip-audit` clean, SECURITY policy) and map each to a decision-trust property.
- **Framing:** reproducibility/auditability as *first-class* properties of a trustworthy digital
  decision system under turbulence — not an afterthought.

## RQ3 — operability envelope

**Question.** What is the system's decision latency, throughput, and resource footprint at
decision-relevant sizes — is it deployable under pressure?

**H-B3.** The system returns a full Pareto set within operationally acceptable time and memory
on commodity hardware across the small/large instances, i.e., it is field-deployable without
specialised infrastructure.

- **Metrics:** wall-clock latency and peak memory per (size, algorithm); throughput
  (decisions/hour); scaling with n_people / n_centers.
- **Comparator:** NSGA-II vs NRGA operability at equal quality (distinct from Paper A's
  NSGA-III speed result, which is not used here).

## Success criteria

- RQ1: a statistically supported gentler/steadier degradation curve for fuzzy-MOEA vs crisp
  under ≥1 turbulence mode (Wilcoxon on per-instance degradation slopes).
- RQ2: REP = 1.0 across the environment matrix, with a documented audit-trail mapping.
- RQ3: reported latency/memory envelope with a defensible "deployable on a laptop" claim.

## New code required (distinct from the existing sensitivity module)

- An **input-turbulence harness**: perturb *inputs/data* (not weights), re-solve, record
  decision-stability metrics vs the clean baseline decision. (The current `sensitivity` module
  perturbs weights/rules — reuse its plumbing, not its perturbation target.)
- An **operability profiler**: latency/memory capture around `solve()`.
- Reuse as-is: REP signatures (`repro`), the crisp baselines, `pip-audit`/CI for the audit story.

See `problem-statement.md` and `scoping.md`.
