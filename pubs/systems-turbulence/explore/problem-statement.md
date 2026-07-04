# Paper B — problem statement

## The problem

High economic turbulence turns resource allocation into a decision made under two
simultaneous pressures: **scarcity** (a shock has pushed demand past supply) and **degraded
information** (the inputs a decision system relies on are noisy, incomplete, or biased in the
middle of the shock). Digital AI decision systems are increasingly used to make these calls,
but two properties that determine whether such a system can be *trusted* in turbulence are
rarely evaluated:

1. **Graceful degradation under input turbulence** — does decision quality fall off slowly and
   predictably as inputs degrade, or does it collapse?
2. **Reproducibility and auditability** — can the same inputs be shown to yield the same
   decision across environments, and can the decision be audited after the fact?

Post-disaster relief allocation is the **acute economic-shock instance** of this problem:
scarce relief capacity must be directed to affected people while the situational data (road
status, centre capacity, group needs) is itself disrupted. It is the sharpest available stress
test for a digital allocation system under turbulence.

## Why a fuzzy multi-objective system

Crisp optimisation assumes precise inputs and a single objective. Under turbulence both
assumptions break: inputs are imprecise, and the decision balances competing aims (fairness,
transport feasibility, centre balance). A **Mamdani fuzzy-inference front-end** encodes expert
judgement as linguistic rules that tolerate imprecise inputs, and a **multi-objective
evolutionary optimiser** (NSGA-II / NRGA) exposes the trade-offs rather than hiding them in a
weighted sum. We expected this design to *degrade gracefully* under input turbulence — but a
pilot refuted that (see `pilot-findings.md`): because the optimiser chases objectives built
from the degraded inputs, the fuzzy-MOEA is in fact markedly *more input-fragile* than a crisp
baseline, re-routing most of the allocation under modest noise. That fragility, not graceful
degradation, is the finding — and it is exactly what makes the reproducibility and audit
guardrails below necessary.

## The gap

The tool exists and is validated for *quality* (Paper A: Pareto structure, 4R theory,
elicitation-weight sensitivity). What has **not** been established — and what this paper
contributes — is the system's behaviour as a *deployable digital decision system under
turbulence*:

- no evaluation of decision stability under **input-data** degradation (distinct from
  Paper A's elicitation-*weight* perturbation);
- no cross-environment **reproducibility** evidence framed as an operational-trust property;
- no **operability** envelope (latency / throughput / footprint) establishing deployability
  under pressure.

## Contribution (systems, not theory)

An auditable, reproducible digital AI decision system for scarce-resource allocation under
turbulence, evaluated on the acute relief case for: **decision fragility under input
turbulence** (fuzzy-MOEA vs a crisp baseline), cross-environment reproducibility, and
operability. The contribution is the **system and its trust/operability properties** — and the
cautionary result that MOEA decision quality is input-fragile, which is why reproducibility and
auditability are load-bearing rather than optional — cross-citing Paper A for the model.

## Boundaries

- Primary (and only) case: post-disaster relief allocation, framed as acute economic-shock
  allocation. No separate synthetic economic instance (decision §5.2).
- Core model: the **3-objective** formulation (FLPP / TFL / CABL), NSGA-II vs NRGA — kept
  distinct from Paper A's 4-objective / NSGA-III core.
- Out of scope: the 4R resilience theory and the many-objective split (that is Paper A).

See `hypothesis-rq.md` for the testable form, and `scoping.md` for framing and distinctness.
