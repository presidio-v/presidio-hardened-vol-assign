# Paper B — design brief

**Venue:** *Systems* (MDPI, SSCI Q1, IF 3.8), SI **"Using Digital AI Systems as a
Response to High Economic Turbulence and Uncertainty."**
GEs Jong-min Kim & Rob Marjerison; AE Belinda Zhao. Deadline **30 Nov 2026**
(~20-day first decision). No confirmed APC waiver (discount enquiry pending with Zhao).
Origin: Daniel forwarded the invite — "take Payman's paper and give it a treatment related
to [the SI theme] so we can submit it asap."

**One-line thesis:** an *auditable, reproducible digital decision system* that keeps
resource-allocation decisions reliable when its inputs are degraded by turbulence and
uncertainty — the system, not the optimisation theory, is the contribution.

---

## 1. The fit problem, stated honestly

The SI is about **digital AI systems under *economic* turbulence and uncertainty**. Our
substrate is **humanitarian disaster allocation** — *operational* turbulence, not economic.
Submitting the SCR/4R paper here would be a poor fit and risks a desk-reject. Paper B must
therefore be **reframed around the SI's two load-bearing words — *systems* and *uncertainty*
— not around disaster relief.** Two honest bridges (pick one with Daniel):

- **(B1) Systems-of-decision under uncertainty (recommended).** Position the tool as a
  general *digital AI decision system* for constrained resource allocation under high
  uncertainty, with disaster relief as one acute *case study* and a second, economically
  framed instance (e.g., allocation of scarce budget/capacity across demand points under
  volatile, turbulent conditions). The contribution is the **system and its behaviour under
  uncertainty**, evaluated on both cases.
- **(B2) Resilience-of-the-AI-system itself.** Argue that "response to turbulence and
  uncertainty" includes the *operational resilience of the decision system* — reproducibility,
  auditability, robustness to input degradation, and graceful behaviour when data is missing
  or noisy. Disaster allocation supplies the stress test.

> Action: confirm the bridge with Daniel and, ideally, a one-line scope check with the GEs
> (Kim/Marjerison) before investing in experiments.

## 2. Distinct contribution vs Paper A (no salami-slicing)

| | **Paper A** (Applied Sciences SCR) | **Paper B** (Systems) |
|---|---|---|
| Object | the **model/theory** — 4R reframing, 4 objectives | the **system** — the digital AI decision tool as an artifact |
| Core model | 4-objective (4R split, NSGA-III) | **3-objective** (FLPP/TFL/CABL, NSGA-II vs NRGA) — deliberately the simpler core, so the two papers do not share their central result |
| Headline result | Robustness/Rapidity visibility gain; NSGA-III 32% faster | system behaviour under **input uncertainty / degradation**; reproducibility & auditability as operational properties |
| Evaluation | Pareto/HV/sensitivity to elicitation noise | stress tests: data degradation, missing inputs, environment/seed reproducibility, audit trail, decision latency/throughput |
| Lens | supply-chain-resilience theory | systems theory: a system under uncertainty, feedback, governance |

Cross-cite Paper A for the model; Paper B contributes the *system + uncertainty behaviour*.
No shared text; different RQ, methods emphasis, and headline results.

## 3. Candidate research questions (pick 1 primary + 1–2 secondary)

- **RQ1 (primary).** How does the decision system's output degrade as its inputs are
  perturbed by turbulence (noisy/missing/biased data), and does the fuzzy-MOEA design
  degrade *gracefully* relative to a crisp/optimisation baseline?
- **RQ2.** To what extent is the system's decision *reproducible and auditable* across
  hardware, seeds, and software environments — and why does that matter for trust in
  AI-assisted decisions under turbulence?
- **RQ3.** What is the operational envelope (decision latency, throughput, resource use)
  of the system at decision-relevant problem sizes, i.e., is it deployable under pressure?

## 4. New analyses Paper B needs (NOT in Paper A)

These must be genuinely new so the paper stands alone:
- **Input-degradation / uncertainty stress tests** — sweep noise/missingness on the FIS
  inputs and the demand data; measure decision stability vs a crisp baseline. (The repo's
  sensitivity machinery is a starting point but must be re-aimed at *input* uncertainty,
  not elicitation-weight noise, to stay distinct from Paper A's H3.)
- **Cross-environment reproducibility benchmark** — REP signature across OS/Python/seed;
  tie to the DEAP→pymoo fallback work as an operational-reliability story.
- **Auditability / governance** — what the tool exposes for audit (manifests, signatures,
  dependency audit, SECURITY policy) framed as digital-system trustworthiness.
- **Operational envelope** — latency/throughput at sizes; "deployable on a laptop" claim
  quantified.
- (Optional) **Second, economically framed case instance** if pursuing bridge B1.

## 5. Tentative structure

1. Intro — decision systems under turbulence/uncertainty; the trust/operability gap.
2. Background — digital AI decision systems; fuzzy reasoning under uncertainty; systems lens.
3. The system — architecture, the 3-objective FIS-MOEA engine, the tool (CLI, hardening,
   reproducibility design).
4. Method — uncertainty/degradation protocol, reproducibility protocol, operational metrics.
5. Results — RQ1–RQ3.
6. Discussion — implications for AI-assisted decision-making under turbulence; limits.
7. Conclusion.

## 6. Logistics

- **Authors:** Rabiei, Arias-Aranda, Stantchev (corresponding TBD — possibly Daniel, as
  the SI invitee of record).
- **Article type:** original Article (review also welcome per the invite, but Article fits).
- **Reuse vs new:** reuse the substrate/tool; ~all framing, RQs, experiments, and results
  are new. Keep a written distinctness statement for our records and for any same-group
  dual-submission scrutiny.
- **Timeline:** design + GE scope check (Aug) → experiments (Sep) → write + internal
  adversarial review (Oct) → submit ≤ mid-Nov (margin before 30 Nov + ~20-day review).

## 7. Open questions for Daniel / the GEs

1. Which bridge — B1 (general decision system + economic case) or B2 (resilience of the
   AI system itself)?
2. Is a humanitarian case study acceptable in this SI, or must the primary instance be
   economic? (worth a one-line GE check).
3. Corresponding author for Systems?
4. APC discount outcome (pending Belinda Zhao).

See [[two_paper_strategy]] and `pubs/v0.2.0-mdpi/plan/two-paper-strategy.md`.
