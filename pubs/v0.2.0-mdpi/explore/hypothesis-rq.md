---
provisional: true
phase: I-C
project: presidio-hardened-vol-assign
target-venue: Applied Sciences (MDPI), SI "Innovations in Supply Chain Resilience"
authors: Rabiei, Arias-Aranda, Stantchev
---

## Research Questions

| ID  | Research Question | Linked H |
|-----|-------------------|----------|
| RQ1 | Does the 4-objective formulation that splits ATRes's Transportation Infeasibility Level (TIL) into separate Robustness (TRD) and Rapidity (RPD) components expose Pareto trade-offs that the 3-objective formulation cannot represent? | H1 |
| RQ2 | Does NSGA-III outperform NSGA-II and NRGA on the 4-objective humanitarian allocation problem with respect to convergence (MID, HV) and diversity (NNS, SM)? | H2a, H2b, H2c |
| RQ3 | How sensitive are the FIS-derived objective values and the resulting Pareto fronts to (a) FIS rule-base perturbation and (b) parametric weight perturbation? | H3a, H3b |
| RQ4 | Does the empirical finding from ATRes — NSGA-II faster than NRGA on large problems — replicate under the 4-objective formulation? | H4 |

## Hypotheses

| ID  | Hypothesis | H0 | Confirmation Criterion | Refutation Criterion |
|-----|------------|----|------------------------|----------------------|
| H1  | The 4-objective Pareto front exposes Robustness/Rapidity trade-offs absent from the 3-objective front: there exist Pareto-optimal solutions in the 4-obj front with low TRD and high RPD (and vice versa) that collapse to dominated or absent points when projected to the 3-obj space. | The 4-obj and 3-obj Pareto fronts are information-equivalent: Spearman rank correlation between TRD and RPD across the 4-obj front is ≥ 0.9 in all problem instances. | At least 20% of solutions on the 4-obj Pareto front have |Spearman ρ(TRD, RPD)| < 0.5 *and* are dominated when projected to 3-obj space (TIL = α·TRD + (1−α)·RPD using ATRes's RWS weighting). | Spearman ρ ≥ 0.9 across all instances and < 5% of 4-obj solutions are non-recoverable in 3-obj space. |
| H2a | NSGA-III achieves significantly higher mean hypervolume than NSGA-II and NRGA on at least the large-size problem (15/450/150 or 20/600/200, TBD in I-D). | No significant difference in HV across the three algorithms at any problem size. | Mann-Whitney U test, p < 0.05 favoring NSGA-III at the largest problem size, and effect size r > 0.3. | p ≥ 0.05 at all problem sizes, or NSGA-III mean HV lower than NSGA-II at the largest size. |
| H2b | NSGA-III achieves diversity (NNS, SM) comparable to or better than NSGA-II and NRGA. | NSGA-III is significantly worse on NNS or SM. | t-test (NNS, SM under normality) and Mann-Whitney U (otherwise): NSGA-III not significantly worse (p ≥ 0.05), and at least one of {NNS, SM} significantly better at one problem size. | NSGA-III significantly worse on NNS or SM at every problem size. |
| H2c | NSGA-III's CPU-time scaling is no worse than NSGA-II's on the largest problem size. | NSGA-III is significantly slower than NSGA-II at the largest size. | Mann-Whitney U on CPU time, p ≥ 0.05 *or* NSGA-III faster at the largest problem size. | NSGA-III significantly slower (p < 0.05, mean rank higher) at the largest problem size. |
| H3a | The 4-objective Pareto front is robust to single-rule deletion in any of the four FIS rule-bases (FIS₁, FIS₂ₐ, FIS₂ᵦ, FIS₃): median ΔHV ≤ 5% across all single-rule-deletion variants. | Median ΔHV > 5% for at least one FIS rule-base. | Across all single-rule deletions on the medium problem size, 30 reps each, median ΔHV ≤ 5% in every FIS. | Median ΔHV > 5% for any FIS, *or* worst-case ΔHV > 20% in any FIS. |
| H3b | The 4-objective Pareto front is robust to ±20% Latin Hypercube perturbation of the parametric weights (WAS, WDS, WIL, WLS, WRC, WPH): coefficient of variation of the four objective means ≤ 10%. | CV > 10% for at least one objective. | LHS, n = 100 samples on the medium problem size, 30 reps each: CV ≤ 10% for all four mean objectives. | CV > 10% for any objective, *or* CV > 25% for any objective (signaling fragile elicitation). |
| H4  | NSGA-II is significantly faster than NRGA on the large problem size with the 4-objective formulation (replication of ATRes finding). | No significant difference, or NRGA faster. | Mann-Whitney U on CPU time, p < 0.05, NSGA-II mean rank lower. | p ≥ 0.05, or NRGA mean rank lower. |

## Internal Consistency Check

The hypothesis set forms a coherent narrative:

- **H1** justifies the 4-objective formulation. If H1 fails (3-obj and 4-obj fronts are equivalent), the methodological contribution collapses to "we used NSGA-III" and the paper should be reframed; this is the highest-stakes hypothesis.
- **H2a–c** justify the NSGA-III addition. If they all fail, NSGA-III adds nothing and the paper reverts to NSGA-II + NRGA on the 4-objective formulation, weakening the algorithmic contribution.
- **H3a–b** establish that the recommendations are not artifacts of rule-base or weight choices. If they fail, the FIS-MOEA approach itself is in question; this is a load-bearing finding for the entire research line, not just this paper.
- **H4** anchors the paper to ATRes's empirical core. Replication is independent of H1–H3 and provides a sanity check that the underlying model behaves consistently after the split.

H1 and H2 are independent: H1 tests the formulation, H2 tests the algorithm. H3 tests neither but tests the model's sensitivity, which any reviewer will ask about.

## What confirmation looks like (the paper as accepted)

- **§3.5 (Resilience mapping)** establishes the 4R↔objective mapping as the
  theoretical contribution.
- **§5 (Algorithm comparison)** reports H2a/b/c with statistical tests on three
  problem sizes and H4 as a replication finding.
- **§6 (Robustness/Rapidity trade-off analysis)** reports H1 with the
  rank-correlation analysis and projection-dominance counts on the Pareto
  fronts.
- **§7 (Sensitivity analysis)** reports H3a/b with rule-deletion and weight-LHS
  studies.

## What refutation looks like (and what it changes)

- **H1 refuted:** the 4-objective formulation is theoretically pleasing but
  empirically redundant. We pivot to a "framing paper" — keep the 4R mapping
  but argue from theory rather than from new empirics. Riskier; not preferred.
- **H2 refuted:** NSGA-III adds nothing here. Drop NSGA-III claims; rewrite
  §5 as a many-objective negative result (still publishable but different
  framing).
- **H3 refuted:** the model is fragile. Add Phase III work on robust FIS
  elicitation; possibly delay submission. Highest-impact failure.
- **H4 refuted:** weakens the link to ATRes; not fatal — note as a finding,
  discuss why the split changed the comparison.
