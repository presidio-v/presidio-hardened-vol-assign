# Paper B — pilot findings (why RQ1 was reframed)

Honest trail: we hypothesised that fuzzy expert-knowledge encoding would make the
decision system *degrade gracefully* under input turbulence (original H-B1). A pilot
refuted it, and two diagnostics confirmed the refutation is genuine. RQ1 is reframed
accordingly; the full matrix is the confirmatory test.

## What we ran

Small instance, `infrastructure_damage_level` + `center_occupancy_rate` NOISE, decisions
scored on the clean ground truth. Fuzzy-MOEA (NSGA-II, 3-obj) vs the crisp greedy baseline.

1. **Pilot** (pop 30 / gen 30): fuzzy drift > crisp, but non-monotonic — inconclusive.
2. **Full-budget** (pop 100 / gen 150): the effect is clean and monotonic.
3. **Decision-rule diagnostic** (equal-weight vs fixed-nearest extraction): tests whether the
   effect is genuine or an artifact of the canonical-decision rule.

## Results (pop 100 / gen 150, small)

**Allocation churn** (fraction of directed people re-routed under noise):

| field | crisp (0.2 / 0.4) | fuzzy (0.2 / 0.4) |
|---|---|---|
| IDL | 0.10 / 0.20 | **0.84 / 0.85** |
| centre-occupancy | 0.21 / 0.34 | **0.84 / 0.84** |

**Realised-objective drift:** fuzzy ≈ 3–5 vs crisp ≈ 1–1.3 (IDL); fuzzy ≈ crisp at high
centre-occupancy noise. Level 0.0 → exactly 0 for both (identity sanity check).

**Decision-rule diagnostic (IDL churn):** equal-weight 0.840 / 0.853 vs fixed-nearest
0.848 / 0.851 — **identical**. The churn is not an extraction artifact; it survives a
composition-independent decision rule.

## Conclusion

- Original H-B1 (graceful degradation) is **refuted**, robustly and monotonically.
- The genuine finding: the fuzzy-MOEA decision system is **markedly more input-fragile**
  than the crisp baseline — it re-routes ~85% of directed people under modest input noise
  (vs 10–34% crisp) onto genuinely worse decisions (higher realised drift). This is a
  property of the system, not of the metric or the decision rule.
- Mechanism: the MOEA optimises objectives that depend on the noisy inputs, so input noise
  reshapes the objective landscape it chases; the crisp heuristic uses those inputs only
  weakly.

## Why this is the stronger paper

The finding *unifies* RQ1 and RQ2: because MOEA decision quality is input-fragile, you cannot
trust a single run — **reproducibility and auditability (RQ2) become the load-bearing
guardrails** for deploying such a system under turbulence. RQ1 now motivates RQ2 instead of
sitting beside it.

These are exploratory pilot numbers (small, few realisations/reps). The reframed RQ1 (below)
is confirmed on the full matrix with the Wilcoxon-on-slopes stats.
