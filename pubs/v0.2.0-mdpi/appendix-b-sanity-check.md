# Appendix B sanity check — body claims vs computed p-values

Internal QA, 2026-06-28. Cross-checks the statistical claims in §5.4
(`sec:perf-stats`) and the abstract against the Mann–Whitney U p-values now in
Appendix B (`tab:appb-4obj`, `tab:appb-3obj`), computed from
`experiments/results/h1_h2_h4/manifest.csv`. Not part of the manuscript.

## Confirmed — body and Appendix B agree

| Body claim (§5.4 / abstract) | Appendix B (4-obj) | Verdict |
|---|---|---|
| NSGA-II and NSGA-III achieve identical NNS (100/100) | NNS NSGA-II vs NSGA-III = "---" (degenerate, both constant at 100) | ✅ |
| Significantly different HV, NSGA-II ahead | HV NSGA-II vs NSGA-III: small 0.027\*, medium 0.001\*, large <0.001\* | ✅ (note small is only just significant) |
| Significantly different CPU, NSGA-III ahead | CPU all pairs <0.001\* at every size | ✅ |
| NRGA has the lowest HV at every size | HV NSGA-II vs NRGA and NSGA-III vs NRGA all <0.001\* | ✅ |

## Discrepancies to reconcile (wording, not science)

### 1. "within-noise MID and SM" (§5.4) contradicts Appendix B
§5.4 says NSGA-II and NSGA-III have "within-noise MID and SM." But Appendix B
shows **MID and SM are significantly different (p<0.001) at every size** for
NSGA-II vs NSGA-III. The means coincide to one decimal (MID 77.7 vs 77.7; SM
0.81 vs 0.81), but the rank test detects a systematic, consistently-ordered
difference under low variance — a genuine structural difference in front
spacing, not noise.
**Fix:** replace "within-noise MID and SM" with something accurate, e.g.
"MID and SM differences that are statistically detectable but practically
negligible (means equal to one decimal place)." Do **not** leave "within-noise"
— it's directly refuted by our own appendix.

### 2. Abstract "statistically equivalent Pareto-front quality" vs significant HV
The abstract says NSGA-III is faster "with statistically equivalent
Pareto-front quality." Under the **fixed** HV reference, HV is *significantly*
different (NSGA-II ahead) — Appendix B and §5.4. The "equivalent" reading only
holds under the instance-aware reference (§9 reference-point-sensitivity
paragraph: "HV-equivalent in any practitioner-relevant sense").
**Fix:** soften the abstract to "comparable Pareto-front quality (HV ranking
depends on the reference-point scheme; see Section 9)" — or make explicit it is
the practitioner/instance-aware reading. As written it reads as contradicting
§5.4.

### 3. "ranking is stable ... across both formulations" overstated for 3-obj HV
§5.4 closes with "the ranking is stable across all three problem sizes and
across both formulations." In the **3-obj** formulation, NSGA-II vs NSGA-III HV
is **not** significant on small (p=0.348) or medium (p=0.284); only large is
(0.018\*). So the NSGA-II HV lead over NSGA-III is not stable across sizes in
the 3-obj case.
**Fix:** qualify — e.g. "the CPU ranking is stable across all sizes and both
formulations; the HV lead of NSGA-II over NSGA-III is significant throughout
the four-objective formulation but only at the large size in the three-objective
baseline."

## Minor / no action

- **NRGA NNS "loses 0–1%":** NNS NSGA-II vs NRGA is *not* significant
  (p=0.334, 4-obj). The "0–1%" is a descriptive statement about the mean, which
  is fine — but it is a practical, not a statistically significant, effect.
  No change needed; just don't claim NNS significance for NRGA anywhere.

## Bottom line
The four-objective headline results (HV: NSGA-II ahead; CPU: NSGA-III ahead;
NNS: tied) are fully corroborated. Three sentences need precision edits (#1
mandatory, #2–#3 recommended) before co-authors see the appendix, since each is
checkable against the new tables. Fold these in alongside the ATRes-review edits.
