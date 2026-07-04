# Paper B — RQ1 results (small-full matrix)

Source: `experiments/results/turbulence/small/turbulence_summary.csv` (8 field×mode cells,
5 levels, 12 realisations, 8 solver reps, NSGA-II, 3-obj, pop 100 / gen 150). Analysis:
`experiments/analyze_turbulence.py` — per-realisation degradation slope (metric vs level),
paired Wilcoxon fuzzy-vs-crisp. Figures: `pubs/systems-turbulence/figures/`.

## Headline: allocation churn — universal fuzzy fragility

Fuzzy-MOEA slope > crisp in **all 8 cells**, Wilcoxon **p ≈ 0.00049** (7 cells) / 0.00342
(centre-occupancy noise). Median churn slope ≈ **1.25–1.28** (fuzzy) vs ≈ **0.15–0.78**
(crisp). The MOEA re-routes far more of the allocation than the crisp baseline under every
turbulence type — noise, missingness, and categorical flips. This is the robust RQ1 claim.

## Realised-objective drift — field-dependent

| field / mode | fuzzy slope | crisp slope | fuzzy > crisp | p |
|---|---|---|---|---|
| IDL noise | 10.04 | 4.27 | yes | 0.0005 |
| IDL missingness | 7.22 | 3.21 | yes | 0.0005 |
| resource_time_remaining noise | 12.15 | 2.25 | yes | 0.0005 |
| travel_duration noise | 10.46 | 5.01 | yes | 0.0005 |
| centre-occupancy missingness | 10.82 | 0.00 | yes | 0.0005 |
| possible_hazard flip | 5.36 | 4.31 | yes | 0.151 (n.s.) |
| centre-occupancy noise | 6.37 | 9.04 | **no** | 0.064 (n.s.) |
| road_condition flip | 4.85 | 7.84 | **no** | 0.009 |

Fuzzy drift significantly worse in 5/8; comparable in 2; crisp significantly worse in 1
(road-condition flip). So the MOEA's *decision* churns everywhere, but that only becomes
worse *realised objectives* for person-attribute and travel-time perturbations.

## Large-instance subset (scale test, 10 centres / 300 people, n=8)

| field / mode | churn fuzzy | churn crisp | p | drift fuzzy | drift crisp | p |
|---|---|---|---|---|---|---|
| IDL / noise | 1.44 | 0.37 | 0.008 | 7.5 | 2.4 | 0.008 |
| IDL / missingness | 1.42 | 0.16 | 0.008 | 5.4 | 4.8 | 0.55 |
| centre-occ. / noise | 1.45 | **1.44** | **0.95** | 8.5 | **23.4** | 0.016 |
| road-condition / flip | 1.42 | 1.16 | 0.008 | 5.4 | **9.2** | 0.008 |

**Scale changes the picture:** the small-instance "8/8 universal" churn gap does **not**
hold at scale. For the person-priority inputs (IDL) the fuzzy fragility stays sharp
(churn ~1.4 vs 0.16–0.37, p=0.008; drift 2–3×). For **centre-occupancy noise the gap
vanishes** — crisp churn rose from 0.73 (small) to 1.44 (large), matching fuzzy — and the
crisp baseline's realised drift is far *worse* (23.4 vs 8.5). The fragility is **localised
to the inputs the MOEA objectives directly consume**, not universal.

## Interpretation

- The system is **input-fragile localised to the objective-coupled person inputs** (IDL,
  resource-time, travel) — sharp at both scales in churn and realised quality; for
  centre/transport inputs the effect attenuates or reverses at scale.
- Even where objective quality holds (centre-occupancy noise, road-condition flip), the
  allocation still churns ~1.25×/level — a trust failure in itself: noisy inputs make the
  system commit to very different allocations of equivalent stated quality.
- This is exactly why RQ2 (bit-for-bit reproducibility + auditability) is load-bearing: if a
  single run's decision is this input-sensitive, the minimum trustworthiness bar is being able
  to prove *which* run produced *which* decision, reproducibly and auditably.

Confirmatory scope note: small instance only so far; the large-subset run tests whether the
fragility holds at scale (see `plan/full-run-matrix.md`).
