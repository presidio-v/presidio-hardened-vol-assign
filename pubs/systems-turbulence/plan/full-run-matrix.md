# Paper B — RQ1 full-run matrix

Orchestrator: `experiments/run_turbulence_full.sh` (env-configurable).
Analysis: `experiments/analyze_turbulence.py <results_dir>` → per-cell degradation
slopes + paired Wilcoxon (fuzzy vs crisp) → `turbulence_summary.csv`.

## Matrix (8 field×mode cells)

| field | modes | feeds objective |
|---|---|---|
| infrastructure_damage_level | noise, missingness | ULPP (directly) |
| resource_time_remaining | noise | ULPP |
| center_occupancy_rate | noise, missingness | CAIL |
| travel_duration | noise | CAIL + transport |
| road_condition | flip | transport (TIL) |
| possible_hazard | flip | transport (TIL) |

Levels `0.0, 0.05, 0.1, 0.2, 0.4`; level 0.0 is the identity sanity check. Both the
fuzzy-MOEA (NSGA-II, 3-obj) and the crisp greedy baseline decide on each perturbed
instance; decisions are scored on the clean ground truth.

## Statistical power

The paired Wilcoxon on per-realisation slopes needs **n ≥ 6** realisations to reach
p < 0.05 at all (n=5 caps p at 0.0625 — seen in the diagnostic). Defaults use
**REAL=12**, giving headroom.

## Configurations & rough runtime (≈3 s/solve small, ≈15 s large)

| config | size | reps | real | solves | est. wall |
|---|---|---|---|---|---|
| **small-full** (default) | small | 8 | 12 | ~3,900 | ~3 h |
| large-subset | large | 6 | 8 | ~1,900 (4 cells) | ~8 h |
| robustness (fixed-nearest rule) | small | 8 | 12 | ~3,900 | ~3 h |

Solves per cell = reps (clean) + levels × real × (1 crisp + reps fuzzy).

## Staged plan (recommended)

1. **small-full** (all 8 cells) — the main RQ1 result + figures. ~3 h, background.
2. **large-subset** — 4 representative cells (IDL/COR noise + one missingness + one flip)
   at reduced reps/real, to show the effect holds at scale. ~8 h, background.
3. **robustness** — re-run small-full with `RULE=fixed-nearest` to confirm the fragility is
   decision-rule-independent (the diagnostic already indicates this).

## Launch

```bash
# small-full (default)
bash experiments/run_turbulence_full.sh
# large subset
SIZE=large REPS=6 REAL=8 bash experiments/run_turbulence_full.sh   # (edit CELLS for the subset)
# analyse
.venv/bin/python -m experiments.analyze_turbulence experiments/results/turbulence/small
```

Then figures (degradation curves fuzzy vs crisp per cell; churn bar) via a
`make_turbulence_figures.py` step (to be written), into `pubs/systems-turbulence/figures/`
(git-ignored `*.pdf`, per Paper A convention).
