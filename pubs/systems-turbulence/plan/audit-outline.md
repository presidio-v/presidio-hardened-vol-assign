# Paper B — audit framing: outline (2026-10-05)

Supersedes `design-brief.md` / the "decision fragility" framing. Evidence: `explore/rq1-floor-results.md`.

**Working title:** *Auditing a Digital AI Allocation System under Input Turbulence: When Exact
Optimisation Should Replace the Evolutionary Search*

**Venue:** MDPI *Systems*, SI "Using Digital AI Systems as a Response to High Economic
Turbulence and Uncertainty". Article, 18–24 pp, ~60–70 refs.

**Thesis.** Before a digital AI decision system is trusted under turbulence, audit whether its
AI is needed and whether its decision is identifiable. Applied to an open, published fuzzy-MOEA
relief-allocation system, a five-step audit shows: the model is separable and solved exactly by
a greedy rule that dominates the deployed optimiser; the optimiser's committed decision changes
for most people between seeds; removing solver noise exposes the true, smooth input-sensitivity;
and a minimal semantic repair is still solved exactly in seconds up to ~10^3 people.

## Research questions
- **RQ1 (Necessity).** Does the deployed optimisation problem require the evolutionary AI?
  Separability; exact greedy vs MOEA fronts (dominance count, HV share); budget curve.
- **RQ2 (Identifiability).** Is the AI system's committed decision identifiable? Seed floor
  (churn, Jaccard, quality spread) vs budget; consensus-of-k (appendix).
- **RQ3 (Turbulence).** How do decisions degrade under input turbulence once solver noise is
  removed? Exact decider dose-response per field/mode vs crisp rule; MOEA floor-adjusted harm.
- **RQ4 (Repair & envelope).** Does a minimal semantic repair (load-coupled CAIL + capacity)
  change the conclusion, and where does exact optimisation stop being practical? κ sweep,
  hard capacity, scaling vs time-matched MOEA.

## Sections
1. **Introduction** — turbulence = scarcity + degraded data; digital AI decision systems
   adopted on the assumption that the AI is needed and its output is a decision; audit as
   method; contributions (protocol; four findings; open artefact; correction of own prior work).
2. **Background** (~2.5 pp) — (a) systems view of decision aids under deep uncertainty
   (requisite variety, robustness, RDM); (b) MOO, solution selection, robustness; exact vs
   heuristic, metaheuristic critique; (c) fuzzy inference in DSS; (d) humanitarian OR;
   (e) data quality, AI trust, algorithmic auditing, EU AI Act Art. 12/15, reproducibility.
   Gap: audits of *deployed* optimisation-based AI decision systems that test necessity and
   identifiability, not only accuracy/fairness.
3. **Relief allocation as a turbulence case** — the system (cite [A] for model/tool);
   **Table: relief-shock ↔ economic-shock mapping** (demand > capacity ↔ demand shock;
   infrastructure damage ↔ logistics disruption; resource time ↔ liquidity runway; occupancy ↔
   capacity utilisation; road/hazard ↔ route/border/tariff status; missingness ↔ information lag).
4. **The audit protocol** — Fig. 1 flow: (S1) structural check (separability) → (S2) exact
   reference & dominance → (S3) seed floor & budget curve → (S4) input turbulence on the exact
   decider → (S5) repair & deployment envelope. Metrics, statistics (cluster bootstrap over
   seeds, paired loss, Spearman trend + Holm), turbulence model (noise/bias/missingness/flip,
   levels, realisations, clipping), instances, software & reproducibility (signatures, env
   fingerprint, manifests, Zenodo).
5. **Results** RQ1–RQ4 (figures below).
6. **Discussion** — for digital AI systems under turbulence: (i) test necessity before
   deploying a stochastic optimiser; (ii) non-identifiability is a trust failure that
   reproducibility alone does not fix (seed pinning makes it repeatable, not meaningful);
   (iii) fragility should be measured on a noise-free decider; (iv) rule-base dynamic range
   (FIS3 ~20 points) bounds what a fuzzy objective can express. Economic transfer: mechanism
   level only. Implications for practitioners/regulators. Correction note on [A].
7. **Limitations** — synthetic instances, one system, relief-only, analogy not data; MIP
   formulation scales as m·n·L; FIS rule bases taken as published.
8. **Conclusions.**
Back matter: CRediT, AI-use statement, Data Availability (Zenodo DOI), COI.
Appendices: A floor tables & consensus-of-k; B full turbulence tables; C MIP formulation &
brute-force verification; D cross-environment reproducibility table.

## Figures / tables
- Fig 1 audit protocol (diagram).
- Fig 2 MOEA fronts vs exact supported front (2-D projections, small) + dominance counts.
- Fig 3 budget curve: clean Q and seed-floor churn vs generations (small & large), exact Q line.
- Fig 4 turbulence dose-response: quality loss & churn vs level, exact vs crisp (+ MOEA above floor).
- Fig 5 envelope: MIP time/gap and quality vs time-matched MOEA across sizes.
- Tab 1 relief↔economic mapping; Tab 2 RQ1 numbers; Tab 3 κ sweep (soft/hard); Tab 4 per-cell
  turbulence summary.

## Open items
- Cross-env reproducibility evidence lost → re-run CI matrix (needs the branch pushed — ask user).
- RQ3 large exact turbulence (running); budget curve large (running); scaling (queued).
- Consensus-of-k from stored fronts (offline).
- Zenodo release of the paper branch (user-gated: GitHub release fires Zenodo+PyPI).
- Paper A correction text (after B is drafted).
