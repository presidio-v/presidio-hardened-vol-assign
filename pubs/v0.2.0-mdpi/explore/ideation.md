---
provisional: true
phase: I-A
project: presidio-hardened-vol-assign
target-venue: Applied Sciences (MDPI), SI "Innovations in Supply Chain Resilience"
target-window: 2026-06-20 to 2026-07-04 (6–8 weeks from invitation 2026-05-09)
authors: Rabiei, Arias-Aranda, Stantchev
---

## Domain

Humanitarian supply chain resilience at the last-mile interface: directing demand
(affected people) to supply nodes (relief centers) under infrastructure disruption,
capacity constraints, and time pressure. Computational core sits at the
intersection of multi-objective evolutionary optimization, fuzzy expert systems
for qualitative reasoning under uncertainty, and supply chain resilience theory.

## State of the Art (summary)

**Supply chain resilience theory.** Bruneau et al. (2003) established the 4R
framework — Robustness, Redundancy, Resourcefulness, Rapidity — for disaster
resilience. Christopher and Peck (2004) and later Tukamuhabwa et al. (2015)
extended the lens to commercial and humanitarian SCM. Hosseini et al. (2016)
provide a quantitative resilience taxonomy. Behl and Dutta (2019) review
humanitarian SCR specifically.

**Humanitarian last-mile allocation.** Hashim et al. (2021), Zhao et al. (2017),
Gama et al. (2016), Chang et al. (2024), Hansuwa et al. (2021), and Beiki et al.
(2020) treat the problem with classical OR objectives — distance, time, cost.
Soghrati Ghasbeh et al. (2022), Wang et al. (2020), and Abounacer et al. (2014)
introduce equity. Almost none reach for resilience-theoretic vocabulary; the
4R framework is not used to organize objective functions.

**FIS-MOEA hybrids in disaster operations.** Rabiei and Arias-Aranda (2021,
ICTIS) and Rabiei et al. (2023, ESWA 226:120142) introduced the FIS+NSGA-II/NRGA
pattern for vehicle routing and volunteer assignment. The ATRes paper currently
in press (Rabiei, Arias-Aranda, Stantchev 2026) extends it to allocation with
three new qualitative indices (FLPP, TFL, CABL).

**Evolutionary algorithms for many objectives.** Deb et al. (2002) NSGA-II is the
two-/three-objective default; Deb and Jain (2014) NSGA-III is reference-point-based
and designed for ≥4 objectives but is increasingly applied to three. Zheng and
Doerr (2024) IEEE TEC offers fresh runtime analysis. Empirical evidence on
small-budget humanitarian MOEA problems is thin.

## Cross-disciplinary Overlaps

- **Operations research × disaster studies × computational intelligence.** The
  ATRes line of work already lives here. The new overlap is with supply chain
  management theory: applying SCM resilience taxonomies to humanitarian MOEA
  models that have so far stayed in pure OR vocabulary.
- **Software engineering × operations research.** Reproducible artifacts are
  rare in this literature. MDPI's open-data policy plus the existing
  `presidio-hardened-vol-assign` Python tool make this overlap concrete and
  publishable.
- **Robust optimization × fuzzy systems.** Sensitivity of FIS-evaluated
  objectives to rule-base and weight perturbation is unexplored — a methodological
  gap, not just an empirical one.

## White Spots

1. **Resilience-theoretic framing of FIS-MOEA humanitarian models.** No paper in
   the FIS-MOEA disaster line maps its objective functions to a named resilience
   framework. The three ATRes indices (ULPP, TIL, CAIL) cover Resourcefulness,
   Robustness+Rapidity, and Redundancy respectively — but this mapping is
   neither stated nor used. Stating it produces actionable trade-off vocabulary
   for practitioners and a theoretical contribution for the SCR community.

2. **NSGA-III in 3-objective humanitarian MOEAs.** NSGA-III is rarely benchmarked
   against NSGA-II/NRGA on humanitarian three-objective problems with FIS-evaluated
   objectives. The conventional wisdom that NSGA-III pays off only at ≥4 objectives
   is not tested in this domain.

3. **Reproducible artifact.** No open-source reference implementation of the
   FIS-MOEA pattern exists in this line of work. ESWA 2023 and ATRes 2026 use
   MATLAB R2024b without code release. `presidio-hardened-vol-assign` v0.1.0
   already implements the model in Python (scikit-fuzzy + DEAP) with 97%
   coverage; releasing it as the citable artifact for the MDPI paper closes
   the gap and aligns with MDPI's open-data ethos.

4. **Sensitivity of FIS-derived objectives.** No systematic study of (a) rule-base
   perturbation (drop a rule, change a consequent class) or (b) weight
   perturbation (vary WAS/WDS/WIL/WLS, WRC/WPH within plausible ranges) on
   the resulting Pareto fronts. Without it, decision-makers cannot judge how
   fragile the recommendations are to expert-elicitation noise.

## Candidate Ideas (ranked)

| Rank | Title (working) | Novelty | Feasibility (6–8 wk) | Impact / SI fit | Verdict |
|------|-----------------|---------|----------------------|-----------------|---------|
| 1 | **Resilient Last-Mile Allocation in Humanitarian Supply Chains: A Reproducible Many-Objective Fuzzy Framework** | High — combines resilience theory + many-objective MOEA + reproducible artifact + sensitivity, all on one substrate | Medium-High — Python tool already exists; NSGA-III is in DEAP; reframing is writing-intensive but bounded | High — directly fits the SI scope; ≥50% new content over ATRes is achievable; appeals to both SCR and MOEA communities | **Selected** |
| 2 | NSGA-III vs. NSGA-II vs. NRGA on Humanitarian MOEAs (pure methodology) | Medium — methodological sibling to the ATRes paper | High — narrowest scope | Low — does not fit the SCR special issue scope; closer to a regular MOEA venue | Reject (wrong venue) |
| 3 | FIS Sensitivity in Disaster Operation Management (sensitivity-only) | Medium-High — opens a real gap | High — narrow scope | Low — too narrow for an SI feature paper; fits a methodology venue better | Reject (too narrow) |
| 4 | Real-world case study (2023 Türkiye / 2024 Valencia) on top of #1 | Highest — empirical anchor | Low — data acquisition risk in 6–8 weeks | High — but high failure risk | Defer to v0.3.0 |

## Selected idea — distinction matrix vs. ATRes

| Dimension | ATRes 2026 (in press) | MDPI 2026 (planned) |
|-----------|------------------------|----------------------|
| Framing | Disaster operations management; OR vocabulary (cost, fairness, infeasibility) | Humanitarian supply chain resilience; 4R vocabulary (Robustness/Redundancy/Resourcefulness/Rapidity) explicitly mapped to objective functions |
| Theoretical contribution | Three new qualitative indices | New 4R↔index mapping; Section 3.5 on resilience-theoretic interpretation |
| Objectives | Three (ULPP, TIL, CAIL) | **Four** (ULPP, TRD, RPD, CAIL) — TIL split into Robustness and Rapidity components |
| Algorithms | NSGA-II, NRGA | NSGA-II, NRGA, **NSGA-III** (added; reference-point-based, designed for many-objective) |
| Problem sizes | Two (5/150/50, 10/300/100) | **Three or four** scaling steps |
| Sensitivity analyses | None | **FIS rule-base perturbation + weight perturbation**, two new sections |
| Implementation | MATLAB R2024b (closed) | Python `presidio-hardened-vol-assign` (open, MIT, citable Zenodo DOI) |
| Reproducibility | Not released | Full code + data + experiment scripts; MDPI Data Availability statement |
| Audience | Disaster logistics community | SCR + MOEA + open-science communities |

The new content fraction (Sections 3.5, parts of 4, parts of 5, all of 6
sensitivity, 7 reproducibility, plus a rewritten 1–2 framing) clears MDPI's
>50% new-content threshold for extended versions. Citation of the ATRes paper
on the title page satisfies MDPI's prior-publication disclosure rule.

## Prior Own Work Inventory

- Rabiei, P.; Arias-Aranda, D. (2021). *Introducing a novel multi-objective
  optimization model for vehicle routing and relief supply distribution in
  post-disaster phase: combining fuzzy inference systems with NSGA-II and
  NRGA.* ICTIS 2021, Wuhan. pp. 1226–1243.
- Rabiei, P.; Arias-Aranda, D.; Stantchev, V. (2023). *Introducing a novel
  multi-objective optimization model for volunteer assignment in the
  post-disaster phase.* Expert Systems with Applications 226: 120142.
  DOI: 10.1016/j.eswa.2023.120142.
- Rabiei, P.; Arias-Aranda, D.; Stantchev, V. (2026, in press). *A Novel
  Multi-Objective Optimization Model for Directing Affected People to Relief
  Centers in Post-Disaster Scenarios: Combining Fuzzy Inference Systems with
  NSGA-II and NRGA.* Autonomous Transportation Research. Manuscript ATRES-S-26-00021.
- `presidio-hardened-vol-assign` v0.1.0 (2026). Python reference implementation.
  GitHub: presidio-v/presidio-hardened-vol-assign. MIT license.

## Decisions taken (binding for I-B onward)

- **4-objective formulation, not 3.** Split ATRes's TIL into two distinct
  objectives so that the 4R framework maps one-to-one onto the objective set:

  | 4R component | Objective (working name) | FIS / source |
  |---|---|---|
  | Resourcefulness | **Mn_ULPP** — Mean Unfairness in People Prioritization | FIS₁: (VS, IDL, RTR) → ULPP |
  | Robustness | **Mn_TRD** — Mean Transport Robustness Deficit | FIS₂ₐ: (RCS, PHS) → TRD |
  | Rapidity | **Mn_RPD** — Mean Rapidity Deficit | FIS₂ᵦ: TD → RPD (single-input fuzzy classifier) |
  | Redundancy | **Mn_CAIL** — Mean Center Allocation Imbalance Level | FIS₃: (COR, RDR, TD) → CAIL |

  This eliminates the conceptual fusion of Robustness with Rapidity inside the
  ATRes TIL index, and surfaces a Robustness/Rapidity Pareto trade-off the
  three-objective formulation could not express.

- **NSGA-III's relevance becomes unambiguous.** With four objectives, the
  reference-point design of NSGA-III (Deb & Jain 2014) is on home turf;
  Zheng & Doerr (2024) show NSGA-II's degradation begins well before four.
  The empirical hypothesis sharpens to a falsifiable claim about
  many-objective MOEA selection mechanisms in humanitarian allocation.

- **`pva` v0.2.0 must implement the 4-objective formulation** (split FIS₂,
  add 4D hypervolume — likely via pymoo since the existing custom 2D
  sweep-line HV does not generalize cleanly). Coverage and CLI changes folded
  into the v0.2.0 roadmap entry.

## Open Questions (to resolve in I-B → I-D)

- Should the new FIS₂ᵦ for Rapidity be a single-input fuzzy classifier on TD
  alone, or should it incorporate a second input (e.g., RTR — resource time
  remaining) to make rapidity context-aware? Single-input is cleaner; two-input
  is arguably more realistic.
- Perturbation magnitudes for the sensitivity studies — Latin Hypercube on
  weights ∈ ±20%, single-rule drops, consequent-class one-step shifts?
- Reference-point set for NSGA-III — Das-Dennis structured points with
  divisions p=4 (giving 35 points for 4D), or two-layer reference points for
  better boundary coverage?
