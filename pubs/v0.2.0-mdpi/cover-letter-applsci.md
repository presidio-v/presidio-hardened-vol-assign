# Cover Letter — Applied Sciences (MDPI)

**To:** Guest Editors, Special Issue *Innovations in Supply Chain Resilience*
(Prof. Agnusdei, Prof. Silvestri, Prof. Di Pietro)
**Journal:** *Applied Sciences* (MDPI)
**Date:** 30 June 2026

**Manuscript:** *Resilient Last-Mile Allocation in Humanitarian Supply Chains:
A Reproducible Many-Objective Fuzzy Framework*

**Authors:** Peyman Rabiei, Daniel Arias-Aranda, Vladimir Stantchev (corresponding)

---

Dear Guest Editors,

We submit the manuscript above for consideration in the Special Issue
*Innovations in Supply Chain Resilience*.

Response-phase relief planning is a triage decision made under a binding
capacity constraint: there is never enough relief to meet need, so affected
people must be directed to relief centres under time pressure. Published
optimisation models for this problem speak the language of operations research
— cost, distance, equity. Civil-protection planners speak the language of
supply-chain resilience — the 4R framework (Robustness, Redundancy,
Resourcefulness, Rapidity) of Bruneau et al. Our contribution closes that
vocabulary gap, and we believe it fits the Special Issue squarely.

**What the paper contributes.**

- A four-objective humanitarian last-mile allocation model whose objectives
  map one-to-one onto the 4R components, each derived from Bruneau et al.'s
  definitions rather than labelled post hoc, and each evaluated by a Mamdani
  fuzzy inference system.
- An empirical demonstration, over 540 runs across three problem sizes, that
  splitting transportation infeasibility into separate Robustness and Rapidity
  axes gives the decision-maker a visibility gain the fused three-objective
  formulation cannot provide — established by rank-correlation and
  projection-dominance analysis against a null-model control.
- A three-way evolutionary-algorithm comparison (NSGA-II, NRGA, NSGA-III)
  showing NSGA-III is ~32% faster than NSGA-II at every problem size with
  statistically equivalent Pareto-front quality.
- Rule-base deletion and weight-perturbation (±20%, 100 Latin-Hypercube
  samples) sensitivity analyses establishing recommendation robustness and
  identifying the Rapidity classifier as a structural sensitivity hot-spot.
- A fully reproducible, MIT-licensed Python reference implementation released
  as an open-source artifact, with all experiment drivers and result manifests.

**Relationship to the authors' prior work.**
This is the primary publication of the model. It builds on the authors' prior
FIS-MOEA research line for disaster operations — post-disaster vehicle routing
and relief distribution (Rabiei & Arias-Aranda, 2021) and volunteer-to-vacancy
assignment (Rabiei, Arias-Aranda & Stantchev, 2023), both published and cited —
but the people-to-relief-centre allocation model, its resilience-theoretic 4R
formulation, the four-objective Robustness/Rapidity split, the NSGA-III
comparison, the rule-base and weight-perturbation sensitivity analyses, and the
open-source implementation are all presented here for the first time. The
manuscript is fully self-contained and is not an extension of any prior
published article.

**Declarations.**
The manuscript is original, has not been published elsewhere, and is not under
consideration by any other journal. All authors have read and approved the
submission and agree to its content. The authors declare no competing
interests.

**Special Issue invitation and APC waiver.**
This submission follows correspondence with the Section Managing Editor,
Dr. Dominic Ling, who confirmed the manuscript's fit with the Special Issue
*Innovations in Supply Chain Resilience* and the applicability of a full
(100%) waiver of the Article Processing Charges, to be applied by the
editorial office after submission.

**Data and code availability.**
The reference implementation, experiment drivers, and result manifests are
publicly available at
`github.com/presidio-v/presidio-hardened-vol-assign`, archived at Zenodo
(DOI: 10.5281/zenodo.21083547).

We thank you for considering our work and look forward to the reviewers'
comments.

Sincerely,

**Vladimir Stantchev** (corresponding author)
Institute of Information Systems, SRH Berlin University of Applied Sciences,
Germany
stantchev@computer.org

on behalf of Peyman Rabiei and Daniel Arias-Aranda
Faculty of Economics and Business, University of Granada, Spain
