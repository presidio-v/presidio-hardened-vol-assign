---
provisional: true
phase: I-B
project: presidio-hardened-vol-assign
target-venue: Applied Sciences (MDPI), SI "Innovations in Supply Chain Resilience"
authors: Rabiei, Arias-Aranda, Stantchev
---

## Problem Statement (1–3 sentences)

Humanitarian last-mile allocation models direct affected people to relief
centers under simultaneous fairness, transport, and capacity constraints, but
none of the published FIS-MOEA approaches in this line of work organize their
objectives around a recognized supply-chain-resilience framework, conflate
network robustness with operational rapidity inside a single transport
infeasibility index, and release no reproducible artifact. The problem this
paper addresses: how to formulate, solve, and validate a four-objective
many-objective allocation model whose objectives map one-to-one onto Bruneau
et al.'s 4R resilience framework (Robustness, Redundancy, Resourcefulness,
Rapidity), evaluated by Mamdani fuzzy inference systems, solved by NSGA-II,
NRGA, and NSGA-III, and shipped with an open-source Python reference
implementation that practitioners and reviewers can run.

## Primary Beneficiary

**Mixed: practitioners and the academic SCR + MOEA communities.**

- *Practitioners* (humanitarian logistics planners, civil-protection agencies)
  gain a decision-support tool with explicit Robustness/Rapidity/Resourcefulness/
  Redundancy trade-offs, expressed in vocabulary that matches the resilience
  policy literature they already work with.
- *Academic SCR community* gains the first FIS-MOEA model whose objective
  functions are explicitly anchored in the 4R framework — a methodological
  bridge between supply-chain resilience theory and computational disaster
  operations management.
- *Academic MOEA community* gains an empirical comparison of NSGA-II, NRGA,
  and NSGA-III on a four-objective humanitarian problem with FIS-evaluated
  objectives — a setting under-represented in the algorithmic literature.

## Tradeoff Assessment (rigor vs. impact)

The paper sits closer to the rigor pole than ATRes did. Splitting TIL into
Robustness and Rapidity components is theoretically motivated (4R) rather than
purely empirically motivated, and exposes the model to harder algorithmic
scrutiny (many-objective MOEAs are known to misbehave). Adding a sensitivity
study and a reproducible artifact further raises the rigor bar.

The impact pole is preserved by keeping the application substrate
(post-disaster relief allocation) and by writing for an applied audience —
*Applied Sciences*' readership, not a pure MOEA venue. Decision-makers and
practitioners are the named audience in §6 (Discussion).

The deliberate balance: methodological rigor *in service of* practitioner-
relevant trade-offs, with the resilience-theory framing as the bridge.

## "So What" Argument

Disaster frequency is rising (CRED 2023: 399 events, 86,473 deaths,
US$202.7 bn damage). Last-mile relief allocation models are the computational
substrate for the response phase. Three concrete payoffs from this work:

1. **Trade-off vocabulary practitioners can use.** Civil-protection agencies
   already plan in 4R vocabulary (or its derivatives — FEMA, UNDRR, EU CCM).
   Optimization tools that emit results in *cost/distance* space force
   translation. Tools that emit results in *Robustness/Redundancy/Resourcefulness/
   Rapidity* space match the policy lens directly.

2. **A Robustness/Rapidity trade-off the 3-objective ATRes model cannot
   express.** When TIL fuses roadworthiness with travel duration, the
   decision-maker cannot ask "should I send a vulnerable person 30 minutes
   farther on a robust road, or 10 minutes on a fragile one?" The
   4-objective formulation answers this directly.

3. **A reproducible reference implementation that closes the FIS-MOEA-disaster
   reproducibility gap.** No paper in this line of work has released code.
   `presidio-hardened-vol-assign` v0.2.0 will be MIT-licensed, Zenodo-citable,
   pip-installable, and runnable on a laptop in minutes.

## Distinction from Existing Work

**One-sentence distinguishability:** The first humanitarian last-mile
allocation model whose four objectives map one-to-one onto Bruneau et al.'s
4R resilience framework, evaluated by FIS, solved by NSGA-III alongside
NSGA-II and NRGA, sensitivity-tested under FIS rule and weight perturbation,
and released as an open-source Python reference implementation.

**Distinction from ATRes 2026 (in press, same authors):** different objective
formulation (4 vs. 3), different algorithm pool (3 vs. 2), different
reproducibility posture (open Python artifact vs. closed MATLAB), additional
sensitivity sections, and a resilience-theoretic framing absent from ATRes.
Estimated >50% new content; satisfies MDPI extended-version policy. ATRes
cited on the title page per MDPI prior-publication rule.

**Distinction from Rabiei et al. 2023 ESWA (volunteer assignment, not
allocation):** different problem (volunteers → tasks vs. people → centers),
different FIS structure, different objectives. Cited as foundational
methodology.

**Distinction from the broader humanitarian-allocation literature**
(Hashim 2021; Zhao 2017, 2019; Gama 2016; Chang 2024; Hansuwa 2021;
Soghrati Ghasbeh 2022; Wang 2020; Abounacer 2014; Tzeng 2007; Beiki 2020):
none of these use the 4R framework to organize objectives, none use
FIS to evaluate qualitative objectives, and none release reproducible
artifacts.

**Distinction from the SCR theory literature** (Bruneau 2003;
Christopher & Peck 2004; Hosseini 2016; Tukamuhabwa 2015; Behl & Dutta 2019):
those works define resilience taxonomies but do not propose computational
allocation models that operationalize them. This paper is the operationalization.
