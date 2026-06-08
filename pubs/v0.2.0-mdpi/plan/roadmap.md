---
phase: II-F
project: presidio-hardened-vol-assign
created: 2026-05-09
---

## Patent Track

**Status:** Not applicable.

The FIS+NSGA-II+NRGA pattern was publicly disclosed in Rabiei et al.
(ESWA 2023, DOI 10.1016/j.eswa.2023.120142). The three-index allocation
formulation is publicly disclosed in Rabiei, Arias-Aranda, Stantchev
(ATRes 2026, in press). The 4R-mapped four-objective extension proposed
here is a theoretical reframing plus an empirical comparison and a
reproducibility artifact — none of these are patentable subject matter
under EPC Art. 52(2) or USPTO §101 in their current form. No patent gate
applies; publication proceeds directly.

---

## v0.1.0 — MVP (DELIVERED 2026-03-31)

**Scope:** Citable Python reference implementation of Rabiei et al.
(ESWA 2023) volunteer-assignment model.

**Artifacts (all delivered):**

- ☑ Tool: `presidio-hardened-vol-assign` v0.1.0 — CLI `pva`,
  scikit-fuzzy + DEAP, NSGA-II + NRGA, three FISs, three objectives;
  139 tests, 97% coverage; MIT licensed; GitHub release.
- ☑ Publication: Rabiei, Arias-Aranda, Stantchev (2026, in press)
  *A Novel Multi-Objective Optimization Model for Directing Affected
  People to Relief Centers in Post-Disaster Scenarios.* Autonomous
  Transportation Research, manuscript ATRES-S-26-00021.
- ☑ Repo hardening: SECURITY.md, dependabot, codeql, CI, 24-hour
  pip-audit cache.

---

## v0.2.0 — MDPI Applied Sciences SI: Resilience extension

**Scope:** Extended journal version of ATRes 2026 for the Applied
Sciences SI *Innovations in Supply Chain Resilience* (Guest Editors:
Agnusdei, Silvestri, Di Pietro). New theoretical framing (4R), new
methodology (4-objective split, NSGA-III), new empirics (sensitivity
analyses, three problem sizes), Python tool as reproducible artifact.

**Window:** 2026-05-09 → 2026-07-04 (8 weeks; 6-week stretch goal).
Editor commitment: 6–8 weeks.

**Artifacts:**

- ☐ **Publication:** *Resilient Last-Mile Allocation in Humanitarian
  Supply Chains: A Reproducible Many-Objective Fuzzy Framework*
  (working title). Applied Sciences (MDPI), SI on Supply Chain
  Resilience. Target submission: **2026-06-27** (week 7); hard
  deadline: **2026-07-04** (week 8).
  - Pattern: `~/.claude/book-implementation-plan.md` (instantiated at
    `plan/book-implementation-plan.md`).
  - Working dir: `pubs/v0.2.0-mdpi/author/`.
  - Style: `~/.claude/vladimir-prose-style.md` (spirit, in English —
    claim-first openers, no hedging, binary framings, concrete
    actors; the German-language register markers do not transfer).
- ☐ **Tool extension:** `pva` v0.2.0 — adds new
  `presidio_vol_assign.allocation` sub-package implementing the ATRes/MDPI
  people-to-relief-center model as a parallel module to the existing
  volunteer-assignment code. New CLI subcommands: `pva allocate` and
  `pva alloc-metrics`. Eight v0.2.0 deliverables tracked in
  `explore/feasibility.md` §"Fast path"; items 1–8 done as of 2026-05-09
  (Week 1 of the schedule, ahead of plan).
  - Pattern: `~/.claude/code-implementation-plan.md`.
  - Repo: GitHub `presidio-v/presidio-hardened-vol-assign`.
  - Release tag: `v0.2.0` mid Week 7. Zenodo DOI mints automatically.
- ☐ **Cover letter:** `pubs/v0.2.0-mdpi/cover-letter-applsci.md`
  (working) and `cover-letter-applsci-submit.pdf` (clean upload
  copy). Must include ATRes prior-publication disclosure per MDPI
  policy and per design-pattern Multiple Submissions rule.
- ☐ **Adversarial pre-submission review:** `pubs/v0.2.0-mdpi/
  adversarial-review.md`. Severity-labelled (Critical/Moderate/Minor).
  All Critical resolved before submission.
- ☐ **Forward feasibility probe** for v0.3.0 — confirm that the SCR
  reframing does not collapse under a reviewer's basic check of the
  4R framework's semantics (Bruneau 2003, Hosseini 2016).

**Sequence (week-by-week):**

| Week | Dates | Deliverable |
|------|-------|-------------|
| 1 | May 9–16 | Phase I (✔ in progress) and II artifacts; co-author alignment email; SI scope confirmation with Agnusdei |
| 2 | May 17–23 | `pva` v0.2.0 critical-path: 4-obj split, NSGA-III, rule-base override |
| 3 | May 24–30 | `pva` weight override, 4D HV, projection helper; Pareto-projection tests; NSGA-III validation on DTLZ2 |
| 4 | May 31–Jun 6 | H1 and H2 experiment runs (3 sizes × 3 algos × 30 reps); first figures |
| 5 | Jun 7–13 | H3a + H3b experiment runs (overnight); statistical tests; final figures; §1–4 draft |
| 6 | Jun 14–20 | §5–9 draft; figures finalised; integrated read-through |
| 7 | Jun 21–27 | Adversarial review; revisions; cover letter; tag `pva` v0.2.0; mint Zenodo DOI |
| 8 | Jun 28–Jul 4 | Final read; clean PDFs (manuscript + cover letter); MDPI portal submission |

**Dependencies on external events:**

- Co-author alignment (Rabiei, Arias-Aranda) — blocks experimental work
  if not received by end of Week 2.
- ATRes/Elsevier copyright clearance — blocks MDPI submission if not
  received by Week 7.
- MDPI portal access — confirm Vladimir has an MDPI account before
  Week 7.

**Submission gate checklist (per design pattern, must all clear before
clicking submit):**

- ☐ Adversarial self-review completed; all Critical issues resolved
- ☐ Cover letter discloses ATRes 2026 as prior publication
- ☐ Cover letter discloses arXiv preprint URL (if posted)
- ☐ Forward feasibility probe complete (no v0.2.0 claim contradicted)
- ☐ Figures and tables: every `\label{}` referenced in prose
- ☐ Bibliography uses shared `~/pubs/lit/lit.bib` (or project-local
  if shared bib not yet established)
- ☐ All figures: no hardcoded "Fig. N" / "Figure N" in titles
- ☐ Zenodo DOI minted, cited in Data Availability statement
- ☐ Repo state at v0.2.0 tag matches paper claims
- ☐ Co-author final approval (Rabiei, Arias-Aranda)

---

## v0.3.0 — Real-world case study (FUTURE, not in MDPI submission)

**Scope:** Apply the 4R-mapped framework to a documented event
(2023 Türkiye–Syria earthquake or 2024 Valencia floods). Empirical
anchoring with publicly available data (EM-DAT, OSM, INFORM Risk
Index, INSPIRE post-event reports).

**Artifacts (placeholder):**

- ☐ Case-study extension paper (target venue TBD — possibly EJOR,
  Computers & OR, or Transportation Science).
- ☐ `pva` v0.3.0 with case-study data importers and visualization.

**Dependencies:** Data acquisition and IRB review (if survey data is
included). Schedule: post-MDPI submission, 2026-Q3.

---

## Notes on naming and identity

- **Project slug:** `presidio-hardened-vol-assign` (per existing repo).
  Note the original project name has a typo (`vol-asssign`); the slug
  used in publications is the corrected `vol-assign`.
- **Tool name:** `pva` (CLI binary).
- **Publication identifier:** `vol-assign-mdpi-2026` (used in
  cover-letter file naming).
