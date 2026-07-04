# Paper B — Systems (MDPI) — workstream

**This directory is a fully self-contained artefact, kept separate from Paper A
(`pubs/v0.2.0-mdpi/`).** Each paper carries its own manuscript, bibliography, figures,
build, and submission bundle so that revised / final / proof versions of A and B can be
produced independently without touching each other. Paper B is developed on its own git
branch (`paper/systems-turbulence`); Paper A lives on `pub/allocation-extension`.

## Target

- **Journal:** *Systems* (MDPI), SSCI **Q1**, IF 3.8.
- **Special Issue:** "Using Digital AI Systems as a Response to High Economic Turbulence
  and Uncertainty" — <https://www.mdpi.com/journal/systems/special_issues/105O3VCX29>
- **Editors:** GEs Jong-min Kim & Rob Kim Marjerison; AE **Belinda Zhao** (belinda.zhao@mdpi.com).
- **Deadline:** 30 Nov 2026 (~20-day first decision). **No confirmed APC waiver** — discount
  enquiry pending with Belinda Zhao.

## One-line thesis

An auditable, reproducible **digital AI decision system** that keeps constrained
resource-allocation decisions reliable when its inputs are degraded by turbulence and
uncertainty — the system, not the optimisation theory, is the contribution.

## Relationship to Paper A (no salami-slicing)

Same substrate (`presidio-hardened-vol-assign`), **distinct contribution**: Paper A is the
4R theory / many-objective paper (Applied Sciences); Paper B is the systems/operations paper
built on the **3-objective** core, with new RQs, new analyses (uncertainty/degradation,
reproducibility-under-perturbation, operability), and honest cross-citation to Paper A.
See `plan/design-brief.md` and the strategy doc on the Paper A branch
(`pubs/v0.2.0-mdpi/plan/two-paper-strategy.md`).

## Layout (Paper A convention, mirrored)

- `plan/` — design brief, roadmap
- `explore/` — problem statement, RQ/hypotheses, scoping
- `author/` — manuscript, bib, figures, Definitions (created when drafting starts)
- `submission/` — bundle + checklist (created near submission)

## Status

Phase 2 (design). **Blocked on one decision (Daniel):** framing bridge B1 vs B2 and whether
a humanitarian primary case is acceptable in an *economic*-turbulence SI — see
`explore/scoping.md`. Code for the new analyses is not yet built.
