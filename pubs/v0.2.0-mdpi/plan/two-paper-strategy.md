# Two-paper strategy (confirmed 2026-06-30)

One substrate — the `presidio-hardened-vol-assign` v0.2.0 FIS-MOEA humanitarian
allocation tool — feeds **two genuinely distinct papers** at two MDPI venues.

## Venues

| | **Paper A — Applied Sciences** | **Paper B — Systems** |
|---|---|---|
| SI | "Innovations in Supply Chain Resilience" (Feature Paper) | "Using Digital AI Systems as a Response to High Economic Turbulence and Uncertainty" |
| Metrics | IF 2.5, CiteScore 5.5 | SSCI **Q1**, IF 3.8 |
| Editors | GEs Agnusdei / Silvestri / Di Pietro; Managing Ed. **Dr. Dominic Ling** | GEs **Jong-min Kim** & **Rob Kim Marjerison**; AE **Belinda Zhao** (belinda.zhao@mdpi.com) |
| APC | **100% waiver CONFIRMED** (Ling applies it after submission) | **No waiver yet** — "may qualify for discount, contact Belinda Zhao" |
| Deadline | ~31 Aug 2026 (per committed proposal) | **30 Nov 2026** (~20-day first decision; free English editing post-accept) |
| SI link | (Applied Sciences SCR SI) | https://www.mdpi.com/journal/systems/special_issues/105O3VCX29 |

## Mapping (confirmed)

- **Paper A → Applied Sciences SCR** = the **current 4R manuscript** (`pubs/v0.2.0-mdpi/author/paper.tex`,
  "Resilient Last-Mile Allocation in Humanitarian Supply Chains"). It is a *superset*
  of the committed proposal (FIS framework + open tool + reproducibility metric, **plus**
  the 4R / 4-objective / NSGA-III extension). Realign the abstract/intro to keep the
  proposal's two promised headline contributions visible (production-grade audit-ready
  **tool**; **bit-for-bit reproducibility as a resilience criterion**). The cover letter
  (`cover-letter-applsci.md`) is already correct for this paper, incl. the APC-waiver note.
- **Paper B → Systems Q1** = a **new, distinct paper** reframing the tool as a *digital AI
  decision system operating under acute disruption / uncertainty*. Distinct RQ + framing +
  ideally distinct analysis (security / auditability / reproducibility-under-perturbation).
  Likely centred on the **3-objective** model (FLPP/TFL/CABL, NSGA-II vs NRGA) to stay
  clearly separate from Paper A's 4-objective core. Honest cross-citation to Paper A.

> The committed Applied Sciences proposal text (title "From Methodology to Practice: A
> Production-Grade Decision Support Tool…", 3 objectives, NSGA-II/NRGA, reproducibility,
> 2 sizes) describes the *tooling* angle — that framing is reused for **Paper B**, while
> Paper A keeps the 4R theory. Abstracts were explicitly "tentative".

## Phases

- **Phase 0 — comms (now):** (1) Ling: SCR feature paper on track ~31 Aug, 4R-forward,
  confirm waiver. (2) Belinda Zhao: intent + APC-discount query for Systems. (3) Daniel
  confirms Paper B angle (he initiated the Systems invite).
- **Phase 1 — Paper A to submission (~3–4 wk):** realign abstract/intro (tool +
  reproducibility-as-resilience); co-author sign-off (Rabiei, Arias-Aranda); submit;
  Ling applies waiver. (Zenodo DOI, CI, reproducibility already done.)
- **Phase 2 — Paper B design (Aug–Sep):** lock distinct RQ + distinguishing analysis;
  decide 3-obj model + the "digital AI system under turbulence" framing.
- **Phase 3 — Paper B build + submit (Oct → ≤ mid-Nov):** write, run new experiments,
  internal adversarial review, sign-off, submit before the 30 Nov crunch.

## Integrity guardrails

Different RQ + framing + headline results; explicit cross-citation; no shared text; no
dual-submission of the same contribution. Same substrate, two distinct contributions.

## Status of Paper A artifacts (as of 2026-06-30)

Manuscript placeholder-free, builds clean (29 pp), Zenodo v0.2.0 archived
(concept DOI 10.5281/zenodo.21083544, version 10.5281/zenodo.21083547), CI green,
affiliation = SRH University Heidelberg, back matter resolved. **Remaining for Paper A:**
abstract/intro realignment + co-author sign-off + portal submission.
