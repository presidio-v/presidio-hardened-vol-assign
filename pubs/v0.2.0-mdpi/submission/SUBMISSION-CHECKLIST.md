# Paper A — Applied Sciences (MDPI) submission checklist

**Venue:** Applied Sciences, SI "Innovations in Supply Chain Resilience" (Feature Paper).
**Portal:** MDPI SuSy — use the SI "Submit to Special Issue" button (Vladimir's MDPI login).
**APC:** 100% waiver confirmed by Dr. Dominic Ling — applied by the editorial office *after*
submission; do **not** pay at submission. Email Ling once the manuscript ID is issued.

## Files to upload
- [ ] **Manuscript** — MDPI-template LaTeX source + PDF *(GATED on template conversion, CR-1)*.
      Current source: `author/paper.tex` (+ `lit.bib`, `../figures/fig5–9`). Builds clean, 29 pp.
- [ ] **Cover letter** — `cover-letter-applsci.md` (ready; addressed to the GEs, carries the
      APC-waiver note). Paste into the SuSy cover-letter box or upload as PDF.
- [ ] **Figures** — high-res PDFs already in `../figures/` (embedded in the build).

## Metadata to enter in SuSy
- [ ] Title: *Resilient Last-Mile Allocation in Humanitarian Supply Chains: A Reproducible
      Many-Objective Fuzzy Framework*
- [ ] Authors + order: Peyman Rabiei, Daniel Arias-Aranda, **Vladimir Stantchev (corresponding)**
- [ ] Affiliations: Rabiei & Arias-Aranda — Faculty of Economics and Business, University of
      Granada, Spain; Stantchev — Institute of Information Systems, SRH University Heidelberg, Germany
- [ ] Corresponding email: stantchev@computer.org
- [ ] ORCIDs — **TODO: collect for all three authors** (SuSy requires at least the corresponding author's)
- [ ] Article type: **Article**
- [ ] Special Issue: "Innovations in Supply Chain Resilience" (auto-selected via the express link)
- [ ] Keywords (from the manuscript)
- [ ] Suggested reviewers (3–5) and any excluded reviewers — **TODO: authors to provide**

## Declarations (all present in the manuscript back matter)
- [ ] Author Contributions (CRediT) — **co-authors confirm roles** before submitting
- [ ] Funding — "This research received no external funding."
- [ ] IRB / Informed Consent — Not applicable (synthetic data)
- [ ] Data Availability — repo + Zenodo DOIs (concept 10.5281/zenodo.21083544; version 10.5281/zenodo.21083547)
- [ ] AI-use disclosure — **co-authors confirm** the "no generative AI tools" statement is literally
      accurate for manuscript preparation (audit M-4); if any AI assistance was used, switch to MDPI's
      standard disclosure form
- [ ] Conflicts of Interest — none declared

## Pre-flight (author-owned)
- [ ] Co-author final approval of the PDF (Rabiei, Arias-Aranda)
- [ ] Refresh the Zenodo record *description* to mention NSGA-III + people-to-relief-centre
      allocation (audit m-2) so the artifact matches the paper
- [ ] After manuscript ID issued: email Dr. Ling to apply the 100% APC waiver

## Blocking gate
- [ ] **CR-1 — convert manuscript to the MDPI `applsci`/`mdpi.cls` template** (currently generic
      `article` + `authblk`). This is the single most likely cause of an editorial bounce-back and
      must be done before upload. See the quality audit (`../quality-audit-paperA.md`, CR-1).
