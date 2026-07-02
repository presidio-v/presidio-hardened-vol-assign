# Paper A — Applied Sciences (MDPI) submission checklist

**Venue:** Applied Sciences, SI "Innovations in Supply Chain Resilience" (Feature Paper).
**Portal:** MDPI SuSy — use the SI "Submit to Special Issue" button (Vladimir's MDPI login).
**APC:** 100% waiver confirmed by Dr. Dominic Ling — applied by the editorial office *after*
submission; do **not** pay at submission. Email Ling once the manuscript ID is issued.

## Files to upload
- [x] **Manuscript** — MDPI-template LaTeX source + PDF. **CR-1 done:** `author/paper.tex`
      converted to `\documentclass[applsci,...]{Definitions/mdpi}`, builds clean (29 pp,
      0 undefined refs). Upload `submission/paperA-mdpi-source.zip` + `submission/paperA-mdpi.pdf`.
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
- [x] ORCIDs — in the manuscript: Rabiei 0000-0003-1534-6680; Arias-Aranda 0000-0003-0292-7435;
      Stantchev 0000-0002-1551-419X. Emails: prabiei@correo.ugr.es; darias@ugr.es; stantchev@computer.org
- [ ] Article type: **Article**
- [ ] Special Issue: "Innovations in Supply Chain Resilience" (auto-selected via the express link)
- [ ] Keywords (from the manuscript)
- [ ] Suggested reviewers (3–5) and any excluded reviewers — **TODO: authors to provide**

## Declarations (all present in the manuscript back matter)
- [x] Author Contributions (CRediT) — co-authors confirmed
- [ ] Funding — "This research received no external funding."
- [ ] IRB / Informed Consent — Not applicable (synthetic data)
- [ ] Data Availability — repo + Zenodo DOIs (concept 10.5281/zenodo.21083544; version 10.5281/zenodo.21083547)
- [x] AI-use disclosure — co-authors confirmed the "no generative AI tools" statement is accurate (audit M-4)
- [ ] Conflicts of Interest — none declared

## Pre-flight (author-owned)
- [x] Co-author final approval of the PDF (Rabiei, Arias-Aranda)
- [x] Refresh the Zenodo record *description* (audit m-2) — live record updated by author;
      `.zenodo.json` updated in repo (NSGA-III + 4R + allocation framing; author ORCIDs) for future releases
- [ ] After manuscript ID issued: email Dr. Ling to apply the 100% APC waiver

## Blocking gate — CLEARED
- [x] **CR-1 — MDPI `applsci`/`mdpi.cls` template conversion — DONE.** Source: `author/paper.tex`
      (+ `author/Definitions/`, text files committed; logos git-ignored, see `author/README-mdpi.md`).
      Builds clean, 29 pp, 0 undefined refs, references in MDPI style.
