# Building the MDPI (Applied Sciences) manuscript

`paper.tex` targets the official MDPI class (`\documentclass[applsci,...]{Definitions/mdpi}`).

## Binaries kept out of git (per the repo's no-large-binaries policy)

Two sets of required files are **git-ignored** and must be restored before building:

1. **MDPI template logos** — `Definitions/logo-*.eps` and `Definitions/logo-*.pdf`.
   Restore by downloading the official MDPI LaTeX template
   (<https://www.mdpi.com/authors/latex>) into `Definitions/`, or by unzipping the
   submission bundle `../submission/paperA-mdpi-source.zip`.
   The text template files (`mdpi.cls`, `mdpi.bst`, `journalnames.tex`, `chicago2.bst`)
   **are** committed.

2. **Figures** — `../figures/fig5–9.pdf`. Regenerate with
   `python experiments/make_figures.py`, or take them from the submission bundle.

## Build

```bash
latexmk -pdf paper.tex   # needs Definitions/ logos + ../figures/ present
```

## Submission bundle

`../submission/paperA-mdpi-source.zip` is the self-contained source (paper.tex, lit.bib,
Definitions/, figures/) that builds as-is — this is what to upload to MDPI SuSy, alongside
`../submission/paperA-mdpi.pdf`.
