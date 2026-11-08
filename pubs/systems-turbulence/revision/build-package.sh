#!/bin/sh
# Build the MDPI revision package for systems-4669426 from paperB-mdpi-source-r1/:
# source zip (same layout as the submission), PDF built from a clean unpack, latexdiff
# PDF against the submitted source, and the three response letters as PDF.
set -e
R=$(cd "$(dirname "$0")" && pwd)
S=$R/paperB-mdpi-source-r1
OLD=$R/../submission/paperB-mdpi-source/author/paper.tex
P=$R/package
rm -rf "$P"; mkdir -p "$P/stage/paperB-mdpi-source/author" "$P/stage/paperB-mdpi-source/figures"
cp -R "$S/author/Definitions" "$S/author/tables" "$S/author/paper.tex" "$S/author/lit-audit.bib" "$P/stage/paperB-mdpi-source/author/"
cp "$S"/figures/*.pdf "$P/stage/paperB-mdpi-source/figures/"
(cd "$P/stage" && find . -name .DS_Store -delete && zip -qr ../paperB-mdpi-source-revised.zip paperB-mdpi-source)
mkdir "$P/build"; (cd "$P/build" && unzip -q ../paperB-mdpi-source-revised.zip)
B=$P/build/paperB-mdpi-source/author
cd "$B"
build() { pdflatex -interaction=nonstopmode "$1.tex" >/dev/null; bibtex "$1" >/dev/null; pdflatex -interaction=nonstopmode "$1.tex" >/dev/null; pdflatex -interaction=nonstopmode "$1.tex" >/dev/null; }
build paper; cp paper.pdf "$P/paperB-mdpi-revised.pdf"
latexpand paper.tex > new-flat.tex
# Tables are treated as opaque blocks (latexdiff cannot mark up inside booktabs/tabularx).
latexdiff --type=UNDERLINE --math-markup=whole --graphics-markup=none --disable-citation-markup \
  --append-textcmd="dataavailability,acknowledgments,authorcontributions" \
  --config="PICTUREENV=(?:picture|DIFnomarkup|tabular|tabularx|tikzpicture)[\w\d*@]*" \
  "$OLD" new-flat.tex > diff.tex 2> latexdiff.log
# The MDPI title, abstract and keywords live in the preamble, which latexdiff does not diff;
# all three were rewritten, so mark them as added.
python3 -I - <<'PY'
import re
s = open("diff.tex").read()
for macro in ("\\Title", "\\abstract", "\\keyword"):
    m = re.search(r"(?m)^" + re.escape(macro) + r"\{", s)
    j = m.end(); depth = 1; k = j
    while depth:
        if s[k] == "%":
            k = s.index("\n", k) + 1; continue
        depth += (s[k] == "{") - (s[k] == "}"); k += 1
    s = s[:j] + "\\DIFadd{" + s[j:k - 1] + "}" + s[k - 1:]
# The Highlights section is new (template update of November 2026): mark its items and headings.
a = s.index("\\renewcommand{\\addhighlights}{")
e = re.compile(r"(?m)^\}\s*(%DIF[^\n]*)?$").search(s, a).start()
blk = re.sub(r"(?m)^\\item (.*?)\s*(%DIF[^\n]*)?$", r"\\item \\DIFadd{\1} \2", s[a:e])
blk = re.sub(r"\\textbf\{(What are[^}]*)\}", r"\\textbf{\\DIFadd{\1}}", blk)
s = s[:a] + blk + s[e:]
open("diff.tex", "w").write(s)
PY
build diff; cp diff.pdf "$P/paperB-mdpi-diff-highlighted.pdf"
cd "$R"
for n in 1 2 3; do
  pandoc "response-reviewer-$n.md" -o "$P/response-reviewer-$n.pdf" --pdf-engine=xelatex \
    -V mainfont="Palatino" -V geometry:margin=2.5cm -V fontsize=11pt
done
grep -a "Output written" "$B/paper.log" "$B/diff.log"
ls -la "$P"
