#!/bin/sh
# Build the MDPI round-2 package for systems-4669426 from paperB-mdpi-source-r2/:
# source zip, PDF from a clean unpack, latexdiff PDF against the round-1 revision (r1),
# supplementary PDF, and the round-2 response letter as PDF.
set -e
R=$(cd "$(dirname "$0")" && pwd)
S=$R/paperB-mdpi-source-r2
O=$R/paperB-mdpi-source-r1/author
P=$R/package-r2
rm -rf "$P"; mkdir -p "$P/stage/paperB-mdpi-source/author" "$P/stage/paperB-mdpi-source/figures"
cp -R "$S/author/Definitions" "$S/author/tables" "$S/author/paper.tex" "$S/author/lit-audit.bib" "$P/stage/paperB-mdpi-source/author/"
cp "$S"/figures/*.pdf "$P/stage/paperB-mdpi-source/figures/"
(cd "$P/stage" && find . -name .DS_Store -delete && zip -qr ../paperB-mdpi-source-revised-r2.zip paperB-mdpi-source)
mkdir "$P/build"; (cd "$P/build" && unzip -q ../paperB-mdpi-source-revised-r2.zip)
B=$P/build/paperB-mdpi-source/author
build() { pdflatex -interaction=nonstopmode "$1.tex" >/dev/null; bibtex "$1" >/dev/null; pdflatex -interaction=nonstopmode "$1.tex" >/dev/null; pdflatex -interaction=nonstopmode "$1.tex" >/dev/null; }
(cd "$O" && latexpand paper.tex > "$B/old-flat.tex")
cd "$B"
build paper; cp paper.pdf "$P/paperB-mdpi-revised-r2.pdf"
latexpand paper.tex > new-flat.tex
latexdiff --type=UNDERLINE --math-markup=whole --graphics-markup=none --disable-citation-markup \
  --append-textcmd="dataavailability,acknowledgments,authorcontributions,supplementary" \
  --config="PICTUREENV=(?:picture|DIFnomarkup|tabular|tabularx|tikzpicture)[\w\d*@]*" \
  old-flat.tex new-flat.tex > diff.tex 2> latexdiff.log
# The abstract lives in the preamble, which latexdiff does not diff; mark it changed as a whole.
python3 -I - <<'PY'
import re
s = open("diff.tex").read()
m = re.search(r"(?m)^\\abstract\{", s)
j = m.end(); depth = 1; k = j
while depth:
    if s[k] == "%":
        k = s.index("\n", k) + 1; continue
    depth += (s[k] == "{") - (s[k] == "}"); k += 1
s = s[:j] + "\\DIFadd{" + s[j:k - 1] + "}" + s[k - 1:]
open("diff.tex", "w").write(s)
PY
build diff; cp diff.pdf "$P/paperB-mdpi-diff-highlighted-r2.pdf"
cd "$R/supplementary"
pdflatex -interaction=nonstopmode supplementary.tex >/dev/null; pdflatex -interaction=nonstopmode supplementary.tex >/dev/null
cp supplementary.pdf "$P/systems-4669426-supplementary.pdf"
cd "$R"
pandoc "response-reviewer-2-r2.md" -o "$P/response-reviewer-2-r2.pdf" --pdf-engine=xelatex \
  -V mainfont="Palatino" -V geometry:margin=2.5cm -V fontsize=11pt
grep -a "Output written" "$B/paper.log" "$B/diff.log"
grep -a "undefined" "$B/paper.log" || true
ls -la "$P"
