#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR/paper"

if command -v bibtex8 >/dev/null 2>&1; then
  BIBTEX_CMD='bibtex8 %O %B'
else
  BIBTEX_CMD='bibtex %O %B'
fi

latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -e "\$bibtex = '$BIBTEX_CMD';" main.tex
latexmk -c main.tex
rm -f main.blg main.run.xml main.synctex.gz
