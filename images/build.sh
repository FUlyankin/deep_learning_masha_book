#!/bin/bash
# ./build.sh problem_set_03/img01_gr2 [problem_set_07/img02_rnn ...]   — собрать указанные картинки
# ./build.sh problem_set_03                                           — собрать всю папку
set -u
cd "$(dirname "$0")"
targets=()
for a in "$@"; do
  if [ -d "$a" ]; then for f in "$a"/*.tex; do targets+=("${f%.tex}"); done
  else targets+=("${a%.tex}"); fi
done
for t in "${targets[@]}"; do
  d=$(dirname "$t"); n=$(basename "$t")
  echo "=== $t"
  ( cd "$d" && xelatex -interaction=nonstopmode "$n.tex" >/dev/null
    if [ -f "$n.pdf" ]; then pdftoppm -png -r 600 -singlefile "$n.pdf" "$n"; else echo "!!! $t: PDF не собрался, смотри $n.log"; fi
    rm -f "$n.aux"; [ -f "$n.pdf" ] && rm -f "$n.log" )
done
