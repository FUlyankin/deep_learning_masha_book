#!/bin/bash
# Сборка картинок листочка: tex -> pdf (xelatex) -> png (pdftoppm, poppler).
#   ./build.sh            — собрать все .tex в папке
#   ./build.sh img01_regr — собрать только указанные (без расширения)
set -u
cd "$(dirname "$0")"

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
    for f in *.tex; do targets+=("${f%.tex}"); done
fi

for name in "${targets[@]}"; do
    echo "=== $name"
    xelatex -interaction=nonstopmode "$name.tex" >/dev/null
    if [ ! -f "$name.pdf" ]; then
        echo "!!! $name: PDF не собрался, смотри $name.log"
        continue
    fi
    pdftoppm -png -r 600 -singlefile "$name.pdf" "$name"
done

rm -f ./*.log ./*.aux
