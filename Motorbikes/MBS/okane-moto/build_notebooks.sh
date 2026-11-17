#!/bin/sh
# regenerate and execute the notebooks from their py:percent sources
set -e
cd "$(dirname "$0")"
for f in nbsrc/*.py; do
  n=notebooks/$(basename "${f%.py}").ipynb
  jupytext --to ipynb -o "$n" "$f"
  PYTHONPATH=. jupyter nbconvert --to notebook --execute --inplace "$n"
done
