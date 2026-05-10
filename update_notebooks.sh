#!/usr/bin/bash

# Runner to update all notebooks that are registered and tracked by py files.
# Maintenance for these files is tracked through the py files and the notebooks
# are updated/rendered via jupytext. (Maintenance is usually changing a few
# lines as a chore rather than writing the whole notebook from scratch, and
# usually it is adding a few lines to an existing notebook to demonstrate how a
# feature changed.)

pyfiles="$(ls notebooks/*.py)"

for file in $pyfiles
do
  # TQDM prints on standard error, which jupyter handles well in a notebook but
  # does not look pleasant if the notebooks are run programmatically like this.
  TQDM_DISABLE=1 PYTHONWARNINGS=ignore uv run jupytext --to ipynb --execute --set-kernel python3 $file
done
