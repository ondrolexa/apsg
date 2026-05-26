#!/bin/bash
set -e

PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(dirname "$0")/../../.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

for nb in 0*.ipynb; do
    echo "=== $nb ==="
    "$PYTHON" -m jupyter nbconvert --to notebook --inplace --execute "$nb"
done
