#!/bin/bash
set -e
cd "$(dirname "$0")"

for nb in 0*.ipynb; do
    echo "=== $nb ==="
    python3 -m jupyter nbconvert --to notebook --inplace --execute "$nb"
done
