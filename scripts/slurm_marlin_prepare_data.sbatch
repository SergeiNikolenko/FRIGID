#!/bin/bash
#SBATCH --job-name=marlin-prepare
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/prepare-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
DATA="$ROOT/data/nplib1/canopus_train_export"
PYTHON=/home/nikolenko/work/Projects/FRIGID/.venv/bin/python
export PYTHONPATH="$CODE/src:$CODE/scripts:$ROOT/vendor"

for split in train val test; do
  "$PYTHON" "$CODE/scripts/prepare_marlin_nplib1.py" \
    --data-dir "$DATA" \
    --split "$split" \
    --output-dir "$ROOT/data/processed/$split"
done

"$PYTHON" - <<'PY'
from pathlib import Path

import pandas as pd

root = Path("/home/nikolenko/work/Projects/MARLIN_reproduction_20260717")
metadata = pd.read_csv(root / "data/processed/test/metadata.csv")
metadata[["inchikey"]].drop_duplicates().to_csv(
    root / "data/nplib1_test_inchikeys.csv", index=False
)
PY

find "$ROOT/data/processed" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$ROOT/manifests/processed_data.sha256"
sha256sum "$ROOT/data/nplib1_test_inchikeys.csv" \
  > "$ROOT/manifests/nplib1_test_inchikeys.sha256"
