#!/bin/bash
#SBATCH --job-name=marlin-mces
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/mces-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
PYTHON=/home/nikolenko/work/Projects/FRIGID/.venv/bin/python
: "${MARLIN_PREDICTIONS:?set MARLIN_PREDICTIONS to predictions.jsonl}"
: "${MARLIN_MCES_OUTPUT:?set MARLIN_MCES_OUTPUT to the output directory}"

cd "$CODE"
PYTHONPATH="$CODE/src:$ROOT/vendor" "$PYTHON" scripts/evaluate_marlin_mces.py \
    --predictions "$MARLIN_PREDICTIONS" \
    --output-dir "$MARLIN_MCES_OUTPUT" \
    --workers "${SLURM_CPUS_PER_TASK:-12}"
