#!/bin/bash
#SBATCH --job-name=marlin-data
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/marlin-data-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
PYTHON=/home/nikolenko/work/Projects/FRIGID/.venv/bin/python
SNAPSHOT=/mnt/netstorage/nikolenko/marlin/safe-gpt-16d0be9ad6177ae683a32a86204530e8ee624a0f
export PYTHONPATH="$CODE/src"
export HF_HOME="$ROOT/cache/huggingface"

cd "$CODE"
"$PYTHON" scripts/materialize_marlin_training_snapshot.py \
  --dataset datamol-io/safe-gpt \
  --revision 16d0be9ad6177ae683a32a86204530e8ee624a0f \
  --output-dir "$SNAPSHOT"
