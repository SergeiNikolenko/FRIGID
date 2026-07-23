#!/bin/bash
#SBATCH --job-name=marlin-mist
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/mist-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
DATA="$ROOT/data/nplib1/canopus_train_export"
PYTHON=$ROOT/.venv/bin/python
CHECKPOINT=/home/nikolenko/work/Projects/FRIGID/repro_cache/mist_canopus.pt
export PYTHONPATH="$CODE/src:$CODE/scripts:$ROOT/vendor"

for split in train val test; do
  "$PYTHON" "$CODE/scripts/export_mist_fingerprints.py" \
    --config "$CODE/configs/spec2mol_benchmark_canopus.yaml" \
    --data-dir "$DATA" \
    --mist-checkpoint "$CHECKPOINT" \
    --fp-threshold 0.187 \
    --split "$split" \
    --output-dir "$ROOT/runs/mist/$split"
done

find "$ROOT/runs/mist" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$ROOT/manifests/mist_outputs.sha256"
