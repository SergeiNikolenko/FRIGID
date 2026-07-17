#!/bin/bash
#SBATCH --job-name=marlin-dreams
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/dreams-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
DREAMS_RUNTIME=/home/nikolenko/work/Projects/mist_molforge_autoresearch
PYTHON="$DREAMS_RUNTIME/.venv/bin/python"
export PYTHONPATH="$CODE/src:$DREAMS_RUNTIME/src"

for split in train val test; do
  "$PYTHON" "$CODE/scripts/export_dreams_fingerprint_bundle.py" \
    --mgf "$ROOT/data/processed/$split/spectra.mgf" \
    --metadata "$ROOT/data/processed/$split/metadata.csv" \
    --fingerprints "$ROOT/data/processed/$split/fingerprints.npz" \
    --output "$ROOT/runs/dreams/$split.npz" \
    --batch-size 128
done

"$PYTHON" "$CODE/scripts/train_frozen_encoder_probe.py" \
  --train-npz "$ROOT/runs/dreams/train.npz" \
  --validation-npz "$ROOT/runs/dreams/val.npz" \
  --output-dir "$ROOT/runs/dreams/probe" \
  --device cuda
