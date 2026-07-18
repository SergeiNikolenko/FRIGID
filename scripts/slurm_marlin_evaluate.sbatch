#!/bin/bash
#SBATCH --job-name=marlin-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/eval-%A_%a.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
PYTHON=/home/nikolenko/work/Projects/FRIGID/.venv/bin/python
CHECKPOINT="${MARLIN_CHECKPOINT:-$ROOT/runs/decoder/checkpoints/step=100000.ckpt}"
OUTPUT_TAG="${MARLIN_OUTPUT_TAG:-full}"
MAX_SPECTRA="${MARLIN_MAX_SPECTRA:-}"
export PYTHONPATH="$CODE/src"
export TOKENIZERS_PARALLELISM=false

cd "$CODE"

COMMON_ARGS=(
  --checkpoint "$CHECKPOINT"
  --tokenizer "$ROOT/data/safe-gpt/tokenizer.json"
  --metadata "$ROOT/data/processed/test/metadata.csv"
  --candidates 384
  --diversity-dropout 0.3
  --ppm-tolerance 10
  --valence-slack 4
)
if [[ -n "$MAX_SPECTRA" ]]; then
  COMMON_ARGS+=(--max-spectra "$MAX_SPECTRA")
fi

if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  "$PYTHON" scripts/evaluate_marlin_nplib1.py \
    "${COMMON_ARGS[@]}" \
    --lane dreams \
    --fingerprints "$ROOT/runs/dreams/probe/test_predictions.npz" \
    --fingerprint-key probs \
    --threshold 0.95 \
    --output-dir "$ROOT/runs/evaluation/$OUTPUT_TAG/dreams"
else
  "$PYTHON" scripts/evaluate_marlin_nplib1.py \
    "${COMMON_ARGS[@]}" \
    --lane mist \
    --fingerprints "$ROOT/runs/mist/test/fingerprints.npz" \
    --fingerprint-key mist_binary \
    --output-dir "$ROOT/runs/evaluation/$OUTPUT_TAG/mist"
fi
