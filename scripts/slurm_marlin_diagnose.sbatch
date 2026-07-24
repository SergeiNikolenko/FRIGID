#!/bin/bash
#SBATCH --job-name=marlin-diagnose
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/diagnose-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
CHECKPOINT="${MARLIN_CHECKPOINT:-$ROOT/runs/decoder/checkpoints/step=100000.ckpt}"
OUTPUT="${MARLIN_OUTPUT:-$ROOT/runs/evaluation/diagnostics/teacher_forced_rows0_31.json}"
ROWS="${MARLIN_ROWS:-32}"
ROW="${MARLIN_ROW:-0}"
NO_EMA="${MARLIN_NO_EMA:-0}"
export PYTHONPATH="$CODE/src:$ROOT/vendor"
cd "$CODE"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --tokenizer "$ROOT/data/safe-gpt/tokenizer.json"
  --metadata "$ROOT/data/processed/test/metadata.csv"
  --output "$OUTPUT"
  --row "$ROW"
  --rows "$ROWS"
)
if [[ "$NO_EMA" == "1" ]]; then
  ARGS+=(--no-ema)
fi

$ROOT/.venv/bin/python scripts/diagnose_marlin_checkpoint.py \
  "${ARGS[@]}"
