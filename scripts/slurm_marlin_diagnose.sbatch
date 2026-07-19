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
export PYTHONPATH="$CODE/src"
cd "$CODE"

/home/nikolenko/work/Projects/FRIGID/.venv/bin/python scripts/diagnose_marlin_checkpoint.py \
  --checkpoint "$ROOT/runs/decoder/checkpoints/step=100000.ckpt" \
  --tokenizer "$ROOT/data/safe-gpt/tokenizer.json" \
  --metadata "$ROOT/data/processed/test/metadata.csv" \
  --output "$ROOT/runs/evaluation/diagnostics/teacher_forced_row0.json"
