#!/bin/bash
#SBATCH --job-name=marlin-train-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/train-smoke-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
PYTHON=/home/nikolenko/work/Projects/FRIGID/.venv/bin/python
export PYTHONPATH="$CODE/src"
export HF_HOME="$ROOT/cache/huggingface"
export TOKENIZERS_PARALLELISM=false

cd "$CODE"
"$PYTHON" scripts/train_marlin.py \
  data.exclude_inchikeys=null \
  loader.batch_size=1 \
  loader.num_workers=0 \
  trainer.max_steps=1 \
  trainer.accumulate_grad_batches=1 \
  output.root="$ROOT/runs/train-smoke" \
  output.checkpoints="$ROOT/runs/train-smoke/checkpoints" \
  output.checkpoint_interval=1
