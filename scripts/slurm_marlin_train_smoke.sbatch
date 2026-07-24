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
SMOKE_ROOT="$ROOT/runs/train-smoke-$SLURM_JOB_ID"
SMOKE_BATCH_SIZE="${MARLIN_SMOKE_BATCH_SIZE:-1}"
SMOKE_MAX_STEPS="${MARLIN_SMOKE_MAX_STEPS:-1}"
SMOKE_SHUFFLE_BUFFER="${MARLIN_SMOKE_SHUFFLE_BUFFER:-128}"
export PYTHONPATH="$CODE/src:$ROOT/vendor"
export HF_HOME="$ROOT/cache/huggingface"
export TOKENIZERS_PARALLELISM=false

source "$CODE/scripts/marlin_runtime_env.sh"

cd "$CODE"
"$PYTHON" scripts/train_marlin.py \
  loader.batch_size="$SMOKE_BATCH_SIZE" \
  loader.num_workers=0 \
  data.shuffle_buffer="$SMOKE_SHUFFLE_BUFFER" \
  data.verify_snapshot_hashes=false \
  trainer.max_steps="$SMOKE_MAX_STEPS" \
  trainer.accumulate_grad_batches=1 \
  trainer.log_every_n_steps=1 \
  output.root="$SMOKE_ROOT" \
  output.checkpoints="$SMOKE_ROOT/checkpoints" \
  output.checkpoint_interval="$SMOKE_MAX_STEPS"
