#!/bin/bash
#SBATCH --job-name=expanding-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/expanding-smoke-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
SHARED_ROOT="${MARLIN_SHARED_ROOT:-/mnt/netstorage/nikolenko/marlin}"
SMOKE_ROOT="${EXPANDING_SMOKE_ROOT:-$SHARED_ROOT/runs/expanding/smoke-$SLURM_JOB_ID}"
export PYTHONPATH="$CODE/src:$ROOT/vendor"
export HF_HOME="${MARLIN_HF_HOME:-$SHARED_ROOT/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

source "$CODE/scripts/marlin_runtime_env.sh"

cd "$CODE"
"$PYTHON" scripts/train_expanding_marlin.py \
    allow_dirty_worktree=true \
    loader.batch_size="${EXPANDING_SMOKE_BATCH_SIZE:-1}" \
    loader.num_workers=0 \
    data.shuffle_buffer=128 \
    data.verify_snapshot_hashes=false \
    trainer.max_steps="${EXPANDING_SMOKE_MAX_STEPS:-1}" \
    trainer.accumulate_grad_batches=1 \
    trainer.log_every_n_steps=1 \
    evaluation.enabled=false \
    tracking.clearml.enabled="${EXPANDING_SMOKE_CLEARML:-true}" \
    output.root="$SMOKE_ROOT" \
    output.checkpoints="$SMOKE_ROOT/checkpoints" \
    output.checkpoint_interval="${EXPANDING_SMOKE_MAX_STEPS:-1}"
