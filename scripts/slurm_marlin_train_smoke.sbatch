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
SHARED_ROOT="${MARLIN_SHARED_ROOT:-/mnt/netstorage/nikolenko/marlin}"
SMOKE_ROOT="${MARLIN_SMOKE_ROOT:-$SHARED_ROOT/runs/train-smoke-$SLURM_JOB_ID}"
SMOKE_BATCH_SIZE="${MARLIN_SMOKE_BATCH_SIZE:-1}"
SMOKE_MAX_STEPS="${MARLIN_SMOKE_MAX_STEPS:-1}"
SMOKE_SHUFFLE_BUFFER="${MARLIN_SMOKE_SHUFFLE_BUFFER:-128}"
FRIGID_WARM_START_CHECKPOINT="${MARLIN_FRIGID_WARM_START_CHECKPOINT:-}"
FRIGID_WARM_START_SHA256="${MARLIN_FRIGID_WARM_START_SHA256:-}"
export PYTHONPATH="$CODE/src:$ROOT/vendor"
export HF_HOME="$ROOT/cache/huggingface"
export TOKENIZERS_PARALLELISM=false

source "$CODE/scripts/marlin_runtime_env.sh"

cd "$CODE"
TRAIN_ARGS=(
    "loader.batch_size=$SMOKE_BATCH_SIZE"
    "loader.num_workers=0"
    "data.shuffle_buffer=$SMOKE_SHUFFLE_BUFFER"
    "data.verify_snapshot_hashes=false"
    "trainer.max_steps=$SMOKE_MAX_STEPS"
    "trainer.accumulate_grad_batches=1"
    "trainer.log_every_n_steps=1"
    "output.root=$SMOKE_ROOT"
    "output.checkpoints=$SMOKE_ROOT/checkpoints"
    "output.checkpoint_interval=$SMOKE_MAX_STEPS"
)
if [[ -n "$FRIGID_WARM_START_CHECKPOINT" ]]; then
    if [[ -z "$FRIGID_WARM_START_SHA256" ]]; then
        echo "MARLIN_FRIGID_WARM_START_SHA256 is required" >&2
        exit 2
    fi
    TRAIN_ARGS+=("frigid_warm_start_checkpoint=${FRIGID_WARM_START_CHECKPOINT//=/\\=}")
    TRAIN_ARGS+=("frigid_warm_start_sha256=$FRIGID_WARM_START_SHA256")
fi
"$PYTHON" scripts/train_marlin.py "${TRAIN_ARGS[@]}"
