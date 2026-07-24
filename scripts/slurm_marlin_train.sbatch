#!/bin/bash
#SBATCH --job-name=marlin-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=30-00:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/train-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
OUTPUT_ROOT="${MARLIN_OUTPUT_ROOT:-$ROOT/runs/decoder}"
MAX_STEPS="${MARLIN_MAX_STEPS:-100000}"
CHECKPOINT_INTERVAL="${MARLIN_CHECKPOINT_INTERVAL:-5000}"
RESUME_CHECKPOINT="${MARLIN_RESUME_CHECKPOINT:-}"
METRIC_INTERVAL="${MARLIN_METRIC_INTERVAL:-50}"
EOS_LOSS_WEIGHT="${MARLIN_EOS_LOSS_WEIGHT:-1.0}"
EOS_MASK_PROBABILITY="${MARLIN_EOS_MASK_PROBABILITY:-0.0}"
VERIFY_SNAPSHOT_HASHES="${MARLIN_VERIFY_SNAPSHOT_HASHES:-true}"
export PYTHONPATH="$CODE/src:$ROOT/vendor"
export HF_HOME="$ROOT/cache/huggingface"
export TOKENIZERS_PARALLELISM=false

source "$CODE/scripts/marlin_runtime_env.sh"

cd "$CODE"
TRAIN_ARGS=(
    "trainer.max_steps=$MAX_STEPS"
    "output.root=$OUTPUT_ROOT"
    "output.checkpoints=$OUTPUT_ROOT/checkpoints"
    "output.checkpoint_interval=$CHECKPOINT_INTERVAL"
    "training.metric_interval=$METRIC_INTERVAL"
    "training.eos_loss_weight=$EOS_LOSS_WEIGHT"
    "training.eos_mask_probability=$EOS_MASK_PROBABILITY"
    "data.verify_snapshot_hashes=$VERIFY_SNAPSHOT_HASHES"
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    TRAIN_ARGS+=("resume_checkpoint=${RESUME_CHECKPOINT//=/\\=}")
fi
"$PYTHON" scripts/train_marlin.py "${TRAIN_ARGS[@]}"
