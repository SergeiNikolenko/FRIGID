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
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    TRAIN_ARGS+=("resume_checkpoint=${RESUME_CHECKPOINT//=/\\=}")
fi
"$PYTHON" scripts/train_marlin.py "${TRAIN_ARGS[@]}"
