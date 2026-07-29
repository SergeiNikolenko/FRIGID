#!/bin/bash
#SBATCH --job-name=expanding-marlin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=30-00:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/expanding-train-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
SHARED_ROOT="${MARLIN_SHARED_ROOT:-/mnt/netstorage/nikolenko/marlin}"
STAGE="${EXPANDING_STAGE:-eflow}"
OUTPUT_ROOT="${EXPANDING_OUTPUT_ROOT:-$SHARED_ROOT/runs/expanding/$STAGE-$SLURM_JOB_ID}"
MAX_STEPS="${EXPANDING_MAX_STEPS:-}"
CHECKPOINT_INTERVAL="${EXPANDING_CHECKPOINT_INTERVAL:-}"
RESUME_CHECKPOINT="${EXPANDING_RESUME_CHECKPOINT:-}"
TEACHER_CHECKPOINT="${EXPANDING_TEACHER_CHECKPOINT:-}"
BATCH_SIZE="${EXPANDING_BATCH_SIZE:-}"
ACCUMULATE="${EXPANDING_ACCUMULATE_GRAD_BATCHES:-}"
NUM_WORKERS="${EXPANDING_NUM_WORKERS:-}"
EVALUATION_ENABLED="${EXPANDING_EVALUATION_ENABLED:-}"
ALLOW_DIRTY="${EXPANDING_ALLOW_DIRTY_WORKTREE:-false}"
export PYTHONPATH="$CODE/src:$ROOT/vendor"
export HF_HOME="${MARLIN_HF_HOME:-$SHARED_ROOT/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

source "$CODE/scripts/marlin_runtime_env.sh"

cd "$CODE"
ARGS=(
    "stage=$STAGE"
    "output.root=${OUTPUT_ROOT//=/\\=}"
    "output.checkpoints=${OUTPUT_ROOT//=/\\=}/checkpoints"
    "allow_dirty_worktree=$ALLOW_DIRTY"
)
append_override() {
    local key="$1"
    local value="$2"
    if [[ -n "$value" ]]; then
        ARGS+=("$key=$value")
    fi
}
append_override trainer.max_steps "$MAX_STEPS"
append_override output.checkpoint_interval "$CHECKPOINT_INTERVAL"
append_override evaluation.interval_steps "$CHECKPOINT_INTERVAL"
append_override loader.batch_size "$BATCH_SIZE"
append_override loader.num_workers "$NUM_WORKERS"
append_override trainer.accumulate_grad_batches "$ACCUMULATE"
append_override evaluation.enabled "$EVALUATION_ENABLED"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    ARGS+=("resume_checkpoint=${RESUME_CHECKPOINT//=/\\=}")
fi
if [[ "$STAGE" == "efm" ]]; then
    if [[ -z "$TEACHER_CHECKPOINT" ]]; then
        echo "EXPANDING_TEACHER_CHECKPOINT is required for EFM" >&2
        exit 2
    fi
    ARGS+=("teacher_checkpoint=${TEACHER_CHECKPOINT//=/\\=}")
    ARGS+=("frigid_warm_start_checkpoint=null")
    ARGS+=("frigid_warm_start_sha256=null")
    ARGS+=("tracking.clearml.task_name=expanding-marlin-efm")
    ARGS+=("tracking.clearml.tags=[experimental,non-paper-architecture,arxiv-2607.21585,conditional-EFM,SAFE]")
fi
"$PYTHON" scripts/train_expanding_marlin.py "${ARGS[@]}"
