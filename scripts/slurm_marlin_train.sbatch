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
SHARED_ROOT="${MARLIN_SHARED_ROOT:-/mnt/netstorage/nikolenko/marlin}"
OUTPUT_ROOT="${MARLIN_OUTPUT_ROOT:-$SHARED_ROOT/runs/decoder}"
MAX_STEPS="${MARLIN_MAX_STEPS:-100000}"
CHECKPOINT_INTERVAL="${MARLIN_CHECKPOINT_INTERVAL:-5000}"
RESUME_CHECKPOINT="${MARLIN_RESUME_CHECKPOINT:-}"
RESUME_WEIGHTS_ONLY_CHECKPOINT="${MARLIN_RESUME_WEIGHTS_ONLY_CHECKPOINT:-}"
METRIC_INTERVAL="${MARLIN_METRIC_INTERVAL:-50}"
EOS_LOSS_WEIGHT="${MARLIN_EOS_LOSS_WEIGHT:-1.0}"
EOS_MASK_PROBABILITY="${MARLIN_EOS_MASK_PROBABILITY:-0.0}"
BALANCED_TOKEN_LOSS_ALPHA="${MARLIN_BALANCED_TOKEN_LOSS_ALPHA:-0.0}"
TOKEN_LOSS_WEIGHT_MAX="${MARLIN_TOKEN_LOSS_WEIGHT_MAX:-20.0}"
FULL_SEQUENCE_MASK_PROBABILITY="${MARLIN_FULL_SEQUENCE_MASK_PROBABILITY:-0.0}"
VERIFY_SNAPSHOT_HASHES="${MARLIN_VERIFY_SNAPSHOT_HASHES:-true}"
BATCH_SIZE="${MARLIN_BATCH_SIZE:-8}"
NUM_WORKERS="${MARLIN_NUM_WORKERS:-8}"
SHUFFLE_BUFFER="${MARLIN_SHUFFLE_BUFFER:-100000}"
ACCUMULATE_GRAD_BATCHES="${MARLIN_ACCUMULATE_GRAD_BATCHES:-32}"
METADATA_CSV="${MARLIN_METADATA_CSV:-}"
METADATA_EXCLUDE_INCHIKEYS="${MARLIN_METADATA_EXCLUDE_INCHIKEYS:-}"
LEARNING_RATE="${MARLIN_LEARNING_RATE:-}"
MOLECULAR_VALIDATION_CSV="${MARLIN_MOLECULAR_VALIDATION_CSV:-}"
MOLECULAR_VALIDATION_INTERVAL="${MARLIN_MOLECULAR_VALIDATION_INTERVAL:-500}"
MOLECULAR_VALIDATION_SAMPLES="${MARLIN_MOLECULAR_VALIDATION_SAMPLES:-2}"
MOLECULAR_VALIDATION_CANDIDATES="${MARLIN_MOLECULAR_VALIDATION_CANDIDATES:-4}"
MOLECULAR_VALIDATION_TEMPERATURE="${MARLIN_MOLECULAR_VALIDATION_TEMPERATURE:-1.0}"
export PYTHONPATH="$CODE/src:$ROOT/vendor"
export HF_HOME="${MARLIN_HF_HOME:-$SHARED_ROOT/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

source "$CODE/scripts/marlin_runtime_env.sh"

cd "$CODE"
TRAIN_ARGS=(
    "trainer.max_steps=$MAX_STEPS"
    "trainer.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES"
    "loader.batch_size=$BATCH_SIZE"
    "loader.num_workers=$NUM_WORKERS"
    "data.shuffle_buffer=$SHUFFLE_BUFFER"
    "output.root=$OUTPUT_ROOT"
    "output.checkpoints=$OUTPUT_ROOT/checkpoints"
    "output.checkpoint_interval=$CHECKPOINT_INTERVAL"
    "training.metric_interval=$METRIC_INTERVAL"
    "training.eos_loss_weight=$EOS_LOSS_WEIGHT"
    "training.eos_mask_probability=$EOS_MASK_PROBABILITY"
    "training.balanced_token_loss_alpha=$BALANCED_TOKEN_LOSS_ALPHA"
    "training.token_loss_weight_max=$TOKEN_LOSS_WEIGHT_MAX"
    "training.full_sequence_mask_probability=$FULL_SEQUENCE_MASK_PROBABILITY"
    "training.molecular_validation_interval=$MOLECULAR_VALIDATION_INTERVAL"
    "training.molecular_validation_samples=$MOLECULAR_VALIDATION_SAMPLES"
    "training.molecular_validation_candidates=$MOLECULAR_VALIDATION_CANDIDATES"
    "training.molecular_validation_temperature=$MOLECULAR_VALIDATION_TEMPERATURE"
    "data.verify_snapshot_hashes=$VERIFY_SNAPSHOT_HASHES"
)
if [[ -n "$LEARNING_RATE" ]]; then
    TRAIN_ARGS+=("optim.learning_rate=$LEARNING_RATE")
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    TRAIN_ARGS+=("resume_checkpoint=${RESUME_CHECKPOINT//=/\\=}")
fi
if [[ -n "$RESUME_WEIGHTS_ONLY_CHECKPOINT" ]]; then
    TRAIN_ARGS+=("+resume_weights_only_checkpoint=${RESUME_WEIGHTS_ONLY_CHECKPOINT//=/\\=}")
    TRAIN_ARGS+=("resume_checkpoint=null")
fi
if [[ -n "$METADATA_CSV" ]]; then
    TRAIN_ARGS+=("+data.metadata_csv=$METADATA_CSV")
fi
if [[ -n "$METADATA_EXCLUDE_INCHIKEYS" ]]; then
    TRAIN_ARGS+=("+data.metadata_exclude_inchikeys=$METADATA_EXCLUDE_INCHIKEYS")
fi
if [[ -n "$MOLECULAR_VALIDATION_CSV" ]]; then
    TRAIN_ARGS+=("training.molecular_validation_csv=$MOLECULAR_VALIDATION_CSV")
fi
"$PYTHON" scripts/train_marlin.py "${TRAIN_ARGS[@]}"
