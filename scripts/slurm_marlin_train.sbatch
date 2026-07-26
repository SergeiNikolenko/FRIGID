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
CONFIG_NAME="${MARLIN_CONFIG_NAME:-marlin_nplib1}"
SHARED_ROOT="${MARLIN_SHARED_ROOT:-/mnt/netstorage/nikolenko/marlin}"
OUTPUT_ROOT="${MARLIN_OUTPUT_ROOT:-}"
MAX_STEPS="${MARLIN_MAX_STEPS:-}"
CHECKPOINT_INTERVAL="${MARLIN_CHECKPOINT_INTERVAL:-}"
RESUME_CHECKPOINT="${MARLIN_RESUME_CHECKPOINT:-}"
RESUME_WEIGHTS_ONLY_CHECKPOINT="${MARLIN_RESUME_WEIGHTS_ONLY_CHECKPOINT:-}"
FRIGID_WARM_START_CHECKPOINT="${MARLIN_FRIGID_WARM_START_CHECKPOINT:-}"
FRIGID_WARM_START_SHA256="${MARLIN_FRIGID_WARM_START_SHA256:-}"
METRIC_INTERVAL="${MARLIN_METRIC_INTERVAL:-}"
EOS_LOSS_WEIGHT="${MARLIN_EOS_LOSS_WEIGHT:-}"
EOS_MASK_PROBABILITY="${MARLIN_EOS_MASK_PROBABILITY:-}"
BALANCED_TOKEN_LOSS_ALPHA="${MARLIN_BALANCED_TOKEN_LOSS_ALPHA:-}"
TOKEN_LOSS_WEIGHT_MAX="${MARLIN_TOKEN_LOSS_WEIGHT_MAX:-}"
FULL_SEQUENCE_MASK_PROBABILITY="${MARLIN_FULL_SEQUENCE_MASK_PROBABILITY:-}"
VERIFY_SNAPSHOT_HASHES="${MARLIN_VERIFY_SNAPSHOT_HASHES:-}"
BATCH_SIZE="${MARLIN_BATCH_SIZE:-}"
NUM_WORKERS="${MARLIN_NUM_WORKERS:-}"
SHUFFLE_BUFFER="${MARLIN_SHUFFLE_BUFFER:-}"
ACCUMULATE_GRAD_BATCHES="${MARLIN_ACCUMULATE_GRAD_BATCHES:-}"
METADATA_CSV="${MARLIN_METADATA_CSV:-}"
METADATA_EXCLUDE_INCHIKEYS="${MARLIN_METADATA_EXCLUDE_INCHIKEYS:-}"
LEARNING_RATE="${MARLIN_LEARNING_RATE:-}"
MOLECULAR_VALIDATION_CSV="${MARLIN_MOLECULAR_VALIDATION_CSV:-}"
MOLECULAR_VALIDATION_INTERVAL="${MARLIN_MOLECULAR_VALIDATION_INTERVAL:-}"
MOLECULAR_VALIDATION_SAMPLES="${MARLIN_MOLECULAR_VALIDATION_SAMPLES:-}"
MOLECULAR_VALIDATION_CANDIDATES="${MARLIN_MOLECULAR_VALIDATION_CANDIDATES:-}"
MOLECULAR_VALIDATION_TEMPERATURE="${MARLIN_MOLECULAR_VALIDATION_TEMPERATURE:-}"
export PYTHONPATH="$CODE/src:$ROOT/vendor"
export HF_HOME="${MARLIN_HF_HOME:-$SHARED_ROOT/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

source "$CODE/scripts/marlin_runtime_env.sh"

cd "$CODE"
TRAIN_ARGS=()
append_override() {
    local key="$1"
    local value="$2"
    if [[ -n "$value" ]]; then
        TRAIN_ARGS+=("$key=$value")
    fi
}
append_override trainer.max_steps "$MAX_STEPS"
append_override trainer.accumulate_grad_batches "$ACCUMULATE_GRAD_BATCHES"
append_override loader.batch_size "$BATCH_SIZE"
append_override loader.num_workers "$NUM_WORKERS"
append_override data.shuffle_buffer "$SHUFFLE_BUFFER"
append_override output.checkpoint_interval "$CHECKPOINT_INTERVAL"
append_override training.metric_interval "$METRIC_INTERVAL"
append_override training.eos_loss_weight "$EOS_LOSS_WEIGHT"
append_override training.eos_mask_probability "$EOS_MASK_PROBABILITY"
append_override training.balanced_token_loss_alpha "$BALANCED_TOKEN_LOSS_ALPHA"
append_override training.token_loss_weight_max "$TOKEN_LOSS_WEIGHT_MAX"
append_override training.full_sequence_mask_probability "$FULL_SEQUENCE_MASK_PROBABILITY"
append_override training.molecular_validation_interval "$MOLECULAR_VALIDATION_INTERVAL"
append_override training.molecular_validation_samples "$MOLECULAR_VALIDATION_SAMPLES"
append_override training.molecular_validation_candidates "$MOLECULAR_VALIDATION_CANDIDATES"
append_override training.molecular_validation_temperature "$MOLECULAR_VALIDATION_TEMPERATURE"
append_override data.verify_snapshot_hashes "$VERIFY_SNAPSHOT_HASHES"
if [[ -n "$OUTPUT_ROOT" ]]; then
    escaped_output_root="${OUTPUT_ROOT//=/\\=}"
    TRAIN_ARGS+=("output.root=$escaped_output_root")
    TRAIN_ARGS+=("output.checkpoints=$escaped_output_root/checkpoints")
fi
if [[ -n "$LEARNING_RATE" ]]; then
    TRAIN_ARGS+=("optim.learning_rate=$LEARNING_RATE")
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    TRAIN_ARGS+=("resume_checkpoint=${RESUME_CHECKPOINT//=/\\=}")
fi
if [[ -n "$RESUME_WEIGHTS_ONLY_CHECKPOINT" ]]; then
    TRAIN_ARGS+=("resume_weights_only_checkpoint=${RESUME_WEIGHTS_ONLY_CHECKPOINT//=/\\=}")
    TRAIN_ARGS+=("resume_checkpoint=null")
fi
if [[ -n "$FRIGID_WARM_START_CHECKPOINT" ]]; then
    if [[ -z "$FRIGID_WARM_START_SHA256" ]]; then
        echo "MARLIN_FRIGID_WARM_START_SHA256 is required" >&2
        exit 2
    fi
    TRAIN_ARGS+=("frigid_warm_start_checkpoint=${FRIGID_WARM_START_CHECKPOINT//=/\\=}")
    TRAIN_ARGS+=("frigid_warm_start_sha256=$FRIGID_WARM_START_SHA256")
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
"$PYTHON" scripts/train_marlin.py --config-name "$CONFIG_NAME" "${TRAIN_ARGS[@]}"
