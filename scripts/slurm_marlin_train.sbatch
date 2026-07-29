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
CODE="${MARLIN_CODE_ROOT:-$ROOT/code}"
CONFIG_NAME="${MARLIN_CONFIG_NAME:-marlin_nplib1}"
SHARED_ROOT="${MARLIN_SHARED_ROOT:-/mnt/netstorage/nikolenko/marlin}"
OUTPUT_ROOT="${MARLIN_OUTPUT_ROOT:-}"
MAX_STEPS="${MARLIN_MAX_STEPS:-}"
CHECKPOINT_INTERVAL="${MARLIN_CHECKPOINT_INTERVAL:-}"
RESUME_CHECKPOINT="${MARLIN_RESUME_CHECKPOINT:-}"
INITIAL_WEIGHTS_CHECKPOINT="${MARLIN_INITIAL_WEIGHTS_CHECKPOINT:-}"
INITIAL_WEIGHTS_ARCHITECTURE_UPGRADE="${MARLIN_INITIAL_WEIGHTS_ARCHITECTURE_UPGRADE:-}"
INITIAL_WEIGHTS_USE_EMA="${MARLIN_INITIAL_WEIGHTS_USE_EMA:-}"
FRIGID_WARM_START_CHECKPOINT="${MARLIN_FRIGID_WARM_START_CHECKPOINT:-}"
FRIGID_WARM_START_SHA256="${MARLIN_FRIGID_WARM_START_SHA256:-}"
METRIC_INTERVAL="${MARLIN_METRIC_INTERVAL:-}"
EOS_LOSS_WEIGHT="${MARLIN_EOS_LOSS_WEIGHT:-}"
EOS_MASK_PROBABILITY="${MARLIN_EOS_MASK_PROBABILITY:-}"
FULL_SEQUENCE_MASK_PROBABILITY="${MARLIN_FULL_SEQUENCE_MASK_PROBABILITY:-}"
CONDITIONING_ONLY_STEPS="${MARLIN_CONDITIONING_ONLY_STEPS:-}"
CROSS_ATTENTION_ONLY_STEPS="${MARLIN_CROSS_ATTENTION_ONLY_STEPS:-}"
BALANCED_TOKEN_LOSS_ALPHA="${MARLIN_BALANCED_TOKEN_LOSS_ALPHA:-}"
TOKEN_LOSS_WEIGHT_MAX="${MARLIN_TOKEN_LOSS_WEIGHT_MAX:-}"
LAYER0_LONG_RESIDUAL_SCALE="${MARLIN_LAYER0_LONG_RESIDUAL_SCALE:-}"
FINGERPRINT_LAYER_NORM="${MARLIN_FINGERPRINT_LAYER_NORM:-}"
FINGERPRINT_SELF_ATTENTION_LAYERS="${MARLIN_FINGERPRINT_SELF_ATTENTION_LAYERS:-}"
FRIGID_COMPATIBLE_LAYER_ORDER="${MARLIN_FRIGID_COMPATIBLE_LAYER_ORDER:-}"
MOLECULAR_EVAL_ENABLED="${MARLIN_MOLECULAR_EVAL_ENABLED:-}"
MOLECULAR_EVAL_INTERVAL="${MARLIN_MOLECULAR_EVAL_INTERVAL:-}"
VERIFY_SNAPSHOT_HASHES="${MARLIN_VERIFY_SNAPSHOT_HASHES:-}"
BATCH_SIZE="${MARLIN_BATCH_SIZE:-}"
NUM_WORKERS="${MARLIN_NUM_WORKERS:-}"
SHUFFLE_BUFFER="${MARLIN_SHUFFLE_BUFFER:-}"
FILTERED_PREFIX_ONLY="${MARLIN_FILTERED_PREFIX_ONLY:-}"
ACCUMULATE_GRAD_BATCHES="${MARLIN_ACCUMULATE_GRAD_BATCHES:-}"
METADATA_CSV="${MARLIN_METADATA_CSV:-}"
METADATA_CSV_SHA256="${MARLIN_METADATA_CSV_SHA256:-}"
METADATA_EXCLUDE_INCHIKEYS="${MARLIN_METADATA_EXCLUDE_INCHIKEYS:-}"
LEARNING_RATE="${MARLIN_LEARNING_RATE:-}"
export PYTHONPATH="$CODE/src:$ROOT/vendor"
export MARLIN_REPRO_ROOT="${MARLIN_REPRO_ROOT:-$ROOT}"
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
if [[ -n "$FILTERED_PREFIX_ONLY" ]]; then
    TRAIN_ARGS+=("+data.filtered_prefix_only=$FILTERED_PREFIX_ONLY")
fi
append_override output.checkpoint_interval "$CHECKPOINT_INTERVAL"
append_override training.metric_interval "$METRIC_INTERVAL"
if [[ -n "$EOS_LOSS_WEIGHT" ]]; then
    TRAIN_ARGS+=("+training.eos_loss_weight=$EOS_LOSS_WEIGHT")
fi
if [[ -n "$EOS_MASK_PROBABILITY" ]]; then
    TRAIN_ARGS+=("+training.eos_mask_probability=$EOS_MASK_PROBABILITY")
fi
if [[ -n "$FULL_SEQUENCE_MASK_PROBABILITY" ]]; then
    TRAIN_ARGS+=(
        "+training.full_sequence_mask_probability=$FULL_SEQUENCE_MASK_PROBABILITY"
    )
fi
if [[ -n "$CONDITIONING_ONLY_STEPS" ]]; then
    TRAIN_ARGS+=(
        "+training.conditioning_only_steps=$CONDITIONING_ONLY_STEPS"
    )
fi
if [[ -n "$CROSS_ATTENTION_ONLY_STEPS" ]]; then
    TRAIN_ARGS+=(
        "+training.cross_attention_only_steps=$CROSS_ATTENTION_ONLY_STEPS"
    )
fi
if [[ -n "$BALANCED_TOKEN_LOSS_ALPHA" ]]; then
    TRAIN_ARGS+=(
        "+training.balanced_token_loss_alpha=$BALANCED_TOKEN_LOSS_ALPHA"
    )
fi
if [[ -n "$TOKEN_LOSS_WEIGHT_MAX" ]]; then
    TRAIN_ARGS+=("+training.token_loss_weight_max=$TOKEN_LOSS_WEIGHT_MAX")
fi
if [[ -n "$LAYER0_LONG_RESIDUAL_SCALE" ]]; then
    TRAIN_ARGS+=("+model.layer0_long_residual_scale=$LAYER0_LONG_RESIDUAL_SCALE")
fi
if [[ -n "$FINGERPRINT_LAYER_NORM" ]]; then
    TRAIN_ARGS+=("model.fingerprint_layer_norm=$FINGERPRINT_LAYER_NORM")
fi
if [[ -n "$FINGERPRINT_SELF_ATTENTION_LAYERS" ]]; then
    TRAIN_ARGS+=(
        "model.fingerprint_self_attention_layers=$FINGERPRINT_SELF_ATTENTION_LAYERS"
    )
fi
if [[ -n "$FRIGID_COMPATIBLE_LAYER_ORDER" ]]; then
    TRAIN_ARGS+=(
        "model.frigid_compatible_layer_order=$FRIGID_COMPATIBLE_LAYER_ORDER"
    )
fi
append_override evaluation.enabled "$MOLECULAR_EVAL_ENABLED"
append_override evaluation.interval_steps "$MOLECULAR_EVAL_INTERVAL"
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
    TRAIN_ARGS+=("frigid_warm_start_checkpoint=null")
    TRAIN_ARGS+=("frigid_warm_start_sha256=null")
fi
if [[ -n "$INITIAL_WEIGHTS_CHECKPOINT" ]]; then
    TRAIN_ARGS+=(
        "initial_weights_checkpoint=${INITIAL_WEIGHTS_CHECKPOINT//=/\\=}"
    )
    TRAIN_ARGS+=("resume_checkpoint=null")
    TRAIN_ARGS+=("frigid_warm_start_checkpoint=null")
    TRAIN_ARGS+=("frigid_warm_start_sha256=null")
fi
if [[ -n "$INITIAL_WEIGHTS_ARCHITECTURE_UPGRADE" ]]; then
    TRAIN_ARGS+=(
        "initial_weights_architecture_upgrade=$INITIAL_WEIGHTS_ARCHITECTURE_UPGRADE"
    )
fi
if [[ -n "$INITIAL_WEIGHTS_USE_EMA" ]]; then
    TRAIN_ARGS+=("initial_weights_use_ema=$INITIAL_WEIGHTS_USE_EMA")
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
    if [[ -z "$METADATA_CSV_SHA256" ]]; then
        echo "MARLIN_METADATA_CSV_SHA256 is required with MARLIN_METADATA_CSV" >&2
        exit 2
    fi
    TRAIN_ARGS+=("++data.metadata_csv=$METADATA_CSV")
    TRAIN_ARGS+=("++data.metadata_csv_sha256=$METADATA_CSV_SHA256")
elif [[ -n "$METADATA_CSV_SHA256" ]]; then
    echo "MARLIN_METADATA_CSV is required with MARLIN_METADATA_CSV_SHA256" >&2
    exit 2
fi
if [[ -n "$METADATA_EXCLUDE_INCHIKEYS" ]]; then
    TRAIN_ARGS+=("+data.metadata_exclude_inchikeys=$METADATA_EXCLUDE_INCHIKEYS")
fi
"$PYTHON" scripts/train_marlin.py --config-name "$CONFIG_NAME" "${TRAIN_ARGS[@]}"
