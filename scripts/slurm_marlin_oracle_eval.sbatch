#!/bin/bash
#SBATCH --job-name=marlin-oracle-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/oracle-eval-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
CHECKPOINT="${MARLIN_CHECKPOINT:?Set MARLIN_CHECKPOINT}"
METADATA="${MARLIN_METADATA:?Set MARLIN_METADATA}"
FINGERPRINTS="${MARLIN_FINGERPRINTS:?Set MARLIN_FINGERPRINTS}"
FINGERPRINT_KEY="${MARLIN_FINGERPRINT_KEY:-ground_truth}"
OUTPUT_DIR="${MARLIN_OUTPUT_DIR:?Set MARLIN_OUTPUT_DIR}"
CANDIDATES="${MARLIN_CANDIDATES:-16}"
MAX_SPECTRA="${MARLIN_MAX_SPECTRA:-4}"
DIVERSITY_DROPOUT="${MARLIN_DIVERSITY_DROPOUT:-0.3}"
TEMPERATURE="${MARLIN_TEMPERATURE:-1.0}"
NO_EMA="${MARLIN_NO_EMA:-0}"
DISABLE_GRAMMAR="${MARLIN_DISABLE_GRAMMAR:-0}"
DISABLE_MASS_SHELL="${MARLIN_DISABLE_MASS_SHELL:-0}"
GENERATION_MODE="${MARLIN_GENERATION_MODE:-block}"
CLEARML_PROJECT="${MARLIN_CLEARML_PROJECT:-}"
CLEARML_TASK_NAME="${MARLIN_CLEARML_TASK_NAME:-}"
CLEARML_TAGS="${MARLIN_CLEARML_TAGS:-marlin,evaluation,oracle}"
source "$CODE/scripts/marlin_runtime_env.sh"

export PYTHONPATH="$CODE/src:$ROOT/vendor"
export TOKENIZERS_PARALLELISM=false
cd "$CODE"

ARGS=(
    --checkpoint "$CHECKPOINT"
    --tokenizer "$ROOT/data/safe-gpt/tokenizer.json"
    --metadata "$METADATA"
    --fingerprints "$FINGERPRINTS"
    --fingerprint-key "$FINGERPRINT_KEY"
    --lane dreams
    --output-dir "$OUTPUT_DIR"
    --candidates "$CANDIDATES"
    --max-spectra "$MAX_SPECTRA"
    --diversity-dropout "$DIVERSITY_DROPOUT"
    --temperature "$TEMPERATURE"
    --generation-mode "$GENERATION_MODE"
    --fix-safe-decode
)
if [[ "$NO_EMA" == "1" ]]; then
    ARGS+=(--no-ema)
fi
if [[ "$DISABLE_GRAMMAR" == "1" ]]; then
    ARGS+=(--disable-grammar-mask)
fi
if [[ "$DISABLE_MASS_SHELL" == "1" ]]; then
    ARGS+=(--disable-mass-shell)
fi
if [[ -n "$CLEARML_PROJECT" || -n "$CLEARML_TASK_NAME" ]]; then
    : "${CLEARML_PROJECT:?Set MARLIN_CLEARML_PROJECT when enabling ClearML}"
    CLEARML_TASK_NAME="${CLEARML_TASK_NAME:-marlin-oracle-eval}"
    ARGS+=(
        --clearml-project "$CLEARML_PROJECT"
        --clearml-task-name "$CLEARML_TASK_NAME-${SLURM_JOB_ID:-local}"
    )
    IFS=',' read -r -a CLEARML_TAG_ARRAY <<< "$CLEARML_TAGS"
    for clearml_tag in "${CLEARML_TAG_ARRAY[@]}"; do
        if [[ -n "$clearml_tag" ]]; then
            ARGS+=(--clearml-tag "$clearml_tag")
        fi
    done
fi

"$PYTHON" scripts/evaluate_marlin_nplib1.py "${ARGS[@]}"
