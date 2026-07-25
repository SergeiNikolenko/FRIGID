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
OUTPUT_DIR="${MARLIN_OUTPUT_DIR:?Set MARLIN_OUTPUT_DIR}"
CANDIDATES="${MARLIN_CANDIDATES:-16}"
MAX_SPECTRA="${MARLIN_MAX_SPECTRA:-4}"
DIVERSITY_DROPOUT="${MARLIN_DIVERSITY_DROPOUT:-0.3}"
NO_EMA="${MARLIN_NO_EMA:-0}"
DISABLE_GRAMMAR="${MARLIN_DISABLE_GRAMMAR:-0}"
DISABLE_MASS_SHELL="${MARLIN_DISABLE_MASS_SHELL:-0}"
GENERATION_MODE="${MARLIN_GENERATION_MODE:-block}"

export PYTHONPATH="$CODE/src:$ROOT/vendor"
export TOKENIZERS_PARALLELISM=false
cd "$CODE"

ARGS=(
    --checkpoint "$CHECKPOINT"
    --tokenizer "$ROOT/data/safe-gpt/tokenizer.json"
    --metadata "$METADATA"
    --fingerprints "$FINGERPRINTS"
    --fingerprint-key oracle
    --lane dreams
    --output-dir "$OUTPUT_DIR"
    --candidates "$CANDIDATES"
    --max-spectra "$MAX_SPECTRA"
    --diversity-dropout "$DIVERSITY_DROPOUT"
    --temperature 1.0
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

"$ROOT/.venv/bin/python" scripts/evaluate_marlin_nplib1.py "${ARGS[@]}"
