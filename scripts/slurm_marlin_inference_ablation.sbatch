#!/bin/bash
#SBATCH --job-name=marlin-inference-ablation
#SBATCH --partition=gpu-shared
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/marlin-inference-ablation-%j.out

set -euo pipefail

PROJECT_ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="${MARLIN_CODE_ROOT:-$PROJECT_ROOT/code}"
RUNTIME_ROOT=/mnt/netstorage/nikolenko/marlin/cache/runtime-inputs-spectrum-v1/16b1af5276034c041e85a4b7c43129a790b4fc091826485b691c93f9f7b699b3
CHECKPOINT="${MARLIN_CHECKPOINT:?MARLIN_CHECKPOINT is required}"
OUTPUT_DIR="${MARLIN_OUTPUT_DIR:?MARLIN_OUTPUT_DIR is required}"
GENERATION_MODE="${MARLIN_GENERATION_MODE:-block}"
DISABLE_MASS_SHELL="${MARLIN_DISABLE_MASS_SHELL:-0}"
PPM_TOLERANCE="${MARLIN_PPM_TOLERANCE:-10.0}"
FINGERPRINT_THRESHOLD="${MARLIN_FINGERPRINT_THRESHOLD:-0.95}"

cd "$CODE"
test -f "$CHECKPOINT"
test ! -e "$OUTPUT_DIR"
export ROOT="$CODE"
export MARLIN_PYTHON="$CODE/.venv/bin/python"
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
source "$CODE/scripts/marlin_runtime_env.sh"

ARGS=(
  scripts/evaluate_marlin_nplib1.py
  --checkpoint "$CHECKPOINT"
  --tokenizer "$RUNTIME_ROOT/tokenizer.json"
  --metadata "$RUNTIME_ROOT/val/metadata.csv"
  --fingerprints "$RUNTIME_ROOT/val/dreams_predictions.npz"
  --fingerprint-key probs
  --lane dreams
  --output-dir "$OUTPUT_DIR"
  --candidates 16
  --diversity-dropout 0.3
  --temperature 1.0
  --generation-mode "$GENERATION_MODE"
  --ppm-tolerance "$PPM_TOLERANCE"
  --seed 42
  --max-spectra 4
  --spec-manifest "$CODE/configs/benchmarks/nplib1_v1/nplib1_val_micro4_v1.tsv"
  --threshold "$FINGERPRINT_THRESHOLD"
  --sample-tokens
)
if [[ "$DISABLE_MASS_SHELL" == "1" ]]; then
  ARGS+=(--disable-mass-shell)
fi

"$MARLIN_PYTHON" -X faulthandler "${ARGS[@]}"
