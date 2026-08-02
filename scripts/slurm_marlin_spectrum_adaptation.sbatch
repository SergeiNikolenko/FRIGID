#!/bin/bash
#SBATCH --job-name=marlin-fp-adapt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/marlin-fp-adapt-%j.out

set -euo pipefail

PROJECT_ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="${MARLIN_CODE_ROOT:-$PROJECT_ROOT/code}"
SHARED_ROOT=/mnt/netstorage/nikolenko/marlin
RUNTIME_ROOT="$SHARED_ROOT/cache/runtime-inputs-spectrum-v1/16b1af5276034c041e85a4b7c43129a790b4fc091826485b691c93f9f7b699b3"
CHECKPOINT="$SHARED_ROOT/runs/safe-gpt-pretrain-v3/checkpoints/step=30000.ckpt"
LOCAL_ROOT="$PROJECT_ROOT/runs"
RUN_ROOT="${MARLIN_ADAPT_RUN_ROOT:-$LOCAL_ROOT/spectrum-fingerprint-adaptation-slurm-$SLURM_JOB_ID}"
EVALUATION_SPECTRA="${MARLIN_EVALUATION_SPECTRA:-32}"
EVALUATION_CANDIDATES="${MARLIN_EVALUATION_CANDIDATES:-16}"
EVALUATION_MANIFEST="${MARLIN_EVALUATION_MANIFEST:-$CODE/configs/benchmarks/nplib1_v1/nplib1_val_micro32_v1.tsv}"
TRAIN_FINGERPRINT_THRESHOLD="${MARLIN_TRAIN_FINGERPRINT_THRESHOLD:-0.90}"
ADAPT_MAX_STEPS="${MARLIN_ADAPT_MAX_STEPS:-100}"
CROSS_ATTENTION_ONLY_STEPS="${MARLIN_CROSS_ATTENTION_ONLY_STEPS:-100}"

cd "$CODE"
test -f "$RUNTIME_ROOT/manifest.json"
test -f "$CHECKPOINT"
test ! -e "$RUN_ROOT"

export MARLIN_PYTHON="$PROJECT_ROOT/code/.venv/bin/python"
export CLEARML_OFFLINE_MODE="${MARLIN_CLEARML_OFFLINE_MODE:-1}"
export CLEARML_CACHE_DIR="${MARLIN_CLEARML_CACHE_DIR:-$PROJECT_ROOT/cache/clearml-offline/$SLURM_JOB_ID}"
export PYTHONFAULTHANDLER=1
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"

ROOT="$PROJECT_ROOT/code"
source "$CODE/scripts/marlin_runtime_env.sh"

"$MARLIN_PYTHON" -X faulthandler scripts/train_marlin_spectrum_adaptation.py \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 6a44763d628c3b5b4a48178ceca034a397564ba1c56bf679594a1152692f7c82 \
  --tokenizer "$RUNTIME_ROOT/tokenizer.json" \
  --metadata "$RUNTIME_ROOT/train/metadata.csv" \
  --fingerprints "$RUNTIME_ROOT/train/dreams_predictions.npz" \
  --fingerprint-key probs \
  --fingerprint-threshold "$TRAIN_FINGERPRINT_THRESHOLD" \
  --exclude-inchikeys "$RUNTIME_ROOT/nplib1_test_inchikeys.csv" \
  --exclude-metadata "$RUNTIME_ROOT/val/metadata.csv" \
  --validation-metadata "$RUNTIME_ROOT/val/metadata.csv" \
  --validation-fingerprints "$RUNTIME_ROOT/val/dreams_predictions.npz" \
  --validation-fingerprint-key probs \
  --validation-fingerprint-threshold 0.95 \
  --evaluation-manifest "$EVALUATION_MANIFEST" \
  --output-dir "$RUN_ROOT" \
  --max-steps "$ADAPT_MAX_STEPS" \
  --evaluation-interval 100 \
  --checkpoint-interval 100 \
  --evaluation-spectra "$EVALUATION_SPECTRA" \
  --evaluation-candidates "$EVALUATION_CANDIDATES" \
  --cross-attention-only-steps "$CROSS_ATTENTION_ONLY_STEPS" \
  --noise-probability 0.5 \
  --noise-min-fraction 0.1 \
  --noise-max-fraction 0.3 \
  --learning-rate 1e-5
