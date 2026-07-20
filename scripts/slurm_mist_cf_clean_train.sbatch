#!/bin/bash
#SBATCH --job-name=mistcf-clean-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/mistcf-clean-train-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
VENDOR="$ROOT/vendor/mist-cf-official"
DATA="$ROOT/data/mist_cf_official/canopus_train"
PYTHON="$ROOT/envs/mist_cf_py38/bin/python"
OUTPUT_ROOT="${MIST_CF_OUTPUT_ROOT:-$ROOT/runs/mist_cf_connectivity_clean}"
export PYTHONPATH="$VENDOR/src"

printf 'PROVENANCE\n'
git -C "$VENDOR" rev-parse HEAD
sha256sum \
  "$ROOT/envs/mist_cf_py38/requirements.lock.txt" \
  "$DATA/splits/split_1_nplib1_connectivity_clean.tsv" \
  "$DATA/decoy_labels/decoy_label_COMMON.tsv"

cd "$ROOT"
"$PYTHON" "$VENDOR/src/mist_cf/mist_cf_score/train.py" \
  --gpu \
  --save-dir "$OUTPUT_ROOT" \
  --dataset-name mist_cf_official/canopus_train \
  --split-file "$DATA/splits/split_1_nplib1_connectivity_clean.tsv" \
  --decoy-label "$DATA/decoy_labels/decoy_label_COMMON.tsv" \
  --subform-dir "$DATA/subformulae/formulae_spec_decoy_label_COMMON" \
  --seed 1 \
  --num-workers 8 \
  --batch-size 4 \
  --max-decoy 32 \
  --max-epochs 200 \
  --learning-rate 0.00045 \
  --lr-decay-frac 0.88 \
  --weight-decay 0 \
  --max-subpeak 20 \
  --layers 2 \
  --dropout 0.1 \
  --hidden-size 128 \
  --form-encoder abs-sines \
  --no-cls-mass-diff
