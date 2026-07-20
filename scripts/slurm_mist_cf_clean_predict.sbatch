#!/bin/bash
#SBATCH --job-name=mistcf-clean-predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/mistcf-clean-predict-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
VENDOR="$ROOT/vendor/mist-cf-official"
PYTHON="$ROOT/envs/mist_cf_py38/bin/python"
MGF="$ROOT/data/processed/test/spectra.mgf"
CHECKPOINT="$ROOT/runs/mist_cf_connectivity_clean/version_0/best.ckpt"
FAST_MODEL="$ROOT/checkpoints/mist_cf/fast_filter_best.ckpt"
OUTPUT="$ROOT/runs/mist_cf_connectivity_clean_prediction/test"
SIRIUS="$ROOT/vendor/sirius/bin/sirius"

if test -e "$OUTPUT/formatted_output.tsv"; then
  echo "Refusing to overwrite an existing clean MIST-CF prediction." >&2
  exit 2
fi

mkdir -p "$OUTPUT"
export PYTHONPATH="$VENDOR/src"
export SIRIUS_PATH="$SIRIUS"

printf 'PROVENANCE\n'
git -C "$CODE" rev-parse HEAD
git -C "$VENDOR" rev-parse HEAD
"$SIRIUS" --version
sha256sum \
  "$ROOT/envs/mist_cf_py38/requirements.lock.txt" \
  "$ROOT/data/mist_cf_official/canopus_train/splits/split_1_nplib1_connectivity_clean.tsv" \
  "$CHECKPOINT" \
  "$FAST_MODEL" \
  "$MGF"

"$PYTHON" "$VENDOR/src/mist_cf/mist_cf_score/predict_mgf.py" \
  --gpu \
  --num-workers 12 \
  --batch-size 8 \
  --save-dir "$OUTPUT" \
  --mgf-file "$MGF" \
  --checkpoint-pth "$CHECKPOINT" \
  --fast-model "$FAST_MODEL" \
  --fast-num 256 \
  --decomp-filter RDBE \
  --decomp-ppm 5 \
  --id-key SCANS

find "$OUTPUT" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$ROOT/manifests/mist_cf_connectivity_clean_prediction.sha256"
