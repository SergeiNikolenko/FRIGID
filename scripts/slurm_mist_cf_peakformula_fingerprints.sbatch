#!/bin/bash
#SBATCH --job-name=mistcf-peakformula-fp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/mistcf-peakformula-fp-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
MIST="$ROOT/vendor/mist-official"
MIST_CF="$ROOT/vendor/mist-cf-official"
PYTHON="$ROOT/envs/mist_official_py38/bin/python"
MGF="$ROOT/data/processed/test/spectra.mgf"
FORMULA_PREDICTIONS="$ROOT/runs/mist_cf_connectivity_clean_prediction/test/formatted_output.tsv"
SUBFORMULAS="$ROOT/runs/mist_cf_connectivity_clean_prediction/test/subform_assigns"
MIST_CF_CHECKPOINT="$ROOT/runs/mist_cf_connectivity_clean/version_0/best.ckpt"
MIST_CHECKPOINT="$ROOT/checkpoints/mist/mist_fp_canopus_pretrain.ckpt"
DATASET="$ROOT/data/paired_spectra/marlin_mist_cf_peakformula_job${SLURM_JOB_ID}"
DATASET_NAME=$(basename "$DATASET")
MIST_OUTPUT="$ROOT/runs/mist_cf_peakformula_fingerprints_job${SLURM_JOB_ID}"

if test -e "$DATASET" || test -e "$MIST_OUTPUT"; then
  echo "Refusing to overwrite an existing MIST-CF PeakFormula lane." >&2
  exit 2
fi

MIST_COMMIT=$(git -C "$MIST" rev-parse HEAD)
MIST_CF_COMMIT=$(git -C "$MIST_CF" rev-parse HEAD)
MIST_CHECKPOINT_SHA256=$(sha256sum "$MIST_CHECKPOINT" | cut -d' ' -f1)
MIST_CF_CHECKPOINT_SHA256=$(sha256sum "$MIST_CF_CHECKPOINT" | cut -d' ' -f1)

printf 'PROVENANCE\n'
printf 'LANE=MIST with MIST-CF top-1 peak-to-subformula adapter\n'
printf 'MIST_DEVICE=cpu\n'
git -C "$CODE" rev-parse HEAD
git -C "$MIST" rev-parse HEAD
git -C "$MIST_CF" rev-parse HEAD
sha256sum \
  "$MGF" \
  "$FORMULA_PREDICTIONS" \
  "$MIST_CF_CHECKPOINT" \
  "$MIST_CHECKPOINT" \
  "$ROOT/envs/mist_official_py38/requirements.lock.txt"

export PYTHONPATH="$CODE:$CODE/src:$CODE/scripts"
"$PYTHON" "$CODE/scripts/prepare_mist_cf_peakformula_dataset.py" \
  --mgf "$MGF" \
  --predictions "$FORMULA_PREDICTIONS" \
  --subformula-dir "$SUBFORMULAS" \
  --output-dir "$DATASET" \
  --mist-cf-git-commit "$MIST_CF_COMMIT" \
  --mist-cf-checkpoint-sha256 "$MIST_CF_CHECKPOINT_SHA256"

mkdir -p "$MIST_OUTPUT"
cd "$ROOT"
export PYTHONPATH="$MIST/src"
"$PYTHON" -c 'from mist import pred_fp; print(f"OFFICIAL_MIST={pred_fp.__file__}")'
"$PYTHON" "$MIST/run_scripts/pred_fp.py" \
  --model-ckpt "$MIST_CHECKPOINT" \
  --save-dir "$MIST_OUTPUT" \
  --dataset-name "$DATASET_NAME" \
  --num-workers 12 \
  --labels-name mist_labels.tsv

export PYTHONPATH="$CODE:$CODE/src:$CODE/scripts"
"$PYTHON" "$CODE/scripts/package_mist_predictions.py" \
  --predictions "$MIST_OUTPUT/fp_preds_${DATASET_NAME}.p" \
  --metadata "$ROOT/data/processed/test/metadata.csv" \
  --formula-manifest "$DATASET/formula_bridge_manifest.json" \
  --feature-bridge-manifest "$DATASET/feature_bridge_manifest.json" \
  --mist-labels "$DATASET/mist_labels.tsv" \
  --output "$MIST_OUTPUT/fingerprints.npz" \
  --official-mist-git-commit "$MIST_COMMIT" \
  --mist-checkpoint-sha256 "$MIST_CHECKPOINT_SHA256"

find "$DATASET" "$MIST_OUTPUT" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$ROOT/manifests/mist_cf_peakformula_fingerprint_lane_job${SLURM_JOB_ID}.sha256"
