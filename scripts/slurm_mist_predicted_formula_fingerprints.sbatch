#!/bin/bash
#SBATCH --job-name=mist-predformula-fp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/mist-predformula-fp-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
MIST="$ROOT/vendor/mist-official"
SIRIUS="$ROOT/vendor/sirius/bin/sirius"
PYTHON="$ROOT/envs/mist_official_py38/bin/python"
MGF="$ROOT/data/processed/test/spectra.mgf"
FORMULA_PREDICTIONS="$ROOT/runs/mist_cf_connectivity_clean_prediction/test/formatted_output.tsv"
DATASET="$ROOT/data/paired_spectra/marlin_mist_cf_clean"
MIST_CHECKPOINT="$ROOT/checkpoints/mist/mist_fp_canopus_pretrain.ckpt"
MIST_OUTPUT="$ROOT/runs/mist_connectivity_clean_fingerprints"

export PYTHONPATH="$CODE/src:$CODE/scripts:$MIST/src"
MIST_COMMIT=$(git -C "$MIST" rev-parse HEAD)
MIST_CHECKPOINT_SHA256=$(sha256sum "$MIST_CHECKPOINT" | cut -d' ' -f1)
SIRIUS_VERSION=5.5.7

printf 'PROVENANCE\n'
git -C "$CODE" rev-parse HEAD
git -C "$MIST" rev-parse HEAD
"$SIRIUS" --version
sha256sum "$MGF" "$FORMULA_PREDICTIONS" "$MIST_CHECKPOINT"
sha256sum "$ROOT/envs/mist_official_py38/requirements.lock.txt"

"$PYTHON" "$CODE/scripts/prepare_mist_predicted_formula_dataset.py" \
  --mgf "$MGF" \
  --predictions "$FORMULA_PREDICTIONS" \
  --output-dir "$DATASET"

"$SIRIUS" \
  --cores 12 \
  --output "$DATASET/sirius_outputs" \
  --naming-convention '%index_%compoundname' \
  --input "$DATASET/forced_formula.mgf" \
  formula \
  --ppm-max-ms2 10

"$PYTHON" "$CODE/scripts/unpack_sirius_for_mist.py" \
  --project-dir "$DATASET/sirius_outputs" \
  --labels "$DATASET/labels.tsv" \
  --output "$DATASET/sirius_outputs/summary_statistics/summary_df.tsv"

mkdir -p "$MIST_OUTPUT"
cd "$ROOT"
"$PYTHON" "$MIST/run_scripts/pred_fp.py" \
  --model-ckpt "$MIST_CHECKPOINT" \
  --save-dir "$MIST_OUTPUT" \
  --dataset-name marlin_mist_cf_clean \
  --num-workers 12 \
  --gpu \
  --labels-name labels.tsv

"$PYTHON" "$CODE/scripts/package_mist_predictions.py" \
  --predictions "$MIST_OUTPUT/fp_preds_marlin_mist_cf_clean.p" \
  --metadata "$ROOT/data/processed/test/metadata.csv" \
  --formula-manifest "$DATASET/formula_bridge_manifest.json" \
  --output "$MIST_OUTPUT/fingerprints.npz" \
  --official-mist-git-commit "$MIST_COMMIT" \
  --sirius-version "$SIRIUS_VERSION" \
  --mist-checkpoint-sha256 "$MIST_CHECKPOINT_SHA256"

find "$DATASET" "$MIST_OUTPUT" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$ROOT/manifests/mist_connectivity_clean_fingerprint_lane.sha256"
