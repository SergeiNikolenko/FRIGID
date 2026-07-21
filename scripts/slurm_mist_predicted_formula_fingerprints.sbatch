#!/bin/bash
#SBATCH --job-name=mist-predformula-fp
#SBATCH --partition=gpu
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
DATASET_PREFIX="$ROOT/data/paired_spectra/marlin_mist_cf_clean_job${SLURM_JOB_ID}"
MIST_CHECKPOINT="$ROOT/checkpoints/mist/mist_fp_canopus_pretrain.ckpt"
MIST_OUTPUT="$ROOT/runs/mist_connectivity_clean_fingerprints_job${SLURM_JOB_ID}"

export PYTHONPATH="$CODE:$CODE/src:$CODE/scripts:$MIST/src"
MIST_COMMIT=$(git -C "$MIST" rev-parse HEAD)
MIST_CHECKPOINT_SHA256=$(sha256sum "$MIST_CHECKPOINT" | cut -d' ' -f1)
SIRIUS_VERSION=5.5.7

printf 'PROVENANCE\n'
printf 'MIST_DEVICE=cpu\n'
git -C "$CODE" rev-parse HEAD
git -C "$MIST" rev-parse HEAD
"$SIRIUS" --version
sha256sum "$MGF" "$FORMULA_PREDICTIONS" "$MIST_CHECKPOINT"
sha256sum "$ROOT/envs/mist_official_py38/requirements.lock.txt"

OVERRIDES=""
FINAL_DATASET=""
for ROUND in $(seq 1 20); do
  DATASET="${DATASET_PREFIX}_round${ROUND}"
  PREPARE_ARGS=(
    --mgf "$MGF"
    --predictions "$FORMULA_PREDICTIONS"
    --output-dir "$DATASET"
  )
  if [[ -n "$OVERRIDES" ]]; then
    PREPARE_ARGS+=(--minimum-candidate-ranks "$OVERRIDES")
  fi
  "$PYTHON" "$CODE/scripts/prepare_mist_predicted_formula_dataset.py" \
    "${PREPARE_ARGS[@]}"

  "$SIRIUS" \
    --cores 12 \
    --output "$DATASET/sirius_outputs" \
    --naming-convention '%compoundname' \
    --input "$DATASET/forced_formula.mgf" \
    formula \
    --ppm-max-ms2 10

  AUDIT="$DATASET/sirius_consistency_audit.json"
  "$PYTHON" "$CODE/scripts/audit_sirius_formula_bridge.py" \
    --project-dir "$DATASET/sirius_outputs" \
    --labels "$DATASET/labels.tsv" \
    --output "$AUDIT" \
    --manifest "$DATASET/formula_bridge_manifest.json"
  MISMATCH_COUNT=$("$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["mismatch_count"])' \
    "$AUDIT")
  if [[ "$MISMATCH_COUNT" == "0" ]]; then
    FINAL_DATASET="$DATASET"
    break
  fi
  printf 'SIRIUS rejected %s candidates in round %s; advancing ranks\n' \
    "$MISMATCH_COUNT" "$ROUND"
  OVERRIDES="$AUDIT"
done

if [[ -z "$FINAL_DATASET" ]]; then
  printf 'No fully SIRIUS-consistent candidate set after 20 rounds\n' >&2
  exit 1
fi
DATASET="$FINAL_DATASET"
DATASET_NAME=$(basename "$DATASET")

"$PYTHON" "$CODE/scripts/unpack_sirius_for_mist.py" \
  --project-dir "$DATASET/sirius_outputs" \
  --labels "$DATASET/labels.tsv" \
  --output "$DATASET/sirius_outputs/summary_statistics/summary_df.tsv" \
  --mist-labels-output "$DATASET/mist_labels.tsv" \
  --formula-manifest "$DATASET/formula_bridge_manifest.json" \
  --sirius-audit "$DATASET/sirius_consistency_audit.json" \
  --bridge-manifest-output "$DATASET/sirius_bridge_manifest.json"

mkdir -p "$MIST_OUTPUT"
cd "$ROOT"
"$PYTHON" "$MIST/run_scripts/pred_fp.py" \
  --model-ckpt "$MIST_CHECKPOINT" \
  --save-dir "$MIST_OUTPUT" \
  --dataset-name "$DATASET_NAME" \
  --num-workers 12 \
  --labels-name mist_labels.tsv

"$PYTHON" "$CODE/scripts/package_mist_predictions.py" \
  --predictions "$MIST_OUTPUT/fp_preds_${DATASET_NAME}.p" \
  --metadata "$ROOT/data/processed/test/metadata.csv" \
  --formula-manifest "$DATASET/formula_bridge_manifest.json" \
  --sirius-bridge-manifest "$DATASET/sirius_bridge_manifest.json" \
  --mist-labels "$DATASET/mist_labels.tsv" \
  --output "$MIST_OUTPUT/fingerprints.npz" \
  --official-mist-git-commit "$MIST_COMMIT" \
  --sirius-version "$SIRIUS_VERSION" \
  --mist-checkpoint-sha256 "$MIST_CHECKPOINT_SHA256"

find "$DATASET" "$MIST_OUTPUT" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$ROOT/manifests/mist_connectivity_clean_fingerprint_lane.sha256"
