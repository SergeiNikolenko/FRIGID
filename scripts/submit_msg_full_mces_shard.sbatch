#!/usr/bin/env bash
#SBATCH --job-name=frigid-mces
#SBATCH --partition=gpu-shared
#SBATCH --gres=shard:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=12:00:00

set -euo pipefail

: "${FRIGID_REPO:?Set FRIGID_REPO to a clean Spectrum worktree}"
: "${BASE_PYTHON:?Set BASE_PYTHON}"
: "${EXPECTED_CODE_COMMIT:?Set EXPECTED_CODE_COMMIT}"
: "${EXPECTED_MANIFEST:?Set EXPECTED_MANIFEST}"
: "${RUNTIME_MANIFEST:?Set RUNTIME_MANIFEST}"
: "${EXPECTED_RUNTIME_MANIFEST_SHA256:?Set EXPECTED_RUNTIME_MANIFEST_SHA256}"
: "${CONTROL_PREDICTIONS:?Set CONTROL_PREDICTIONS}"
: "${UNION_PREDICTIONS:?Set UNION_PREDICTIONS}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new shard directory}"
: "${START_INDEX:?Set START_INDEX}"
: "${MAX_SPECTRA:?Set MAX_SPECTRA}"

if [[ "$(git -C "$FRIGID_REPO" rev-parse HEAD)" != "$EXPECTED_CODE_COMMIT" ]]; then
  echo "MCES code commit mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "$FRIGID_REPO" status --porcelain --untracked-files=all)" ]]; then
  echo "MCES worktree must be clean: $FRIGID_REPO" >&2
  exit 2
fi

"$BASE_PYTHON" "$FRIGID_REPO/scripts/evaluate_mces_predictions.py" \
  --predictions "control=$CONTROL_PREDICTIONS" \
  --predictions "union=$UNION_PREDICTIONS" \
  --expected-manifest "$EXPECTED_MANIFEST" \
  --runtime-manifest "$RUNTIME_MANIFEST" \
  --expected-runtime-manifest-sha256 "$EXPECTED_RUNTIME_MANIFEST_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --start-index "$START_INDEX" \
  --max-spectra "$MAX_SPECTRA" \
  --n-jobs 4 \
  --query-batch-size 8
