#!/usr/bin/env bash
#SBATCH --job-name=frigid-full-finalize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

: "${FRIGID_REPO:?Set FRIGID_REPO to a clean Spectrum worktree}"
: "${FRIGID_ASSET_ROOT:?Set FRIGID_ASSET_ROOT with the prepared environment}"
: "${FINALIZATION_RUN_DIR:?Set FINALIZATION_RUN_DIR to a new output directory}"
: "${CONTROL_SHARDS_FILE:?Set CONTROL_SHARDS_FILE}"
: "${TEMPERATURE_SHARDS_FILE:?Set TEMPERATURE_SHARDS_FILE}"
: "${EXPECTED_MANIFEST:?Set EXPECTED_MANIFEST}"
: "${RETRIEVAL_MANIFEST:?Set RETRIEVAL_MANIFEST}"
: "${MOLFORGE_MANIFEST:?Set MOLFORGE_MANIFEST}"
: "${EXPECTED_CODE_COMMIT:?Set EXPECTED_CODE_COMMIT}"
: "${EXPECTED_MANIFEST_SHA256:?Set EXPECTED_MANIFEST_SHA256}"
: "${EXPECTED_CONTROL_SHARDS_SHA256:?Set EXPECTED_CONTROL_SHARDS_SHA256}"
: "${EXPECTED_TEMPERATURE_SHARDS_SHA256:?Set EXPECTED_TEMPERATURE_SHARDS_SHA256}"
: "${EXPECTED_RETRIEVAL_MANIFEST_SHA256:?Set EXPECTED_RETRIEVAL_MANIFEST_SHA256}"
: "${EXPECTED_MOLFORGE_MANIFEST_SHA256:?Set EXPECTED_MOLFORGE_MANIFEST_SHA256}"

PYTHON_BIN="${PYTHON_BIN:-$FRIGID_ASSET_ROOT/.venv/bin/python}"
for path in "$PYTHON_BIN" "$CONTROL_SHARDS_FILE" "$TEMPERATURE_SHARDS_FILE" \
  "$EXPECTED_MANIFEST" "$RETRIEVAL_MANIFEST" "$MOLFORGE_MANIFEST"; do
  if [[ ! -e "$path" ]]; then
    echo "Required finalization input does not exist: $path" >&2
    exit 2
  fi
done
if [[ -e "$FINALIZATION_RUN_DIR" ]]; then
  echo "Finalization output already exists: $FINALIZATION_RUN_DIR" >&2
  exit 2
fi

actual_commit="$(git -C "$FRIGID_REPO" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_CODE_COMMIT" ]]; then
  echo "Code commit mismatch: expected $EXPECTED_CODE_COMMIT, got $actual_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$FRIGID_REPO" status --porcelain --untracked-files=all)" ]]; then
  echo "FRIGID_REPO must be clean: $FRIGID_REPO" >&2
  exit 2
fi

sha256_file() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

verify_sha256() {
  local label="$1" path="$2" expected="$3" actual
  actual="$(sha256_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA-256 mismatch: expected $expected, got $actual" >&2
    exit 2
  fi
}

verify_sha256 "Locked manifest" "$EXPECTED_MANIFEST" "$EXPECTED_MANIFEST_SHA256"
verify_sha256 \
  "Control shard list" "$CONTROL_SHARDS_FILE" "$EXPECTED_CONTROL_SHARDS_SHA256"
verify_sha256 \
  "Temperature shard list" \
  "$TEMPERATURE_SHARDS_FILE" \
  "$EXPECTED_TEMPERATURE_SHARDS_SHA256"
verify_sha256 \
  "Retrieval manifest" \
  "$RETRIEVAL_MANIFEST" \
  "$EXPECTED_RETRIEVAL_MANIFEST_SHA256"
verify_sha256 \
  "MolForge manifest" \
  "$MOLFORGE_MANIFEST" \
  "$EXPECTED_MOLFORGE_MANIFEST_SHA256"

"$PYTHON_BIN" "$FRIGID_REPO/scripts/finalize_full_four_source.py" \
  --control-shards-file "$CONTROL_SHARDS_FILE" \
  --temperature-shards-file "$TEMPERATURE_SHARDS_FILE" \
  --expected-manifest "$EXPECTED_MANIFEST" \
  --retrieval-manifest "$RETRIEVAL_MANIFEST" \
  --molforge-manifest "$MOLFORGE_MANIFEST" \
  --output-dir "$FINALIZATION_RUN_DIR" \
  --bootstrap-resamples 10000 \
  --seed 42
