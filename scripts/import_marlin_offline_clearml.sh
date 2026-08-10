#!/bin/bash

set -euo pipefail

SOURCE_JOB_ID="${1:?source Slurm job ID is required}"
PROJECT_ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_BASE="${MARLIN_CLEARML_CACHE_ROOT:-$PROJECT_ROOT/cache/clearml-offline}"
CACHE_ROOT="$CACHE_BASE/$SOURCE_JOB_ID"
IMPORT_TIMEOUT_SECONDS="${MARLIN_CLEARML_IMPORT_TIMEOUT_SECONDS:-300}"

if [[ ! "$IMPORT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MARLIN_CLEARML_IMPORT_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi

export MARLIN_PYTHON="$PROJECT_ROOT/code/.venv/bin/python"
unset CLEARML_OFFLINE_MODE
ROOT="$PROJECT_ROOT/code"
source "$CODE/scripts/marlin_runtime_env.sh"

mapfile -d '' SESSIONS < <(
  find "$CACHE_ROOT/offline" \
    -maxdepth 1 \
    -type f \
    -name '*.zip' \
    -print0
)
if (( ${#SESSIONS[@]} != 1 )); then
  echo "expected exactly one offline session in $CACHE_ROOT/offline" >&2
  exit 2
fi

IMPORT_LOG="$CACHE_ROOT/import.log"
echo "Importing ${SESSIONS[0]} with a ${IMPORT_TIMEOUT_SECONDS}s timeout" \
  | tee "$IMPORT_LOG"

set +e
timeout --foreground --kill-after=30s "$IMPORT_TIMEOUT_SECONDS" \
  "$PROJECT_ROOT/code/.venv/bin/clearml-task" \
  --import-offline-session "${SESSIONS[0]}" \
  2>&1 | tee -a "$IMPORT_LOG"
PIPE_STATUSES=("${PIPESTATUS[@]}")
set -e

IMPORT_STATUS="${PIPE_STATUSES[0]}"
TEE_STATUS="${PIPE_STATUSES[1]}"
if (( TEE_STATUS != 0 )); then
  echo "failed to write ClearML import log: $IMPORT_LOG" >&2
  exit "$TEE_STATUS"
fi
if (( IMPORT_STATUS == 124 )); then
  echo "ClearML offline import timed out after ${IMPORT_TIMEOUT_SECONDS}s. Verify API reachability and retry; log: $IMPORT_LOG" >&2
  exit 124
fi
if (( IMPORT_STATUS != 0 )); then
  echo "ClearML offline import failed with status $IMPORT_STATUS; log: $IMPORT_LOG" >&2
  exit "$IMPORT_STATUS"
fi
