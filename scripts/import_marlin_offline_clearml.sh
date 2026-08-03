#!/bin/bash

set -euo pipefail

SOURCE_JOB_ID="${1:?source Slurm job ID is required}"
PROJECT_ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_BASE="${MARLIN_CLEARML_CACHE_ROOT:-$PROJECT_ROOT/cache/clearml-offline}"
CACHE_ROOT="$CACHE_BASE/$SOURCE_JOB_ID"

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

"$PROJECT_ROOT/code/.venv/bin/clearml-task" \
  --import-offline-session "${SESSIONS[0]}" \
  | tee "$CACHE_ROOT/import.log"
