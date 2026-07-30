#!/bin/bash

set -euo pipefail

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}.clearai.innopolis.university,.university.innopolis.ru"
export no_proxy="${no_proxy:+$no_proxy,}.clearai.innopolis.university,.university.innopolis.ru"

PUBLISH_ARGS=(
  --checkpoint "${MARLIN_PUBLISH_CHECKPOINT:?MARLIN_PUBLISH_CHECKPOINT is required}" \
  --artifact-name "${MARLIN_PUBLISH_ARTIFACT_NAME:?MARLIN_PUBLISH_ARTIFACT_NAME is required}"
)
if [[ -n "${MARLIN_PUBLISH_CHECKPOINT_SHA256:-}" ]]; then
  PUBLISH_ARGS+=(--expected-sha256 "$MARLIN_PUBLISH_CHECKPOINT_SHA256")
fi

python scripts/publish_marlin_checkpoint.py "${PUBLISH_ARGS[@]}"
