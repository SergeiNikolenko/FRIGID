#!/bin/bash

set -euo pipefail

SHARED_ROOT="${MARLIN_SHARED_ROOT:-/mnt/netstorage/nikolenko/marlin}"
RUNTIME_ROOT="${MARLIN_RUNTIME_INPUT_ROOT:-$SHARED_ROOT/runtime-inputs-spectrum-v1}"
TASK_ID="${CLEARML_TASK_ID:?CLEARML_TASK_ID is required}"
RUN_ROOT="$SHARED_ROOT/runs/spectrum-fingerprint-adaptation-$TASK_ID"
CHECKPOINT_CACHE_ROOT="$SHARED_ROOT/cache/checkpoints"

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}.clearai.innopolis.university,.university.innopolis.ru"
export no_proxy="${no_proxy:+$no_proxy,}.clearai.innopolis.university,.university.innopolis.ru"
export PYTHONFAULTHANDLER=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/materialize_marlin_runtime_inputs.py

CHECKPOINT="$(
  python scripts/materialize_marlin_checkpoint.py \
    --task-id "${MARLIN_SOURCE_CHECKPOINT_TASK_ID:?MARLIN_SOURCE_CHECKPOINT_TASK_ID is required}" \
    --artifact-name "${MARLIN_SOURCE_CHECKPOINT_ARTIFACT:?MARLIN_SOURCE_CHECKPOINT_ARTIFACT is required}" \
    --expected-sha256 "${MARLIN_SOURCE_CHECKPOINT_SHA256:?MARLIN_SOURCE_CHECKPOINT_SHA256 is required}" \
    --cache-root "$CHECKPOINT_CACHE_ROOT" \
    --output-name "source.ckpt"
)"

test ! -e "$RUN_ROOT"

python -X faulthandler scripts/train_marlin_spectrum_adaptation.py \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$MARLIN_SOURCE_CHECKPOINT_SHA256" \
  --tokenizer "$RUNTIME_ROOT/tokenizer.json" \
  --metadata "$RUNTIME_ROOT/train/metadata.csv" \
  --fingerprints "$RUNTIME_ROOT/train/dreams_predictions.npz" \
  --fingerprint-key probs \
  --fingerprint-threshold "${MARLIN_TRAIN_FINGERPRINT_THRESHOLD:-0.95}" \
  --exclude-inchikeys "$RUNTIME_ROOT/nplib1_test_inchikeys.csv" \
  --exclude-metadata "$RUNTIME_ROOT/val/metadata.csv" \
  --validation-metadata "$RUNTIME_ROOT/val/metadata.csv" \
  --validation-fingerprints "$RUNTIME_ROOT/val/dreams_predictions.npz" \
  --validation-fingerprint-key probs \
  --validation-fingerprint-threshold "${MARLIN_VALIDATION_FINGERPRINT_THRESHOLD:-0.95}" \
  --output-dir "$RUN_ROOT" \
  --max-steps "${MARLIN_MAX_STEPS:-100}" \
  --evaluation-interval "${MARLIN_EVALUATION_INTERVAL:-100}" \
  --checkpoint-interval "${MARLIN_CHECKPOINT_INTERVAL:-100}" \
  --evaluation-spectra "${MARLIN_EVALUATION_SPECTRA:-4}" \
  --evaluation-candidates "${MARLIN_EVALUATION_CANDIDATES:-16}" \
  --cross-attention-only-steps "${MARLIN_CROSS_ATTENTION_ONLY_STEPS:-100}" \
  --learning-rate "${MARLIN_LEARNING_RATE:-1e-5}"
