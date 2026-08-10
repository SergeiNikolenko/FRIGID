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

# materialize_marlin_runtime_inputs.py has its own default root, which does
# not match RUNTIME_ROOT. Pin both to the same resolved path, otherwise the
# bundle unpacks into one directory and training reads from another.
export MARLIN_RUNTIME_INPUT_ROOT="$RUNTIME_ROOT"

python scripts/materialize_marlin_runtime_inputs.py

CHECKPOINT_SOURCE_ARGS=()
if [[ -n "${MARLIN_SOURCE_CHECKPOINT_ARTIFACT_URI:-}" ]]; then
  CHECKPOINT_SOURCE_ARGS+=(
    --artifact-uri "$MARLIN_SOURCE_CHECKPOINT_ARTIFACT_URI"
  )
else
  CHECKPOINT_SOURCE_ARGS+=(
    --task-id "${MARLIN_SOURCE_CHECKPOINT_TASK_ID:?MARLIN_SOURCE_CHECKPOINT_TASK_ID is required}"
    --artifact-name "${MARLIN_SOURCE_CHECKPOINT_ARTIFACT:?MARLIN_SOURCE_CHECKPOINT_ARTIFACT is required}"
  )
fi
CHECKPOINT="$(
  python scripts/materialize_marlin_checkpoint.py \
    "${CHECKPOINT_SOURCE_ARGS[@]}" \
    --expected-sha256 "${MARLIN_SOURCE_CHECKPOINT_SHA256:?MARLIN_SOURCE_CHECKPOINT_SHA256 is required}" \
    --cache-root "$CHECKPOINT_CACHE_ROOT" \
    --output-name "checkpoint.ckpt"
)"

SOFT_FINGERPRINT_ARGS=()
if [[ "${MARLIN_SOFT_FINGERPRINT:-0}" == "1" ]]; then
  SOFT_FINGERPRINT_ARGS+=(--soft-fingerprint)
fi
if [[ "${MARLIN_MASS_REACHABILITY_PRUNE:-0}" == "1" ]]; then
  SOFT_FINGERPRINT_ARGS+=(--mass-reachability-prune)
fi
if [[ "${MARLIN_FORBID_ISOTOPE_TOKENS:-0}" == "1" ]]; then
  SOFT_FINGERPRINT_ARGS+=(--forbid-isotope-tokens)
fi
if [[ "${MARLIN_RESTRICT_ORGANIC_ELEMENTS:-0}" == "1" ]]; then
  SOFT_FINGERPRINT_ARGS+=(--restrict-organic-elements)
fi

test ! -e "$RUN_ROOT"

# Held-out checkpoint selection, on by default: without it the recipe has no
# early stopping, which is how a 100,000-step run reported a flat Exact@1 while a
# matched re-evaluation showed candidate return halving after step 20,000.
SELECTION_ARGS=()
if [[ "${MARLIN_SELECT_BEST_CHECKPOINT:-1}" == "1" ]]; then
  SELECTION_ARGS+=(--select-best-checkpoint)
fi

python -X faulthandler scripts/train_marlin_spectrum_adaptation.py \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$MARLIN_SOURCE_CHECKPOINT_SHA256" \
  --tokenizer "$RUNTIME_ROOT/tokenizer.json" \
  --metadata "$RUNTIME_ROOT/train/metadata.csv" \
  --fingerprints "$RUNTIME_ROOT/train/dreams_predictions.npz" \
  --fingerprint-key probs \
  --fingerprint-threshold "${MARLIN_TRAIN_FINGERPRINT_THRESHOLD:-0.90}" \
  "${SOFT_FINGERPRINT_ARGS[@]}" \
  --exclude-inchikeys "$RUNTIME_ROOT/nplib1_test_inchikeys.csv" \
  --exclude-metadata "$RUNTIME_ROOT/val/metadata.csv" \
  --validation-metadata "$RUNTIME_ROOT/val/metadata.csv" \
  --validation-fingerprints "$RUNTIME_ROOT/val/dreams_predictions.npz" \
  --validation-fingerprint-key probs \
  --validation-fingerprint-threshold "${MARLIN_VALIDATION_FINGERPRINT_THRESHOLD:-0.95}" \
  --evaluation-manifest "${MARLIN_EVALUATION_MANIFEST:-$PWD/configs/benchmarks/nplib1_v1/nplib1_val_full396_v1.tsv}" \
  --output-dir "$RUN_ROOT" \
  --max-steps "${MARLIN_MAX_STEPS:-100}" \
  --evaluation-interval "${MARLIN_EVALUATION_INTERVAL:-100}" \
  --checkpoint-interval "${MARLIN_CHECKPOINT_INTERVAL:-100}" \
  --evaluation-spectra "${MARLIN_EVALUATION_SPECTRA:-396}" \
  --evaluation-candidates "${MARLIN_EVALUATION_CANDIDATES:-8}" \
  --evaluation-shards "${MARLIN_EVALUATION_SHARDS:-16}" \
  --validation-loss-fraction "${MARLIN_VALIDATION_LOSS_FRACTION:-0.05}" \
  --validation-loss-split-seed "${MARLIN_VALIDATION_LOSS_SPLIT_SEED:-0}" \
  "${SELECTION_ARGS[@]}" \
  --selection-metric "${MARLIN_SELECTION_METRIC:-candidate_return_rate}" \
  --selection-patience "${MARLIN_SELECTION_PATIENCE:-3}" \
  --cross-attention-only-steps "${MARLIN_CROSS_ATTENTION_ONLY_STEPS:-100}" \
  --learning-rate "${MARLIN_LEARNING_RATE:-1e-5}"
