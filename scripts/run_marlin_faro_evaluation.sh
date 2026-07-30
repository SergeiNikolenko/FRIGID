#!/bin/bash

set -euo pipefail

SHARED_ROOT="${MARLIN_SHARED_ROOT:-/mnt/netstorage/nikolenko/marlin}"
RUNTIME_ROOT="${MARLIN_RUNTIME_INPUT_ROOT:-$SHARED_ROOT/runtime-inputs-v1}"
CHECKPOINT="${MARLIN_EVAL_CHECKPOINT:?MARLIN_EVAL_CHECKPOINT is required}"
SOURCE_STEP="${MARLIN_EVAL_STEP:?MARLIN_EVAL_STEP is required}"
TASK_ID="${CLEARML_TASK_ID:?CLEARML_TASK_ID is required}"
OUTPUT_ROOT="$SHARED_ROOT/evaluations/faro-paper-gate-$TASK_ID"

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}.clearai.innopolis.university,.university.innopolis.ru"
export no_proxy="${no_proxy:+$no_proxy,}.clearai.innopolis.university,.university.innopolis.ru"
export PYTHONFAULTHANDLER=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/materialize_marlin_runtime_inputs.py
test -f "$CHECKPOINT"
test ! -e "$OUTPUT_ROOT"

EXTRA_ARGS=()
if [[ -n "${MARLIN_EVAL_THRESHOLD:-}" ]]; then
  EXTRA_ARGS+=(--threshold "$MARLIN_EVAL_THRESHOLD")
fi
if [[ -n "${MARLIN_EVAL_CANDIDATE_BATCH_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--candidate-batch-size "$MARLIN_EVAL_CANDIDATE_BATCH_SIZE")
fi
if [[ "${MARLIN_EVAL_DISABLE_GRAMMAR_MASK:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-grammar-mask)
fi
if [[ "${MARLIN_EVAL_DISABLE_MASS_SHELL:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-mass-shell)
fi
if [[ "${MARLIN_EVAL_SAMPLE_TOKENS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--sample-tokens)
fi
if [[ "${MARLIN_EVAL_NO_EMA:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-ema)
fi

python -X faulthandler scripts/evaluate_marlin_nplib1.py \
  --checkpoint "$CHECKPOINT" \
  --tokenizer "$RUNTIME_ROOT/tokenizer.json" \
  --metadata "$RUNTIME_ROOT/${MARLIN_EVAL_METADATA_REL:-val/metadata.csv}" \
  --fingerprints "$RUNTIME_ROOT/${MARLIN_EVAL_FINGERPRINTS_REL:-val/fingerprints.npz}" \
  --fingerprint-key "${MARLIN_EVAL_FINGERPRINT_KEY:-ground_truth}" \
  --lane dreams \
  --output-dir "$OUTPUT_ROOT" \
  --candidates "${MARLIN_EVAL_CANDIDATES:-16}" \
  --max-spectra "${MARLIN_EVAL_MAX_SPECTRA:-4}" \
  --diversity-dropout "${MARLIN_EVAL_DIVERSITY_DROPOUT:-0.3}" \
  --temperature 1.0 \
  --generation-mode "${MARLIN_EVAL_GENERATION_MODE:-block}" \
  --ppm-tolerance 10.0 \
  --seed 42 \
  --clearml-iteration "$SOURCE_STEP" \
  --clearml-task-id "$TASK_ID" \
  "${EXTRA_ARGS[@]}"
