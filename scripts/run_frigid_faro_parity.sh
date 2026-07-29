#!/bin/bash

set -euo pipefail

SHARED_ROOT="${MARLIN_SHARED_ROOT:-/mnt/netstorage/nikolenko/marlin}"
RUNTIME_ROOT="${MARLIN_RUNTIME_INPUT_ROOT:-$SHARED_ROOT/runtime-inputs-v1}"
CHECKPOINT="${FRIGID_CHECKPOINT:-$SHARED_ROOT/checkpoints/frigid/DLM.ckpt}"
TASK_ID="${CLEARML_TASK_ID:?CLEARML_TASK_ID is required}"
OUTPUT_ROOT="$SHARED_ROOT/evaluations/frigid-parity-$TASK_ID"

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}.clearai.innopolis.university,.university.innopolis.ru"
export no_proxy="${no_proxy:+$no_proxy,}.clearai.innopolis.university,.university.innopolis.ru"
export PYTHONFAULTHANDLER=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$SHARED_ROOT/cache/huggingface}"

python scripts/materialize_marlin_runtime_inputs.py
test -f "$CHECKPOINT"
test ! -e "$OUTPUT_ROOT"

python -X faulthandler scripts/evaluate_frigid_parity.py \
  --checkpoint "$CHECKPOINT" \
  --metadata "$RUNTIME_ROOT/val/metadata.csv" \
  --fingerprints "$RUNTIME_ROOT/val/fingerprints.npz" \
  --fingerprint-key ground_truth \
  --output-dir "$OUTPUT_ROOT" \
  --candidates "${FRIGID_PARITY_CANDIDATES:-16}" \
  --max-spectra "${FRIGID_PARITY_MAX_SPECTRA:-4}" \
  --temperature "${FRIGID_PARITY_TEMPERATURE:-0.8}" \
  --randomness "${FRIGID_PARITY_RANDOMNESS:-0.5}" \
  --conditioning "${FRIGID_PARITY_CONDITIONING:-fingerprint}" \
  --ppm-tolerance 10.0 \
  --seed 42 \
  --clearml-task-id "$TASK_ID" \
  ${FRIGID_PARITY_ORACLE_TARGET_LENGTH:+--oracle-target-length}
