#!/bin/bash

set -euo pipefail

SHARED_ROOT="${MARLIN_SHARED_ROOT:-/mnt/netstorage/nikolenko/marlin}"
RUNTIME_ROOT="${MARLIN_RUNTIME_INPUT_ROOT:-$SHARED_ROOT/runtime-inputs-v1}"
CHECKPOINT="${MARLIN_EVAL_CHECKPOINT:-}"
SOURCE_STEP="${MARLIN_EVAL_STEP:?MARLIN_EVAL_STEP is required}"
TASK_ID="${CLEARML_TASK_ID:?CLEARML_TASK_ID is required}"
OUTPUT_ROOT="$SHARED_ROOT/evaluations/faro-paper-gate-$TASK_ID"
EVALUATION_PROFILE="${MARLIN_EVAL_PROFILE:-screening}"
EVALUATION_SEED="${MARLIN_EVAL_SEED:-42}"
if [[ "$EVALUATION_PROFILE" == "paper-parity" ]]; then
  EVALUATION_CANDIDATES="${MARLIN_EVAL_CANDIDATES:-384}"
else
  EVALUATION_CANDIDATES="${MARLIN_EVAL_CANDIDATES:-16}"
fi

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}.clearai.innopolis.university,.university.innopolis.ru"
export no_proxy="${no_proxy:+$no_proxy,}.clearai.innopolis.university,.university.innopolis.ru"
export PYTHONFAULTHANDLER=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/materialize_marlin_runtime_inputs.py

if [[ ! -f "$CHECKPOINT" && -n "${MARLIN_EVAL_CHECKPOINT_MODEL_ID:-}" ]]; then
  CHECKPOINT="$(
    python scripts/materialize_marlin_checkpoint.py \
      --model-id "$MARLIN_EVAL_CHECKPOINT_MODEL_ID" \
      --cache-root "$SHARED_ROOT/checkpoints/marlin" \
      --output-name "step=$SOURCE_STEP.ckpt"
  )"
elif [[ ! -f "$CHECKPOINT" && -n "${MARLIN_EVAL_CHECKPOINT_TASK_ID:-}" ]]; then
  CHECKPOINT="$(
    python scripts/materialize_marlin_checkpoint.py \
      --task-id "$MARLIN_EVAL_CHECKPOINT_TASK_ID" \
      --artifact-name "${MARLIN_EVAL_CHECKPOINT_ARTIFACT_NAME:?MARLIN_EVAL_CHECKPOINT_ARTIFACT_NAME is required}" \
      --expected-sha256 "${MARLIN_EVAL_CHECKPOINT_SHA256:?MARLIN_EVAL_CHECKPOINT_SHA256 is required}" \
      --cache-root "$SHARED_ROOT/checkpoints/marlin"
  )"
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "MARLIN evaluation checkpoint is unavailable on this worker: ${CHECKPOINT:-<unset>}" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "MARLIN evaluation output already exists: $OUTPUT_ROOT" >&2
  exit 2
fi

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
if [[ "${MARLIN_EVAL_SOFT_FINGERPRINT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--soft-fingerprint)
fi

PANEL_ARGS=()
if [[ -n "${MARLIN_EVAL_SPEC_MANIFEST:-}" ]]; then
  PANEL_ARGS+=(--spec-manifest "$MARLIN_EVAL_SPEC_MANIFEST")
else
  PANEL_ARGS+=(--max-spectra "${MARLIN_EVAL_MAX_SPECTRA:-4}")
fi

python -X faulthandler scripts/evaluate_marlin_nplib1.py \
  --checkpoint "$CHECKPOINT" \
  --tokenizer "$RUNTIME_ROOT/tokenizer.json" \
  --metadata "$RUNTIME_ROOT/${MARLIN_EVAL_METADATA_REL:-val/metadata.csv}" \
  --fingerprints "$RUNTIME_ROOT/${MARLIN_EVAL_FINGERPRINTS_REL:-val/fingerprints.npz}" \
  --fingerprint-key "${MARLIN_EVAL_FINGERPRINT_KEY:-ground_truth}" \
  --lane dreams \
  --output-dir "$OUTPUT_ROOT" \
  --candidates "$EVALUATION_CANDIDATES" \
  --diversity-dropout "${MARLIN_EVAL_DIVERSITY_DROPOUT:-0.3}" \
  --temperature 1.0 \
  --generation-mode "${MARLIN_EVAL_GENERATION_MODE:-block}" \
  --ppm-tolerance 10.0 \
  --seed "$EVALUATION_SEED" \
  --clearml-iteration "$SOURCE_STEP" \
  --clearml-task-id "$TASK_ID" \
  --evaluation-profile "$EVALUATION_PROFILE" \
  "${PANEL_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

if [[ "$EVALUATION_PROFILE" == "paper-parity" ]]; then
  python -X faulthandler scripts/evaluate_marlin_mces.py \
    --predictions "$OUTPUT_ROOT/predictions.jsonl" \
    --output-dir "$OUTPUT_ROOT/mces" \
    --workers "${MARLIN_EVAL_MCES_WORKERS:-8}" \
    --clearml-task-id "$TASK_ID" \
    --clearml-iteration "$SOURCE_STEP"
fi
