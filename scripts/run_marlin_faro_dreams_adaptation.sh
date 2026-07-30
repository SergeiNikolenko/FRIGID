#!/bin/bash

set -euo pipefail

SHARED_ROOT="${MARLIN_SHARED_ROOT:-/mnt/netstorage/nikolenko/marlin}"
RUNTIME_ROOT="${MARLIN_RUNTIME_INPUT_ROOT:?MARLIN_RUNTIME_INPUT_ROOT is required}"
MAX_STEPS="${MARLIN_ADAPT_MAX_STEPS:-250}"
CONDITIONING_STEPS="${MARLIN_ADAPT_CONDITIONING_STEPS:-25}"
GATE_INTERVAL="${MARLIN_ADAPT_GATE_INTERVAL:-250}"
TASK_ID="${CLEARML_TASK_ID:?CLEARML_TASK_ID is required}"
RUN_ROOT="$SHARED_ROOT/runs/faro-dreams-adaptation-$TASK_ID"

if (( CONDITIONING_STEPS < 0 || CONDITIONING_STEPS >= MAX_STEPS )); then
  echo "conditioning steps must be in [0, max_steps)" >&2
  exit 2
fi
CROSS_ATTENTION_STEPS=$((MAX_STEPS - CONDITIONING_STEPS))

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}.clearai.innopolis.university,.university.innopolis.ru"
export no_proxy="${no_proxy:+$no_proxy,}.clearai.innopolis.university,.university.innopolis.ru"
export PYTHONFAULTHANDLER=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/materialize_marlin_runtime_inputs.py

CHECKPOINT="$(
  python scripts/materialize_marlin_checkpoint.py \
    --task-id "${MARLIN_ADAPT_CHECKPOINT_TASK_ID:?MARLIN_ADAPT_CHECKPOINT_TASK_ID is required}" \
    --artifact-name "${MARLIN_ADAPT_CHECKPOINT_ARTIFACT_NAME:?MARLIN_ADAPT_CHECKPOINT_ARTIFACT_NAME is required}" \
    --expected-sha256 "${MARLIN_ADAPT_CHECKPOINT_SHA256:?MARLIN_ADAPT_CHECKPOINT_SHA256 is required}" \
    --cache-root "$SHARED_ROOT/checkpoints/marlin"
)"
test -f "$CHECKPOINT"
test ! -e "$RUN_ROOT"

python scripts/train_marlin.py \
  frigid_warm_start_checkpoint=null \
  frigid_warm_start_sha256=null \
  "initial_weights_checkpoint=${CHECKPOINT//=/\\=}" \
  initial_weights_architecture_upgrade=false \
  initial_weights_use_ema=false \
  data.tokenizer_file="$RUNTIME_ROOT/tokenizer.json" \
  data.exclude_inchikeys="$RUNTIME_ROOT/nplib1_test_inchikeys.csv" \
  data.length_audit_manifest="$RUNTIME_ROOT/length_audit.json" \
  data.hf_cache_dir="$SHARED_ROOT/cache/huggingface" \
  data.snapshot_manifest="$SHARED_ROOT/safe-gpt-16d0be9ad6177ae683a32a86204530e8ee624a0f/manifest.json" \
  data.filtered_prefix_cache_manifest="$SHARED_ROOT/cache/filtered-safe-prefix-v1/manifest.json" \
  +data.paired_metadata="$RUNTIME_ROOT/train/metadata.csv" \
  +data.paired_fingerprints="$RUNTIME_ROOT/train/dreams_predictions.npz" \
  +data.paired_fingerprint_key=probs \
  +data.paired_fingerprint_ids_key=spectrum_ids \
  +data.paired_fingerprint_threshold=0.90 \
  model.fingerprint_layer_norm=false \
  model.fingerprint_self_attention_layers=0 \
  model.frigid_compatible_layer_order=false \
  training.noise_probability=0.0 \
  +training.adapt_fingerprint=true \
  +training.conditioning_only_steps="$CONDITIONING_STEPS" \
  +training.cross_attention_only_steps="$CROSS_ATTENTION_STEPS" \
  optim.learning_rate="${MARLIN_ADAPT_LEARNING_RATE:-1.0e-5}" \
  evaluation.metadata="$RUNTIME_ROOT/val/metadata.csv" \
  evaluation.fingerprints="$RUNTIME_ROOT/val/dreams_predictions.npz" \
  evaluation.fingerprint_key=probs \
  +evaluation.fingerprint_threshold=0.90 \
  +evaluation.use_ema=false \
  evaluation.interval_steps="$GATE_INTERVAL" \
  evaluation.max_spectra="${MARLIN_ADAPT_EVAL_SPECTRA:-4}" \
  evaluation.candidates="${MARLIN_ADAPT_EVAL_CANDIDATES:-16}" \
  trainer.devices=1 \
  trainer.accumulate_grad_batches=32 \
  trainer.max_steps="$MAX_STEPS" \
  output.root="$RUN_ROOT" \
  output.checkpoints="$RUN_ROOT/checkpoints" \
  output.checkpoint_interval="$GATE_INTERVAL" \
  tracking.clearml.task_name=marlin-faro-dreams-adaptation

python scripts/publish_marlin_checkpoint.py \
  --checkpoint "$RUN_ROOT/checkpoints/step=${MAX_STEPS}.ckpt" \
  --artifact-name "adapted-step-${MAX_STEPS}.ckpt"
