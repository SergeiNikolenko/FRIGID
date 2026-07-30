#!/bin/bash

set -euo pipefail

SHARED_ROOT="${MARLIN_SHARED_ROOT:-/mnt/netstorage/nikolenko/marlin}"
RUNTIME_ROOT="${MARLIN_RUNTIME_INPUT_ROOT:-$SHARED_ROOT/runtime-inputs-v1}"
REVISION=16d0be9ad6177ae683a32a86204530e8ee624a0f
FILE_LIST_SHA256=ca5356076d4e6e4920a019d55af0a769b0984d736f3feb9799045f3c49244e6f
FRIGID_SHA256=b6177c2d43448380aba80ff41c01461ea34ca2ca93b213986954c5afb7f0f457
MAX_STEPS="${MARLIN_MAX_STEPS:-100}"
GATE_INTERVAL="${MARLIN_GATE_INTERVAL:-100}"
EVALUATION_INTERVAL="${MARLIN_EVALUATION_INTERVAL:-$GATE_INTERVAL}"
EVALUATION_ENABLED="${MARLIN_EVALUATION_ENABLED:-true}"
CHECKPOINT_INTERVAL="${MARLIN_CHECKPOINT_INTERVAL:-$GATE_INTERVAL}"
RESUME_CHECKPOINT="${MARLIN_RESUME_CHECKPOINT:-}"
RESUME_CHECKPOINT_TASK_ID="${MARLIN_RESUME_CHECKPOINT_TASK_ID:-}"
RESUME_CHECKPOINT_ARTIFACT="${MARLIN_RESUME_CHECKPOINT_ARTIFACT:-}"
RESUME_CHECKPOINT_SHA256="${MARLIN_RESUME_CHECKPOINT_SHA256:-}"
TASK_SUFFIX="${CLEARML_TASK_ID:-manual}"
RUN_ROOT="$SHARED_ROOT/runs/faro-paper-gate-$TASK_SUFFIX"
CACHE_ROOT="$SHARED_ROOT/cache/filtered-safe-prefix-v1"
CHECKPOINT_CACHE_ROOT="$SHARED_ROOT/cache/checkpoints"
GLOBAL_BATCH_SIZE=$((8 * 2 * 16))
STREAM_CACHE_ROWS="${MARLIN_STREAM_CACHE_ROWS:-$((MAX_STEPS * GLOBAL_BATCH_SIZE))}"
STREAM_CACHE_ROOT="$SHARED_ROOT/cache/shuffled-safe-stream-v1-$STREAM_CACHE_ROWS"

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}.clearai.innopolis.university,.university.innopolis.ru"
export no_proxy="${no_proxy:+$no_proxy,}.clearai.innopolis.university,.university.innopolis.ru"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/materialize_marlin_runtime_inputs.py

if [[ -n "$RESUME_CHECKPOINT" && -n "$RESUME_CHECKPOINT_TASK_ID" ]]; then
  echo "set either MARLIN_RESUME_CHECKPOINT or artifact-backed resume variables" >&2
  exit 2
fi
if [[ -n "$RESUME_CHECKPOINT_TASK_ID" ]]; then
  test -n "$RESUME_CHECKPOINT_ARTIFACT"
  test -n "$RESUME_CHECKPOINT_SHA256"
  RESUME_CHECKPOINT="$(
    python scripts/materialize_marlin_checkpoint.py \
      --task-id "$RESUME_CHECKPOINT_TASK_ID" \
      --artifact-name "$RESUME_CHECKPOINT_ARTIFACT" \
      --expected-sha256 "$RESUME_CHECKPOINT_SHA256" \
      --cache-root "$CHECKPOINT_CACHE_ROOT"
  )"
fi

python - \
  "$SHARED_ROOT/worker_cache_ready.json" \
  "$REVISION" \
  "$FILE_LIST_SHA256" \
  "$FRIGID_SHA256" <<'PY'
import json
import sys

path, revision, file_list_sha256, frigid_sha256 = sys.argv[1:]
ready = json.load(open(path))
expected = {
    "safe_gpt_revision": revision,
    "safe_gpt_file_list_sha256": file_list_sha256,
    "frigid_checkpoint_sha256": frigid_sha256,
}
for key, value in expected.items():
    if ready.get(key) != value:
        raise ValueError(f"worker cache {key} mismatch")
PY

if [[ ! -f "$CACHE_ROOT/manifest.json" ]]; then
  python scripts/build_marlin_filtered_prefix_cache.py \
    --snapshot-manifest "$SHARED_ROOT/safe-gpt-$REVISION/manifest.json" \
    --dataset datamol-io/safe-gpt \
    --revision "$REVISION" \
    --file-list-sha256 "$FILE_LIST_SHA256" \
    --tokenizer "$RUNTIME_ROOT/tokenizer.json" \
    --exclude-inchikeys "$RUNTIME_ROOT/nplib1_test_inchikeys.csv" \
    --rows 100000 \
    --output-dir "$CACHE_ROOT"
fi

if [[ ! -f "$STREAM_CACHE_ROOT/manifest.json" ]]; then
  python scripts/build_marlin_shuffled_stream_cache.py \
    --snapshot-manifest "$SHARED_ROOT/safe-gpt-$REVISION/manifest.json" \
    --filtered-prefix-manifest "$CACHE_ROOT/manifest.json" \
    --dataset datamol-io/safe-gpt \
    --revision "$REVISION" \
    --file-list-sha256 "$FILE_LIST_SHA256" \
    --tokenizer "$RUNTIME_ROOT/tokenizer.json" \
    --exclude-inchikeys "$RUNTIME_ROOT/nplib1_test_inchikeys.csv" \
    --max-length 256 \
    --seed 42 \
    --shuffle-buffer 100000 \
    --rows "$STREAM_CACHE_ROWS" \
    --output-dir "$STREAM_CACHE_ROOT"
fi

test ! -e "$RUN_ROOT"

INITIALIZATION_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  test -f "$RESUME_CHECKPOINT"
  INITIALIZATION_ARGS+=(
    "resume_checkpoint=${RESUME_CHECKPOINT//=/\\=}"
    "frigid_warm_start_checkpoint=null"
    "frigid_warm_start_sha256=null"
  )
fi

python scripts/train_marlin.py \
  data.tokenizer_file="$RUNTIME_ROOT/tokenizer.json" \
  data.exclude_inchikeys="$RUNTIME_ROOT/nplib1_test_inchikeys.csv" \
  data.length_audit_manifest="$RUNTIME_ROOT/length_audit.json" \
  data.hf_cache_dir="$SHARED_ROOT/cache/huggingface" \
  data.snapshot_manifest="$SHARED_ROOT/safe-gpt-$REVISION/manifest.json" \
  data.filtered_prefix_cache_manifest="$CACHE_ROOT/manifest.json" \
  data.shuffled_stream_cache_manifest="$STREAM_CACHE_ROOT/manifest.json" \
  evaluation.metadata="$RUNTIME_ROOT/val/metadata.csv" \
  evaluation.fingerprints="$RUNTIME_ROOT/val/fingerprints.npz" \
  evaluation.enabled="$EVALUATION_ENABLED" \
  evaluation.interval_steps="$EVALUATION_INTERVAL" \
  trainer.devices=2 \
  trainer.accumulate_grad_batches=16 \
  trainer.max_steps="$MAX_STEPS" \
  output.root="$RUN_ROOT" \
  output.checkpoints="$RUN_ROOT/checkpoints" \
  output.checkpoint_interval="$CHECKPOINT_INTERVAL" \
  tracking.clearml.task_name=marlin-faro-paper-gate \
  "${INITIALIZATION_ARGS[@]}"
