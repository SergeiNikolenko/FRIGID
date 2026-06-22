#!/usr/bin/env bash
set -euo pipefail
cd /home/nikolenko/work/Projects/FRIGID
RUN_DIR="repro_runs/oracle_refinement_big_clearml_20260622T160222Z"
LABELS="repro_cache/msg/labels.tsv"
SPLITS="repro_cache/msg/split.tsv"
CONFIG="configs/spec2mol_benchmark_msg.yaml"
WEIGHTS="repro_cache/DLM.ckpt"
HF_CACHE="repro_cache/hf"
TRAIN_LIMIT=5000
VAL_LIMIT=1000
CANDIDATES=10
STEPS=10000
BATCH=8

mkdir -p "$RUN_DIR"
{
  echo "run_dir=$RUN_DIR"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$(git rev-parse HEAD)"
  echo "train_limit=$TRAIN_LIMIT"
  echo "val_limit=$VAL_LIMIT"
  echo "candidates=$CANDIDATES"
  echo "steps=$STEPS"
  echo "batch=$BATCH"
} > "$RUN_DIR/run_metadata.env"

PYTHONPATH=src .venv/bin/python scripts/build_oracle_candidate_predictions.py \
  --labels "$LABELS" \
  --splits "$SPLITS" \
  --split train \
  --output "$RUN_DIR/train_candidate_predictions.csv" \
  --limit "$TRAIN_LIMIT" \
  --candidates "$CANDIDATES" \
  > "$RUN_DIR/train_candidates.log" 2>&1

PYTHONPATH=src .venv/bin/python scripts/build_oracle_candidate_predictions.py \
  --labels "$LABELS" \
  --splits "$SPLITS" \
  --split val \
  --output "$RUN_DIR/val_candidate_predictions.csv" \
  --limit "$VAL_LIMIT" \
  --candidates "$CANDIDATES" \
  > "$RUN_DIR/val_candidates.log" 2>&1

PYTHONPATH=src .venv/bin/python scripts/build_oracle_traces.py \
  --predictions-csv "$RUN_DIR/train_candidate_predictions.csv" \
  --output-jsonl "$RUN_DIR/train_oracle_traces.jsonl" \
  --split train \
  --stats-json "$RUN_DIR/train_oracle_trace_stats.json" \
  --mcs-timeout-seconds 2 \
  > "$RUN_DIR/train_trace_build.log" 2>&1

PYTHONPATH=src .venv/bin/python scripts/build_oracle_traces.py \
  --predictions-csv "$RUN_DIR/val_candidate_predictions.csv" \
  --output-jsonl "$RUN_DIR/val_oracle_traces.jsonl" \
  --split val \
  --stats-json "$RUN_DIR/val_oracle_trace_stats.json" \
  --mcs-timeout-seconds 2 \
  > "$RUN_DIR/val_trace_build.log" 2>&1

# Wait for GPU 0 to be free enough for training. This avoids killing the active ICEBERG scaling run.
echo "waiting_for_gpu_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_DIR/run_metadata.env"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu_used_mb=$used gpu_util_pct=$util" >> "$RUN_DIR/gpu_wait.log"
  if [ "${used:-999999}" -lt 2000 ] && [ "${util:-100}" -lt 20 ]; then
    break
  fi
  sleep 120
done

echo "training_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_DIR/run_metadata.env"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/smoke_oracle_refinement_training.py \
  --config "$CONFIG" \
  --oracle-traces "$RUN_DIR/train_oracle_traces.jsonl" \
  --output-dir "$RUN_DIR/training" \
  --load-weights "$WEIGHTS" \
  --steps "$STEPS" \
  --batch-size "$BATCH" \
  --hf-cache-dir "$HF_CACHE" \
  --checkpoint-every-steps 1000 \
  --clearml \
  --clearml-project "FRIGID/oracle-refinement" \
  --clearml-task-name "${RUN_DIR#repro_runs/}" \
  > "$RUN_DIR/train.log" 2>&1

echo "training_finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_DIR/run_metadata.env"
CKPT="$RUN_DIR/training/checkpoints/oracle-refinement-step=10000.ckpt"
if [ ! -f "$CKPT" ]; then
  CKPT="$RUN_DIR/training/checkpoints/last.ckpt"
fi

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python scripts/refine_oracle_trace_candidates.py \
  --checkpoint "$CKPT" \
  --oracle-traces "$RUN_DIR/val_oracle_traces.jsonl" \
  --output-dir "$RUN_DIR/refinement_eval_val_step10000_limit256" \
  --limit 256 \
  --num-samples 4 \
  --fingerprint-source candidate \
  > "$RUN_DIR/eval.log" 2>&1

echo "eval_finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_DIR/run_metadata.env"
cat "$RUN_DIR/refinement_eval_val_step10000_limit256/refinement_summary.json" > "$RUN_DIR/final_eval_summary.json"
