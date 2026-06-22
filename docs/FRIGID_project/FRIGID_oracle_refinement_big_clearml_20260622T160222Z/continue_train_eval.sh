#!/usr/bin/env bash
set -euo pipefail
cd /home/nikolenko/work/Projects/FRIGID
RUN_DIR="repro_runs/oracle_refinement_big_clearml_20260622T160222Z"
CONFIG="configs/spec2mol_benchmark_msg.yaml"
WEIGHTS="repro_cache/DLM.ckpt"
HF_CACHE="repro_cache/hf"
STEPS=10000
BATCH=8

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
