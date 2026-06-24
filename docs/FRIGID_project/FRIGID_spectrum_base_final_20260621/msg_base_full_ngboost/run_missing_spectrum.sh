#!/usr/bin/env bash
set -euo pipefail
cd /home/nikolenko/work/Projects/FRIGID
BASE=/home/nikolenko/work/Projects/FRIGID/repro_runs/20260616T203000Z_msg_paper_repro/msg_base_full_ngboost
TOKEN_MODEL=/home/nikolenko/work/Projects/FRIGID/token_models/models/best_ngboost_MSG.joblib
mkdir -p "$BASE/logs"
if [ ! -s "$TOKEN_MODEL" ]; then
  echo "Missing token model: $TOKEN_MODEL" >&2
  exit 44
fi
for task in $(seq 20 35); do
  IDX=$(printf '%03d' "$task")
  OUT_DIR="$BASE/shard_outputs/shard_${IDX}"
  DATA_DIR="$BASE/shard_data/shard_${IDX}"
  if [ -s "$OUT_DIR/aggregate_statistics.json" ]; then
    echo "SKIP $(date '+%F %T') shard_${IDX} already complete"
    continue
  fi
  rm -rf "$OUT_DIR"
  mkdir -p "$OUT_DIR"
  LOG="$BASE/logs/spectrum-shard-${IDX}.log"
  echo "START $(date '+%F %T') shard_${IDX} host=$(hostname) CVD=${CUDA_VISIBLE_DEVICES:-unset}" | tee "$LOG"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader | tee -a "$LOG"
  .venv/bin/python scripts/benchmark_spec2mol.py \
    --config configs/spec2mol_benchmark_msg.yaml \
    --mist-checkpoint repro_cache/mist_msg.pt \
    --dlm-checkpoint repro_cache/DLM.ckpt \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_DIR" \
    --formula-matches 10 \
    --max-attempts 100 \
    --batch-size 16 \
    --seed 42 \
    --use-shared-cross-attention \
    --token-model "$TOKEN_MODEL" \
    --sigma-lambda 3.0 >> "$LOG" 2>&1
  echo "DONE $(date '+%F %T') shard_${IDX}" | tee -a "$LOG"
done
.venv/bin/python "$BASE/aggregate_base_ngboost.py" > "$BASE/aggregate/aggregate_spectrum_resume.log" 2>&1
