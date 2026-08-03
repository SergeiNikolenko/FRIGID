#!/usr/bin/env bash
set -euo pipefail
cd /home/nikolenko/work/Projects/FRIGID
BASE=/home/nikolenko/work/Projects/FRIGID/repro_runs/20260616T203000Z_msg_paper_repro/msg_base_full_ngboost
TESTROOT=$(ls -dt "$BASE"/batch_size_speed_test_* | head -1)
TOKEN_MODEL=/home/nikolenko/work/Projects/FRIGID/token_models/models/best_ngboost_MSG.joblib
for BS in 16 32 64; do
  OUT="$TESTROOT/bs_${BS}"
  LOG="$TESTROOT/bs_${BS}.log"
  rm -rf "$OUT"
  mkdir -p "$OUT"
  echo "START batch_size=$BS $(date '+%F %T')" | tee "$LOG"
  /usr/bin/time -f "WALL_SECONDS %e" .venv/bin/python scripts/benchmark_spec2mol.py \
    --config configs/spec2mol_benchmark_msg.yaml \
    --mist-checkpoint repro_cache/mist_msg.pt \
    --dlm-checkpoint repro_cache/DLM.ckpt \
    --data-dir "$BASE/shard_data/shard_020" \
    --output-dir "$OUT" \
    --formula-matches 10 \
    --max-attempts 100 \
    --batch-size "$BS" \
    --max-spectra 5 \
    --seed 42 \
    --use-shared-cross-attention \
    --token-model "$TOKEN_MODEL" \
    --sigma-lambda 3.0 >> "$LOG" 2>&1
  echo "DONE batch_size=$BS $(date '+%F %T')" | tee -a "$LOG"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a "$LOG"
done
