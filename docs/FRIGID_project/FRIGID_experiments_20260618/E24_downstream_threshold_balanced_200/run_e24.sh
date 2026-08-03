#!/usr/bin/env bash
set -euo pipefail
cd /home/nikolenko/work/Projects/FRIGID
source .venv/bin/activate
OUT_ROOT=repro_runs/20260618T_spectrum_msg_paper_repro/experiments/E24_downstream_threshold_balanced_200
{
  echo "E24 downstream threshold balanced 200"
  echo "START_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "HOST $(hostname)"
  echo "GIT_COMMIT $(git rev-parse HEAD)"
  echo "GPU_STATUS_BEGIN"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  echo "GPU_STATUS_END"
} | tee "$OUT_ROOT/run_meta.log"
for THRESHOLD in 0.187 0.25 0.30; do
  SAFE=${THRESHOLD/./p}
  OUT="$OUT_ROOT/threshold_${SAFE}"
  mkdir -p "$OUT"
  echo "=== THRESHOLD $THRESHOLD START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee "$OUT/run.log"
  /usr/bin/time -f 'WALL_SECONDS %e' python scripts/benchmark_spec2mol.py \
    --config configs/spec2mol_benchmark_msg.yaml \
    --mist-checkpoint repro_cache/mist_msg.pt \
    --dlm-checkpoint repro_cache/DLM.ckpt \
    --data-dir repro_cache/msg \
    --split test \
    --subset-csv .thoughts/subsets/mist_quality_balanced_200.csv \
    --formula-matches 3 \
    --max-attempts 30 \
    --batch-size 16 \
    --seed 42 \
    --fp-threshold "$THRESHOLD" \
    --use-shared-cross-attention \
    --token-model /home/nikolenko/work/Projects/FRIGID/token_models/models/best_ngboost_MSG.joblib \
    --sigma-lambda 3.0 \
    --output-dir "$OUT" \
    2>&1 | tee -a "$OUT/run.log"
  echo "=== THRESHOLD $THRESHOLD END $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$OUT/run.log"
done
{
  echo "END_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "GPU_STATUS_FINAL"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
} | tee -a "$OUT_ROOT/run_meta.log"
