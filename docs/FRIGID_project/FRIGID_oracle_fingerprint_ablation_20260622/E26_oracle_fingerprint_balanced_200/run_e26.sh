#!/usr/bin/env bash
set -euo pipefail
cd /home/nikolenko/work/Projects/FRIGID
source .venv/bin/activate
OUT=repro_runs/20260618T_spectrum_msg_paper_repro/experiments/E26_oracle_fingerprint_balanced_200
PATCHED_CKPT=repro_runs/20260618T_spectrum_msg_paper_repro/experiments/E24_downstream_threshold_balanced_200/DLM_local_hfcache.ckpt
{
  echo "E26 oracle fingerprint balanced 200"
  echo "START_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "HOST $(hostname)"
  echo "GIT_COMMIT $(git rev-parse HEAD)"
  echo "PATCHED_CKPT $PATCHED_CKPT"
  echo "GPU_STATUS_BEGIN"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  echo "GPU_STATUS_END"
} | tee "$OUT/run_meta.log"
HF_HOME=/home/nikolenko/.cache/huggingface \
HF_HUB_CACHE=/home/nikolenko/.cache/huggingface/hub \
TRANSFORMERS_CACHE=/home/nikolenko/.cache/huggingface/hub \
/usr/bin/time -f 'WALL_SECONDS %e' python scripts/benchmark_spec2mol.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --mist-checkpoint repro_cache/mist_msg.pt \
  --dlm-checkpoint "$PATCHED_CKPT" \
  --data-dir repro_cache/msg \
  --split test \
  --subset-csv .thoughts/subsets/mist_quality_balanced_200.csv \
  --formula-matches 3 \
  --max-attempts 30 \
  --batch-size 16 \
  --seed 42 \
  --fp-threshold 0.187 \
  --use-shared-cross-attention \
  --token-model /home/nikolenko/work/Projects/FRIGID/token_models/models/best_ngboost_MSG.joblib \
  --sigma-lambda 3.0 \
  --use-oracle-fingerprint \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"
{
  echo "END_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "GPU_STATUS_FINAL"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
} | tee -a "$OUT/run_meta.log"
