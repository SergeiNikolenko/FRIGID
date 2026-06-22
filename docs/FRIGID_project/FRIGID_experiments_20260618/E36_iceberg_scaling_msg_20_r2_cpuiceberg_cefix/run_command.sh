#!/usr/bin/env bash
set -euo pipefail
cd /home/nikolenko/work/Projects/FRIGID
source .venv/bin/activate
{
  echo "START_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "HOST $(hostname)"
  echo "COMMIT $(git rev-parse HEAD)"
  python - <<'PY'
import torch, importlib.metadata as im
print('ENV torch', torch.__version__)
for pkg in ['torchdata', 'dgl', 'ray', 'pathos']:
    try:
        print('ENV', pkg, im.version(pkg))
    except Exception as e:
        print('ENV', pkg, 'missing', e)
PY
  python scripts/spec2mol_scaling.py \
    --config configs/spec2mol_benchmark_msg.yaml \
    --mist-checkpoint repro_cache/mist_msg.pt \
    --dlm-checkpoint /home/nikolenko/work/Projects/FRIGID/repro_runs/oracle_fingerprint_ablation_20260622/DLM.local_hf_cache.ckpt \
    --data-dir repro_cache/msg \
    --output-dir repro_runs/20260622T_iceberg_scaling_msg_20_r2_cpuiceberg_cefix/results \
    --split test \
    --max-spectra 20 \
    --use-shared-cross-attention \
    --token-model token_models/models/best_ngboost_MSG.joblib \
    --sigma-lambda 3.0 \
    --iceberg-gen-ckpt repro_cache/checkpoints/iceberg/iceberg_msg_model1.ckpt \
    --iceberg-inten-ckpt repro_cache/checkpoints/iceberg/iceberg_msg_model2.ckpt \
    --iceberg-python-path /home/nikolenko/work/Projects/FRIGID/.venv/bin/python \
    --iceberg-batch-size 4 \
    --iceberg-cpu \
    --iceberg-results-dir repro_runs/20260622T_iceberg_scaling_msg_20_r2_cpuiceberg_cefix/iceberg_results \
    --batch-size 32 \
    --num-unique-to-refine 4 \
    --masks-per-molecule 2 \
    --num-rounds 2 \
    --mask-prob 0.5 \
    --collision-energies 10 20 30 40 50 \
    --max-output-preds 100 \
    --incl-unknown-instrument
  echo "DONE_UTC $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee repro_runs/20260622T_iceberg_scaling_msg_20_r2_cpuiceberg_cefix/run.log
