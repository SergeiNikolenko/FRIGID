#!/bin/bash
# Record the real decode of a few spectra, attempt by attempt.
#
# This runs the ordinary clean-panel evaluation with the settings the
# 321-spectrum run used, and adds --trace-output so every token the block
# decoder commits and every ending it reaches is written down. The page built
# from it therefore shows the run itself, not a reconstruction of it.
#
# Usage:
#   sbatch scripts/slurm_marlin_attempt_fan.sbatch
#SBATCH --job-name=marlin-fan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/attempt-fan-%j.out

set -uo pipefail

CODE=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/code
RT=/mnt/netstorage/nikolenko/marlin/cache/runtime-inputs-spectrum-v1/16b1af5276034c041e85a4b7c43129a790b4fc091826485b691c93f9f7b699b3
CKPT=/mnt/netstorage/nikolenko/marlin/cache/checkpoints/control-r2/aed408c7d2c01c86a4b257e5119c28b11404971c3df3fd09a76c054ef0e7b14f/step=100000.ckpt
OUT=${OUT:-/mnt/netstorage/nikolenko/marlin/evaluations/attempt-fan}
PANEL=${PANEL:-$OUT/panel.tsv}
CANDIDATES=${CANDIDATES:-8}
CAP=${CAP:-1800}

cd "$CODE"
mkdir -p "$OUT"

env -u LD_PRELOAD \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
  ./.venv/bin/python -X faulthandler scripts/evaluate_marlin_nplib1.py \
  --checkpoint "$CKPT" --tokenizer "$RT/tokenizer.json" \
  --metadata "$RT/val/metadata.csv" \
  --fingerprints "$RT/val/dreams_predictions.npz" \
  --fingerprint-key probs --lane dreams --output-dir "$OUT" \
  --candidates "$CANDIDATES" --diversity-dropout 0.3 --temperature 1.0 \
  --generation-mode block --ppm-tolerance 10.0 --eos-boost 1.0 --seed 42 \
  --spec-manifest "$PANEL" \
  --threshold 0.95 --sample-tokens --soft-fingerprint \
  --mass-reachability-prune --forbid-isotope-tokens --restrict-organic-elements \
  --isotope-token omit --no-ema --per-spectrum-seconds "$CAP" \
  --trace-output "$OUT/trace.jsonl"

echo "ATTEMPT FAN COMPLETE"
