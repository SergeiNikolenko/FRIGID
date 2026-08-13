#!/bin/bash
# End-to-end panel throughput at HEAD (post-R1), measured rather than assumed.
#
# DECODER_PROGRAM.md 14.1 records that R1 fixed the deadline overshoot
# (+36.6% -> +0.10%) but that end-to-end panel throughput at HEAD has never been
# measured, and 13.8 item 5 lists "re-price one clean-panel shard end to end on
# the post-R1 tree" as owed. This job pays that debt on the local A100 and
# samples per-process GPU utilisation while it runs, so the question "is the
# constrained decoder GPU work at all" gets an answer instead of an assumption.
#SBATCH --job-name=marlin-clean321-throughput
#SBATCH --partition=gpu-shared
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/marlin-throughput-%j.out

set -uo pipefail
ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
cd "$CODE"

RT=/mnt/netstorage/nikolenko/marlin/cache/runtime-inputs-spectrum-v1/16b1af5276034c041e85a4b7c43129a790b4fc091826485b691c93f9f7b699b3
CKPT=/mnt/netstorage/nikolenko/marlin/cache/checkpoints/control-r2/aed408c7d2c01c86a4b257e5119c28b11404971c3df3fd09a76c054ef0e7b14f/step=100000.ckpt
BASE=/mnt/netstorage/nikolenko/marlin/evaluations/uncapped-v1
MANIFEST=${MANIFEST:-$BASE/shards-s24/shard00.tsv}
OUT=${OUT:-$BASE/throughput-gpu-shard00}
CAP=${CAP:-1800}

mkdir -p "$OUT"
# Per-process SM%, so a co-resident job on the same card cannot be mistaken for ours.
nvidia-smi pmon -d 5 -s um -o DT > "$OUT/pmon.log" 2>&1 &
SAMPLER=$!

env -u LD_PRELOAD \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
  ./.venv/bin/python -X faulthandler scripts/evaluate_marlin_nplib1.py \
  --checkpoint "$CKPT" --tokenizer "$RT/tokenizer.json" \
  --metadata "$RT/val/metadata.csv" \
  --fingerprints "$RT/val/dreams_predictions.npz" \
  --fingerprint-key probs --threshold 0.95 --lane dreams \
  --output-dir "$OUT" --device cuda \
  --candidates 8 --diversity-dropout 0.3 --temperature 1.0 \
  --generation-mode block --ppm-tolerance 10.0 --eos-boost 1.0 --seed 42 \
  --spec-manifest "$MANIFEST" \
  --sample-tokens --soft-fingerprint \
  --mass-reachability-prune --forbid-isotope-tokens --restrict-organic-elements \
  --isotope-token omit --no-ema --per-spectrum-seconds "$CAP"
STATUS=$?

kill "$SAMPLER" 2>/dev/null
echo "throughput shard finished status=$STATUS -> $OUT"
