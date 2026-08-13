#!/bin/bash
# Stage 1 of the corpus arm: fp2mol under a measured fingerprint corruption.
#
# This is the pair the decisive oracle-vs-predicted result asks for. The same
# control-r2 weights score 19.00% Exact@1 on the clean 321 panel under a true
# Morgan fingerprint and 1.25% under the DreaMS one (docs/DECODER_PROGRAM.md
# 12.2), so the decoder-side objective is "decode correctly under a *wrong*
# fingerprint", and the corruption law is the thing under test:
#
#   MARLIN_ERROR_MODEL_VARIANT=fitted          the measured out-of-fold law
#   MARLIN_ERROR_MODEL_VARIANT=fitted_control  the rate-matched uniform control
#
# The control holds the pooled sensitivity and the pooled false-positive rate of
# the fitted model and removes only the frequency dependence and the row latent,
# which is exactly CoRe-Gen's -4.77 pp ablation.
#
# Why here and not on the ClearML queue. Measured 2026-08-13 from inside a Slurm
# allocation on node `spectrum` (job 790):
#   * the 67 GB corpus, the fitted model, its control and the runtime inputs are
#     all visible on /mnt/netstorage --- the worker's /mnt/netstorage is a
#     different, per-worker ext4 that has none of them and 8% free space;
#   * one 1,048,576-row group decodes off the NFS in 0.189 s = 5.5 M rows/s;
#   * Fp2MolStream + the real collator deliver 599 molecules/s at 8 workers
#     against a training demand of 56.8 mol/s (global batch 256 at the fleet's
#     measured 4.51 s/step), i.e. 10.6x headroom;
#   * the fitted corruption costs 6.4 ms per batch of 32 on CPU, and it runs on
#     the training device.
# So the corpus does not need staging, subsetting or a cache: it streams.
#
# Usage (never evicts another user; it waits for the card):
#   MARLIN_ERROR_MODEL_VARIANT=fitted \
#     scripts/submit_when_gpu_idle.sh scripts/slurm_marlin_corpus_stage1.sbatch
#
#SBATCH --job-name=marlin-corpus-stage1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/marlin-corpus-stage1-%j.out

set -euo pipefail

PROJECT_ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="${MARLIN_CODE_ROOT:-$PROJECT_ROOT/code}"
SHARED_ROOT=/mnt/netstorage/nikolenko/marlin
RUNTIME_ROOT="$SHARED_ROOT/cache/runtime-inputs-spectrum-v1/16b1af5276034c041e85a4b7c43129a790b4fc091826485b691c93f9f7b699b3"
CORPUS="$SHARED_ROOT/safe-gpt-16d0be9ad6177ae683a32a86204530e8ee624a0f"
ERROR_MODEL_DIR="$SHARED_ROOT/artifacts/encoder-error-model-oof-v1"

# The released DLM in MARLIN layout. Not control-r2: control-r2 is 100,000 steps
# of the broken recipe (272.8x the displacement budget) and lives only on the
# ClearML worker's local disk, so warm-starting from it here would be both
# unreproducible and confounded (docs/DECODER_PROGRAM.md 14.4 item 8).
CHECKPOINT="${MARLIN_STAGE1_CHECKPOINT:-$SHARED_ROOT/checkpoints/frigid-warmstart-marlin/step=0.ckpt}"
CHECKPOINT_SHA256="${MARLIN_STAGE1_CHECKPOINT_SHA256:-71f463709a68071396b1b030923c17a98a12f9dfa51fc5fe89a620faaab35244}"

VARIANT="${MARLIN_ERROR_MODEL_VARIANT:-fitted}"
case "$VARIANT" in
  fitted)         ERROR_MODEL="$ERROR_MODEL_DIR/encoder_error_model_oof.npz" ;;
  fitted_control) ERROR_MODEL="$ERROR_MODEL_DIR/encoder_error_model_oof.npz" ;;
  *) echo "MARLIN_ERROR_MODEL_VARIANT must be fitted or fitted_control" >&2; exit 2 ;;
esac

MAX_STEPS="${MARLIN_MAX_STEPS:-10000}"
ROW_GROUPS="${MARLIN_CORPUS_ROW_GROUPS:-8}"
BATCH_SIZE="${MARLIN_BATCH_SIZE:-8}"
ACCUMULATE="${MARLIN_ACCUMULATE_GRAD_BATCHES:-32}"
NUM_WORKERS="${MARLIN_NUM_WORKERS:-8}"
RUN_ROOT="${MARLIN_RUN_ROOT:-$SHARED_ROOT/runs/corpus-stage1-$VARIANT-$SLURM_JOB_ID}"

cd "$CODE"
test -f "$RUNTIME_ROOT/manifest.json"
test -f "$CORPUS/manifest.json"
test -f "$ERROR_MODEL"
test -f "$CODE/data/nplib1_holdout_inchikeys_v2.csv"
test ! -e "$RUN_ROOT"

export MARLIN_PYTHON="$CODE/.venv/bin/python"
export CLEARML_OFFLINE_MODE="${MARLIN_CLEARML_OFFLINE_MODE:-1}"
export CLEARML_CACHE_DIR="${MARLIN_CLEARML_CACHE_DIR:-$PROJECT_ROOT/cache/clearml-offline/$SLURM_JOB_ID}"
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"

ROOT="$CODE"
source "$CODE/scripts/marlin_runtime_env.sh"

# The in-training panel evaluation is uncapped, pre-R1 in its own process and
# nothing consumes its result, so it runs once, at the end, and the run is
# bounded by the conditioning probe instead (docs/DECODER_PROGRAM.md 14.4 risk 1).
# evaluation.interval_steps must equal output.checkpoint_interval
# (src/marlin/periodic_evaluation.py:52).
"$MARLIN_PYTHON" -X faulthandler scripts/train_marlin_spectrum_adaptation.py \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --tokenizer "$RUNTIME_ROOT/tokenizer.json" \
  --metadata "$RUNTIME_ROOT/train/metadata.csv" \
  --fingerprints "$RUNTIME_ROOT/train/dreams_predictions.npz" \
  --fingerprint-key probs \
  --fingerprint-threshold 0.95 \
  --soft-fingerprint \
  --exclude-inchikeys "$RUNTIME_ROOT/nplib1_test_inchikeys.csv" \
  --exclude-metadata "$RUNTIME_ROOT/val/metadata.csv" \
  --validation-metadata "$RUNTIME_ROOT/val/metadata.csv" \
  --validation-fingerprints "$RUNTIME_ROOT/val/dreams_predictions.npz" \
  --validation-fingerprint-key probs \
  --validation-fingerprint-threshold 0.95 \
  --evaluation-manifest "$CODE/configs/benchmarks/nplib1_v1/nplib1_val_clean322_v1.tsv" \
  --output-dir "$RUN_ROOT" \
  --corpus-snapshot "$CORPUS" \
  --corpus-row-groups "$ROW_GROUPS" \
  --corpus-seed 0 \
  --corpus-exclude-inchikeys "$CODE/data/nplib1_holdout_inchikeys_v2.csv" \
  --fingerprint-noise-mode "$VARIANT" \
  --fingerprint-error-model "$ERROR_MODEL" \
  --noise-probability 1.0 \
  --max-steps "$MAX_STEPS" \
  --evaluation-interval "$MAX_STEPS" \
  --checkpoint-interval "$MAX_STEPS" \
  --evaluation-spectra 321 \
  --evaluation-candidates 8 \
  --evaluation-shards 16 \
  --batch-size "$BATCH_SIZE" \
  --accumulate-grad-batches "$ACCUMULATE" \
  --num-workers "$NUM_WORKERS" \
  --lr-schedule warmup_cosine \
  --derive-learning-rate \
  --lr-warmup-steps 1000 \
  --loss-reduction token_mean \
  --time-sampling per_sequence_antithetic \
  --fp32-forward \
  --conditioning-probe-interval 1000 \
  --conditioning-probe-metadata "$RUNTIME_ROOT/val/metadata.csv" \
  --conditioning-probe-fingerprints "$RUNTIME_ROOT/val/dreams_predictions.npz" \
  --conditioning-probe-fingerprint-key probs \
  --conditioning-probe-size 64 \
  --conditioning-probe-batch-size 8 \
  --probe-early-stopping-metric probe_top1_predicted \
  --probe-early-stopping-patience 4 \
  --mass-reachability-prune \
  --forbid-isotope-tokens \
  --restrict-organic-elements
