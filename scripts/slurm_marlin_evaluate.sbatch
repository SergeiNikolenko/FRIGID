#!/bin/bash
#SBATCH --job-name=marlin-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/eval-%A_%a.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
PYTHON=$ROOT/.venv/bin/python
CHECKPOINT="${MARLIN_CHECKPOINT:-$ROOT/runs/decoder/checkpoints/step=100000.ckpt}"
OUTPUT_TAG="${MARLIN_OUTPUT_TAG:-full}"
MAX_SPECTRA="${MARLIN_MAX_SPECTRA:-}"
CANDIDATES="${MARLIN_CANDIDATES:-384}"
DISABLE_GRAMMAR="${MARLIN_DISABLE_GRAMMAR:-0}"
DISABLE_MASS_SHELL="${MARLIN_DISABLE_MASS_SHELL:-0}"
TEMPERATURE="${MARLIN_TEMPERATURE:-1.0}"
DIVERSITY_DROPOUT="${MARLIN_DIVERSITY_DROPOUT:-0.3}"
PPM_TOLERANCE="${MARLIN_PPM_TOLERANCE:-10}"
VALENCE_SLACK="${MARLIN_VALENCE_SLACK:-4}"
EOS_BOOST="${MARLIN_EOS_BOOST:-1.0}"
export PYTHONPATH="$CODE/src:$ROOT/vendor"
export TOKENIZERS_PARALLELISM=false

cd "$CODE"

COMMON_ARGS=(
  --checkpoint "$CHECKPOINT"
  --tokenizer "$ROOT/data/safe-gpt/tokenizer.json"
  --metadata "$ROOT/data/processed/test/metadata.csv"
  --candidates "$CANDIDATES"
  --diversity-dropout "$DIVERSITY_DROPOUT"
  --temperature "$TEMPERATURE"
  --ppm-tolerance "$PPM_TOLERANCE"
  --valence-slack "$VALENCE_SLACK"
  --eos-boost "$EOS_BOOST"
)
if [[ -n "$MAX_SPECTRA" ]]; then
  COMMON_ARGS+=(--max-spectra "$MAX_SPECTRA")
fi
if [[ "$DISABLE_GRAMMAR" == "1" ]]; then
  COMMON_ARGS+=(--disable-grammar-mask)
fi
if [[ "$DISABLE_MASS_SHELL" == "1" ]]; then
  COMMON_ARGS+=(--disable-mass-shell)
fi

if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  "$PYTHON" scripts/evaluate_marlin_nplib1.py \
    "${COMMON_ARGS[@]}" \
    --lane dreams \
    --fingerprints "$ROOT/runs/dreams/probe/test_predictions.npz" \
    --fingerprint-key probs \
    --threshold 0.95 \
    --output-dir "$ROOT/runs/evaluation/$OUTPUT_TAG/dreams"
else
  : "${MARLIN_MIST_FINGERPRINTS:?Set to a predicted-formula MIST export; the legacy ground-truth-formula oracle is forbidden}"
  : "${MARLIN_MIST_PROVENANCE:?Set to the MIST-CF predicted-formula provenance manifest}"
  if [[ "$MARLIN_MIST_FINGERPRINTS" == "$ROOT/runs/mist/test/fingerprints.npz" ]]; then
    echo "Refusing the legacy ground-truth-formula MIST oracle" >&2
    exit 2
  fi
  "$PYTHON" scripts/evaluate_marlin_nplib1.py \
    "${COMMON_ARGS[@]}" \
    --lane mist \
    --fingerprints "$MARLIN_MIST_FINGERPRINTS" \
    --fingerprint-key "${MARLIN_MIST_FINGERPRINT_KEY:-mist_binary}" \
    --lane-provenance "$MARLIN_MIST_PROVENANCE" \
    --output-dir "$ROOT/runs/evaluation/$OUTPUT_TAG/mist"
fi
