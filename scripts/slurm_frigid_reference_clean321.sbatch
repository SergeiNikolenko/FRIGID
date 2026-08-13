#!/bin/bash
# The FRIGID reference number on OUR panel.
#
# Every comparison this project makes to FRIGID is to a README figure measured on
# MassSpecGym, which is a different axis. Nobody has ever run FRIGID's released
# DLM through its own sampler on the NPLIB1 clean 321 panel, so "our
# re-implementation is worth having" has never had a number to beat.
#
# Two arms, one variable: the conditioning FRIGID's decoder was actually trained
# with. The released checkpoint has formula_dropout_prob = 0.0 and never saw a
# formula-free input, so the fingerprint-only arm is the matched control for our
# own lane and the formula arm is FRIGID at its best.
#
# Cost, from the Slurm record: 24.7 s/spectrum on a whole A100, so 321 spectra is
# ~2.2 A100-hours per arm and ~4.4 for the pair.
#SBATCH --job-name=frigid-reference-clean321
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/frigid-reference-%A_%a.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
PYTHON="$CODE/.venv/bin/python"

CONDITIONINGS=(fingerprint formula-fingerprint)
CONDITIONING="${CONDITIONINGS[$SLURM_ARRAY_TASK_ID]}"

# The panel, not the whole val fold: val396 is burned (75 of its 396 rows share a
# connectivity block with the adaptation training set).
METADATA="$ROOT/runs/frigid-reference/clean321_metadata.csv"
FINGERPRINTS="/mnt/netstorage/nikolenko/marlin/runtime-inputs-v3-0a1a6afd/val/dreams_predictions.npz"
CHECKPOINT="$ROOT/checkpoints/DLM.ckpt"
OUTPUT="$ROOT/runs/frigid-reference/clean321-${CONDITIONING}-c8"

# randomness is passed explicitly on purpose: configs/spec2mol_benchmark_msg.yaml
# ships randomness: 10.0 on both FRIGID branches, and that default invalidated two
# long runs.
cd "$CODE"
env -u LD_PRELOAD PYTHONPATH="$CODE/src" "$PYTHON" scripts/evaluate_frigid_parity.py \
  --checkpoint "$CHECKPOINT" \
  --metadata "$METADATA" \
  --fingerprints "$FINGERPRINTS" \
  --fingerprint-key probs \
  --threshold 0.95 \
  --max-spectra 321 \
  --candidates 8 \
  --temperature 1.0 \
  --randomness 0.1 \
  --seed 42 \
  --conditioning "$CONDITIONING" \
  --output-dir "$OUTPUT"

echo "done ${CONDITIONING} -> ${OUTPUT}"
