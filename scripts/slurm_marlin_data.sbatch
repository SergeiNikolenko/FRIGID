#!/bin/bash
#SBATCH --job-name=marlin-data
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/data-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
DATA="$ROOT/data"
mkdir -p "$DATA/nplib1"
cd "$DATA"

rm -f canopus_train_export_v2.tar
PYTHONPATH="$ROOT/vendor" $ROOT/.venv/bin/python \
  "$ROOT/code/scripts/download_zenodo_ranges.py" \
  https://zenodo.org/api/records/8316682/files/canopus_train_export_v2.tar/content \
  canopus_train_export_v2.tar \
  --size 4100802560 \
  --md5 161e3b92a266ee2a369ba1ae03990a24 \
  --workers 32 \
  --chunk-mib 32
sha256sum canopus_train_export_v2.tar > canopus_train_export_v2.tar.sha256
tar -xf canopus_train_export_v2.tar -C nplib1
find nplib1 -type f -printf '%s %p\n' | sort -nr > nplib1_inventory.txt
