#!/bin/bash
# Shard a MARLIN evaluation across CPU cores.
#
# Why this exists: the constrained decoder is CPU bound, not GPU bound. A single
# worker sits at 99% of one core while the A100 idles at 2%, because the grammar mask
# and the mass reachability search run on the host. Sharding by spectrum therefore
# buys close to linear speedup on core count, which is what makes evaluating a full
# split routine instead of a multi-day campaign.
#
# Measured on the 803-spectrum locked test split at 8 candidates, step=100000:
#   one process, before the vocabulary restrictions ..... about 3.5 days
#   one process, after them ............................. about 17 h
#   16 shards, after them .............................. about 1.1 h
# The restrictions give 4.87x and the sharding 16x, so roughly 78x together.
#
# Spectra count and candidate budget are independent. Fix the split at the full
# panel, which is what makes a number comparable, and treat the budget as a ladder:
# CANDIDATES=8 lands in about an hour, 64 overnight, 384 is the paper's figure.
#
# Each worker is pinned to a single BLAS thread. Without that, 16 torch processes
# each grab all 24 cores, the load average hits 86, and the whole thing runs slower
# than serial.
#
# Usage:
#   CANDIDATES=8 CAP=1800 SHARDS=16 scripts/evaluate_marlin_sharded.sh <checkpoint>
set -uo pipefail
cd /home/nikolenko/work/Projects/MARLIN_reproduction_20260717/code

RT=/mnt/netstorage/nikolenko/marlin/cache/runtime-inputs-spectrum-v1/16b1af5276034c041e85a4b7c43129a790b4fc091826485b691c93f9f7b699b3
PANEL=${PANEL:-configs/benchmarks/nplib1_v1/nplib1_test_locked_full803_v1.tsv}
SPLIT=${SPLIT:-test}
CANDIDATES=${CANDIDATES:-8}
CAP=${CAP:-1800}
SHARDS=${SHARDS:-16}
CKPT="$1"
ROOT=${ROOT:-/mnt/netstorage/nikolenko/marlin/evaluations/sharded-${SPLIT}-c${CANDIDATES}}

mkdir -p "$ROOT/shards"

# Interleaved, so no single shard inherits the whole runtime tail.
env -u LD_PRELOAD ./.venv/bin/python - "$PANEL" "$ROOT/shards" "$SHARDS" <<'PY'
import sys
from pathlib import Path
panel, outdir, shards = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
lines = Path(panel).read_text().splitlines()
header, rows = lines[0], lines[1:]
for i in range(shards):
    (outdir / f"shard{i:02d}.tsv").write_text(
        "\n".join([header, *rows[i::shards]]) + "\n"
    )
print(f"{len(rows)} spectra over {shards} shards")
PY

echo "803 spectra x $CANDIDATES candidates, $SHARDS shards, cap ${CAP}s"
for i in $(seq -w 0 $((SHARDS-1))); do
  OUT="$ROOT/shard$i"
  [ -f "$OUT/metrics.json" ] && continue
  env -u LD_PRELOAD \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
    ./.venv/bin/python -X faulthandler scripts/evaluate_marlin_nplib1.py \
    --checkpoint "$CKPT" --tokenizer "$RT/tokenizer.json" \
    --metadata "$RT/$SPLIT/metadata.csv" \
    --fingerprints "$RT/$SPLIT/dreams_predictions.npz" \
    --fingerprint-key probs --lane dreams --output-dir "$OUT" \
    --candidates "$CANDIDATES" --diversity-dropout 0.3 --temperature 1.0 \
    --generation-mode block --ppm-tolerance 10.0 --eos-boost 1.0 --seed 42 \
    --spec-manifest "$ROOT/shards/shard$i.tsv" \
    --threshold 0.95 --sample-tokens --soft-fingerprint \
    --mass-reachability-prune --forbid-isotope-tokens --restrict-organic-elements \
    --isotope-token omit --no-ema --per-spectrum-seconds "$CAP" \
    > "$ROOT/shard$i.log" 2>&1 &
  sleep 2
done
wait
echo "ALL SHARDS FINISHED"

cat "$ROOT"/shard*/predictions.jsonl > "$ROOT/predictions.jsonl"
echo "merged $(wc -l < "$ROOT/predictions.jsonl") rows"

env -u LD_PRELOAD PYTHONPATH=src ./.venv/bin/python scripts/evaluate_marlin_mces.py \
  --predictions "$ROOT/predictions.jsonl" --output-dir "$ROOT/mces" --workers 10 \
  > "$ROOT/mces.log" 2>&1
env -u LD_PRELOAD PYTHONPATH=src ./.venv/bin/python scripts/score_frigid_convention.py \
  "full803-c${CANDIDATES}=$ROOT/predictions.jsonl" --json "$ROOT/frigid_convention.json"
echo "FULL 803 COMPLETE at $CANDIDATES candidates"
