#!/bin/bash
# The decisive oracle-vs-DreaMS pair on the clean 321 panel, at a cap high enough
# that truncation stops censoring the result.
#
# Why this exists: the pair reported in DECODER_PROGRAM.md §12 ran at a 300 s cap
# on the pre-R1 tree and truncated 261/321 (oracle) and 298/321 (DreaMS), so both
# 19.00% and 1.25% are lower bounds. R1 (a766389) removed the full-support mask
# call, so the same panel is affordable at a cap that does not bind.
#
# The decoder is CPU bound -- one worker sits at ~100% of one core while the A100
# idles -- so this runs as N single-core shards on the login host and never takes
# a GPU queue slot.
#
# Usage: ARM=oracle|dreams CAP=1800 SHARDS=10 scripts/evaluate_clean321_pair_uncapped.sh
set -uo pipefail
cd /home/nikolenko/work/Projects/MARLIN_reproduction_20260717/code

RT=/mnt/netstorage/nikolenko/marlin/cache/runtime-inputs-spectrum-v1/16b1af5276034c041e85a4b7c43129a790b4fc091826485b691c93f9f7b699b3
CKPT=${CKPT:-/mnt/netstorage/nikolenko/marlin/cache/checkpoints/control-r2/aed408c7d2c01c86a4b257e5119c28b11404971c3df3fd09a76c054ef0e7b14f/step=100000.ckpt}
BASE=${BASE:-/mnt/netstorage/nikolenko/marlin/evaluations/uncapped-v1}
ARM=${ARM:-dreams}
CAP=${CAP:-1800}
SHARDS=${SHARDS:-10}
DEVICE=${DEVICE:-cpu}

case "$ARM" in
  # The oracle bundle is binary, so 0.95 and 0.5 select the same bits; 0.95 is
  # used so the only key that differs from the DreaMS arm's run_signature is the
  # fingerprint source, exactly as in the DECODER_PROGRAM.md 12 pair.
  oracle) FP="$RT/val/oracle_fingerprints.npz"; KEY=ground_truth; THR=0.95 ;;
  dreams) FP="$RT/val/dreams_predictions.npz"; KEY=probs;        THR=0.95 ;;
  *) echo "ARM must be oracle or dreams" >&2; exit 2 ;;
esac

ROOT="$BASE/clean321-${ARM}-c8-cap${CAP}"
SHARDDIR="$BASE/shards-s${SHARDS}"
mkdir -p "$ROOT" "$SHARDDIR"

env -u LD_PRELOAD ./.venv/bin/python - configs/benchmarks/nplib1_v1/nplib1_val_clean322_v1.tsv "$SHARDDIR" "$SHARDS" <<'PY'
import sys
from pathlib import Path
panel, outdir, shards = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
lines = Path(panel).read_text().splitlines()
header, rows = lines[0], lines[1:]
for i in range(shards):
    (outdir / f"shard{i:02d}.tsv").write_text("\n".join([header, *rows[i::shards]]) + "\n")
print(f"{len(rows)} spectra over {shards} interleaved shards", flush=True)
PY

echo "arm=$ARM cap=${CAP}s shards=$SHARDS device=$DEVICE -> $ROOT"
for i in $(seq -f "%02g" 0 $((SHARDS-1))); do
  OUT="$ROOT/shard$i"
  [ -f "$OUT/metrics.json" ] && continue
  env -u LD_PRELOAD \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
    ./.venv/bin/python -X faulthandler scripts/evaluate_marlin_nplib1.py \
    --checkpoint "$CKPT" --tokenizer "$RT/tokenizer.json" \
    --metadata "$RT/val/metadata.csv" \
    --fingerprints "$FP" --fingerprint-key "$KEY" --threshold "$THR" \
    --lane dreams --output-dir "$OUT" --device "$DEVICE" \
    --candidates 8 --diversity-dropout 0.3 --temperature 1.0 \
    --generation-mode block --ppm-tolerance 10.0 --eos-boost 1.0 --seed 42 \
    --spec-manifest "$SHARDDIR/shard$i.tsv" \
    --sample-tokens --soft-fingerprint \
    --mass-reachability-prune --forbid-isotope-tokens --restrict-organic-elements \
    --isotope-token omit --no-ema --per-spectrum-seconds "$CAP" \
    > "$ROOT/shard$i.log" 2>&1 &
  sleep 2
done
wait
echo "ALL SHARDS FINISHED $ARM"
cat "$ROOT"/shard*/predictions.jsonl > "$ROOT/predictions.jsonl"
echo "merged $(wc -l < "$ROOT/predictions.jsonl") rows"
