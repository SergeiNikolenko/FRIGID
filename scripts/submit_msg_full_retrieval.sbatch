#!/usr/bin/env bash
#SBATCH --job-name=frigid-retrieval-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

: "${FRIGID_REPO:?Set FRIGID_REPO to a clean Spectrum worktree}"
: "${FRIGID_ASSET_ROOT:?Set FRIGID_ASSET_ROOT}"
: "${RUN_DIR:?Set RUN_DIR to a new retrieval run directory}"
: "${EXPECTED_MANIFEST:?Set EXPECTED_MANIFEST}"
: "${EXPECTED_CODE_COMMIT:?Set EXPECTED_CODE_COMMIT}"
: "${EXPECTED_CONFIG_SHA256:?Set EXPECTED_CONFIG_SHA256}"
: "${EXPECTED_DATA_SPLIT_SHA256:?Set EXPECTED_DATA_SPLIT_SHA256}"
: "${EXPECTED_LABELS_SHA256:?Set EXPECTED_LABELS_SHA256}"
: "${EXPECTED_MIST_CHECKPOINT_SHA256:?Set EXPECTED_MIST_CHECKPOINT_SHA256}"
: "${EXPECTED_MANIFEST_SHA256:?Set EXPECTED_MANIFEST_SHA256}"

PYTHON_BIN="${PYTHON_BIN:-$FRIGID_ASSET_ROOT/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$FRIGID_REPO/configs/spec2mol_benchmark_msg.yaml}"
DATA_DIR="${DATA_DIR:-$FRIGID_ASSET_ROOT/data/msg}"
DATA_SPLIT_PATH="$DATA_DIR/split.tsv"
LABELS_PATH="$DATA_DIR/labels.tsv"
MIST_CHECKPOINT="${MIST_CHECKPOINT:-$FRIGID_ASSET_ROOT/repro_cache/mist_msg.pt}"
PIPELINE_MANIFEST="$RUN_DIR/RUN_MANIFEST.json"

if [[ -e "$RUN_DIR" ]]; then
  echo "Retrieval run directory already exists: $RUN_DIR" >&2
  exit 2
fi
for path in "$PYTHON_BIN" "$CONFIG_PATH" "$DATA_SPLIT_PATH" "$LABELS_PATH" \
  "$MIST_CHECKPOINT" "$EXPECTED_MANIFEST"; do
  if [[ ! -e "$path" ]]; then
    echo "Required input does not exist: $path" >&2
    exit 2
  fi
done

actual_commit="$(git -C "$FRIGID_REPO" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_CODE_COMMIT" ]]; then
  echo "Code commit mismatch: expected $EXPECTED_CODE_COMMIT, got $actual_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$FRIGID_REPO" status --porcelain --untracked-files=all)" ]]; then
  echo "FRIGID_REPO must be clean: $FRIGID_REPO" >&2
  exit 2
fi

sha256_file() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

verify_sha256() {
  local label="$1" path="$2" expected="$3" actual
  actual="$(sha256_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA-256 mismatch: expected $expected, got $actual" >&2
    exit 2
  fi
}

verify_sha256 "Config" "$CONFIG_PATH" "$EXPECTED_CONFIG_SHA256"
verify_sha256 "Data split" "$DATA_SPLIT_PATH" "$EXPECTED_DATA_SPLIT_SHA256"
verify_sha256 "Labels" "$LABELS_PATH" "$EXPECTED_LABELS_SHA256"
verify_sha256 "MIST checkpoint" "$MIST_CHECKPOINT" "$EXPECTED_MIST_CHECKPOINT_SHA256"
verify_sha256 "Expected manifest" "$EXPECTED_MANIFEST" "$EXPECTED_MANIFEST_SHA256"

export FRIGID_REPO FRIGID_ASSET_ROOT RUN_DIR EXPECTED_MANIFEST
export EXPECTED_CODE_COMMIT EXPECTED_CONFIG_SHA256 EXPECTED_DATA_SPLIT_SHA256
export EXPECTED_LABELS_SHA256 EXPECTED_MIST_CHECKPOINT_SHA256
export EXPECTED_MANIFEST_SHA256 CONFIG_PATH DATA_DIR DATA_SPLIT_PATH LABELS_PATH
export MIST_CHECKPOINT PIPELINE_MANIFEST

mkdir -p "$RUN_DIR"
"$PYTHON_BIN" - <<'PY'
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

manifest = {
    "schema_version": 1,
    "purpose": "frigid_full_train_only_retrieval",
    "status": "running",
    "start_timestamp": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS", "unknown"),
    },
    "code": {
        "repo": str(Path(os.environ["FRIGID_REPO"]).resolve()),
        "commit": os.environ["EXPECTED_CODE_COMMIT"],
        "dirty": False,
    },
    "expected_manifest": {
        "path": str(Path(os.environ["EXPECTED_MANIFEST"]).resolve()),
        "sha256": os.environ["EXPECTED_MANIFEST_SHA256"],
        "query_count": 17082,
    },
    "inputs": {
        "config_sha256": os.environ["EXPECTED_CONFIG_SHA256"],
        "data_split_sha256": os.environ["EXPECTED_DATA_SPLIT_SHA256"],
        "labels_sha256": os.environ["EXPECTED_LABELS_SHA256"],
        "mist_checkpoint_sha256": os.environ["EXPECTED_MIST_CHECKPOINT_SHA256"],
    },
    "retrieval_policy": {
        "library_split": "train",
        "query_split": "test",
        "query_train_molecule_overlap": 0,
        "top_k": 10,
        "formula_first": True,
        "similarity": "exact Morgan-4096 Tanimoto",
        "similarity_backend": "matrix",
        "target_fields_used_by_ranking": [],
    },
    "outputs": {},
}
Path(os.environ["PIPELINE_MANIFEST"]).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

mark_failed() {
  local status="$?"
  trap - EXIT
  if [[ -f "$PIPELINE_MANIFEST" ]]; then
    EXIT_CODE="$status" "$PYTHON_BIN" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["PIPELINE_MANIFEST"])
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["status"] = "failed"
manifest["exit_code"] = int(os.environ["EXIT_CODE"])
manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  fi
  exit "$status"
}
trap mark_failed EXIT

cd "$FRIGID_REPO"
"$PYTHON_BIN" scripts/export_mist_fingerprints.py \
  --config "$CONFIG_PATH" \
  --mist-checkpoint "$MIST_CHECKPOINT" \
  --data-dir "$DATA_DIR" \
  --fp-threshold 0.187 \
  --output-dir "$RUN_DIR/mist_export" \
  --split test \
  --spec-manifest "$EXPECTED_MANIFEST" \
  --batch-size 32 \
  --num-workers 4 \
  --seed 42

"$PYTHON_BIN" scripts/retrieve_train_candidates.py \
  --labels-tsv "$LABELS_PATH" \
  --split-tsv "$DATA_SPLIT_PATH" \
  --mist-metadata-csv "$RUN_DIR/mist_export/metadata.csv" \
  --mist-fingerprints-npz "$RUN_DIR/mist_export/fingerprints.npz" \
  --output-dir "$RUN_DIR/retrieval" \
  --top-k 10 \
  --fingerprint-bits 4096 \
  --fingerprint-radius 2 \
  --query-splits test \
  --similarity-backend matrix \
  --device cuda \
  --query-batch-size 256

"$PYTHON_BIN" - <<'PY'
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


run_dir = Path(os.environ["RUN_DIR"])
delimiter = "\t" if Path(os.environ["EXPECTED_MANIFEST"]).suffix == ".tsv" else ","
with Path(os.environ["EXPECTED_MANIFEST"]).open(newline="") as handle:
    expected = [row["spec_name"] for row in csv.DictReader(handle, delimiter=delimiter)]
metadata = pd.read_csv(run_dir / "mist_export/metadata.csv")
candidates = pd.read_csv(run_dir / "retrieval/candidate_scores.csv")
details = pd.read_csv(run_dir / "retrieval/detailed_results.csv")
if metadata["spec_name"].astype(str).tolist() != expected:
    raise ValueError("MIST metadata does not match the ordered full manifest")
if details["spec_name"].astype(str).tolist() != expected:
    raise ValueError("Retrieval details do not match the ordered full manifest")
counts = candidates.groupby("query_spec_name", sort=False).size()
if len(candidates) != 170820 or len(counts) != 17082 or not (counts == 10).all():
    raise ValueError("Retrieval candidate coverage is not exactly 10 x 17,082")
with np.load(run_dir / "mist_export/fingerprints.npz") as arrays:
    for key in ("mist_probs", "mist_binary", "ground_truth"):
        if arrays[key].shape != (17082, 4096):
            raise ValueError(f"Unexpected {key} shape: {arrays[key].shape}")

child_manifests = {
    "mist_export": run_dir / "mist_export/run_manifest.json",
    "retrieval": run_dir / "retrieval/run_manifest.json",
}
for name, path in child_manifests.items():
    child = json.loads(path.read_text(encoding="utf-8"))
    if child.get("status") != "completed" or child.get("code", {}).get("dirty") is not False:
        raise ValueError(f"{name} child manifest is not a clean completed run")

manifest_path = Path(os.environ["PIPELINE_MANIFEST"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["status"] = "completed"
manifest["exit_code"] = 0
manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
manifest["outputs"] = {
    "mist_export_manifest": {
        "path": str(child_manifests["mist_export"]),
        "sha256": sha256_file(child_manifests["mist_export"]),
    },
    "retrieval_manifest": {
        "path": str(child_manifests["retrieval"]),
        "sha256": sha256_file(child_manifests["retrieval"]),
    },
    "candidate_scores": {
        "path": str(run_dir / "retrieval/candidate_scores.csv"),
        "sha256": sha256_file(run_dir / "retrieval/candidate_scores.csv"),
        "row_count": len(candidates),
    },
}
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

trap - EXIT
