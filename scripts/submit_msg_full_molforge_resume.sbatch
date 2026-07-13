#!/usr/bin/env bash
#SBATCH --job-name=frigid-molforge-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

: "${FRIGID_REPO:?Set FRIGID_REPO to a clean Spectrum worktree}"
: "${MOLFORGE_REPO:?Set MOLFORGE_REPO to a clean Spectrum worktree}"
: "${MOLFORGE_PYTHON:?Set MOLFORGE_PYTHON to the prepared Python executable}"
: "${SOURCE_RUN_DIR:?Set SOURCE_RUN_DIR to the preserved prefix run}"
: "${RUN_DIR:?Set RUN_DIR to a new suffix run directory}"
: "${EXPECTED_MANIFEST:?Set EXPECTED_MANIFEST to the locked full manifest}"
: "${EXPECTED_FRIGID_COMMIT:?Set EXPECTED_FRIGID_COMMIT}"
: "${EXPECTED_MOLFORGE_COMMIT:?Set EXPECTED_MOLFORGE_COMMIT}"
: "${EXPECTED_MANIFEST_SHA256:?Set EXPECTED_MANIFEST_SHA256}"
: "${EXPECTED_SOURCE_CONFIG_SHA256:?Set EXPECTED_SOURCE_CONFIG_SHA256}"
: "${EXPECTED_SOURCE_PREFIX_SHA256:?Set EXPECTED_SOURCE_PREFIX_SHA256}"
: "${EXPECTED_MIST_CHECKPOINT_SHA256:?Set EXPECTED_MIST_CHECKPOINT_SHA256}"
: "${EXPECTED_MOLFORGE_CHECKPOINT_SHA256:?Set EXPECTED_MOLFORGE_CHECKPOINT_SHA256}"
: "${EXPECTED_PREFIX_ROWS:?Set EXPECTED_PREFIX_ROWS}"

SOURCE_CONFIG="${SOURCE_CONFIG:-$SOURCE_RUN_DIR/config.yaml}"
SOURCE_PREDICTIONS="${SOURCE_PREDICTIONS:-$SOURCE_RUN_DIR/output/thr_0p172/predictions.jsonl}"
RUN_MANIFEST="$RUN_DIR/RUN_MANIFEST.json"
THRESHOLD=0.172

if [[ -e "$RUN_DIR" ]]; then
  echo "MolForge suffix run directory already exists: $RUN_DIR" >&2
  exit 2
fi
for path in "$MOLFORGE_PYTHON" "$SOURCE_CONFIG" "$SOURCE_PREDICTIONS" \
  "$EXPECTED_MANIFEST"; do
  if [[ ! -e "$path" ]]; then
    echo "Required input does not exist: $path" >&2
    exit 2
  fi
done

verify_clean_commit() {
  local label="$1" repo="$2" expected="$3" actual
  actual="$(git -C "$repo" rev-parse HEAD)"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label commit mismatch: expected $expected, got $actual" >&2
    exit 2
  fi
  if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=all)" ]]; then
    echo "$label worktree must be clean: $repo" >&2
    exit 2
  fi
}

sha256_file() {
  "$MOLFORGE_PYTHON" - "$1" <<'PY'
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

verify_clean_commit "FRIGID" "$FRIGID_REPO" "$EXPECTED_FRIGID_COMMIT"
verify_clean_commit "MolForge" "$MOLFORGE_REPO" "$EXPECTED_MOLFORGE_COMMIT"
verify_sha256 "Locked manifest" "$EXPECTED_MANIFEST" "$EXPECTED_MANIFEST_SHA256"
verify_sha256 "Source config" "$SOURCE_CONFIG" "$EXPECTED_SOURCE_CONFIG_SHA256"
verify_sha256 "Source prefix" "$SOURCE_PREDICTIONS" "$EXPECTED_SOURCE_PREFIX_SHA256"

export FRIGID_REPO MOLFORGE_REPO MOLFORGE_PYTHON SOURCE_RUN_DIR RUN_DIR
export EXPECTED_MANIFEST EXPECTED_FRIGID_COMMIT EXPECTED_MOLFORGE_COMMIT
export EXPECTED_MANIFEST_SHA256 EXPECTED_SOURCE_CONFIG_SHA256
export EXPECTED_SOURCE_PREFIX_SHA256 EXPECTED_MIST_CHECKPOINT_SHA256
export EXPECTED_MOLFORGE_CHECKPOINT_SHA256 EXPECTED_PREFIX_ROWS
export SOURCE_CONFIG SOURCE_PREDICTIONS RUN_MANIFEST THRESHOLD

mkdir -p "$RUN_DIR"
"$MOLFORGE_PYTHON" - <<'PY'
import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


run_dir = Path(os.environ["RUN_DIR"])
expected_manifest = Path(os.environ["EXPECTED_MANIFEST"])
source_predictions = Path(os.environ["SOURCE_PREDICTIONS"])
source_config = Path(os.environ["SOURCE_CONFIG"])
prefix_rows = int(os.environ["EXPECTED_PREFIX_ROWS"])

expected = pd.read_csv(expected_manifest, sep="\t", dtype=str).fillna("")
if len(expected) != 17082 or expected["spec_name"].nunique() != 17082:
    raise ValueError("Locked manifest must contain 17,082 unique spectra")
expected_names = expected["spec_name"].astype(str).tolist()

prefix = [json.loads(line) for line in source_predictions.read_text().splitlines()]
prefix_names = [str(record.get("spec_name", "")) for record in prefix]
if len(prefix) != prefix_rows or prefix_names != expected_names[:prefix_rows]:
    raise ValueError("Preserved MolForge predictions are not the declared manifest prefix")
suffix_names = expected_names[prefix_rows:]
if not suffix_names:
    raise ValueError("No MolForge suffix remains to run")

config = yaml.safe_load(source_config.read_text())
labels_path = Path(config["data"]["labels_file"])
split_path = Path(config["data"]["split_file"])
mist_checkpoint = Path(config["mist_encoder"]["checkpoint"])
molforge_checkpoint = Path(config["molforge"]["checkpoint"])
if sha256_file(mist_checkpoint) != os.environ["EXPECTED_MIST_CHECKPOINT_SHA256"]:
    raise ValueError("MIST checkpoint hash mismatch")
if sha256_file(molforge_checkpoint) != os.environ["EXPECTED_MOLFORGE_CHECKPOINT_SHA256"]:
    raise ValueError("MolForge checkpoint hash mismatch")

labels = pd.read_csv(labels_path, sep="\t", dtype=str).fillna("")
splits = pd.read_csv(split_path, sep="\t", dtype=str).fillna("")
if labels["spec"].duplicated().any() or splits["name"].duplicated().any():
    raise ValueError("Source labels or split table contains duplicate spectrum IDs")
labels = labels.set_index("spec", drop=False).loc[suffix_names].reset_index(drop=True)
splits = splits.set_index("name", drop=False).loc[suffix_names].reset_index(drop=True)
if labels["spec"].astype(str).tolist() != suffix_names:
    raise ValueError("Suffix labels do not match the locked manifest order")
if splits["name"].astype(str).tolist() != suffix_names:
    raise ValueError("Suffix split does not match the locked manifest order")
if splits["split"].astype(str).unique().tolist() != ["test"]:
    raise ValueError("MolForge suffix contains non-test rows")

suffix_labels = run_dir / "labels_suffix.tsv"
suffix_split = run_dir / "split_suffix.tsv"
suffix_manifest = run_dir / "manifest_suffix.tsv"
suffix_config = run_dir / "config_suffix.yaml"
labels.to_csv(suffix_labels, sep="\t", index=False)
splits.to_csv(suffix_split, sep="\t", index=False)
expected.iloc[prefix_rows:].to_csv(suffix_manifest, sep="\t", index=False)
config["data"]["labels_file"] = str(suffix_labels)
config["data"]["split_file"] = str(suffix_split)
config.setdefault("evaluation", {})["split"] = "test"
config["evaluation"]["max_spectra"] = None
suffix_config.write_text(yaml.safe_dump(config, sort_keys=False))

manifest = {
    "schema_version": 1,
    "purpose": "frigid_full_molforge_suffix",
    "status": "running",
    "start_timestamp": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS", "unknown"),
    },
    "code": {
        "frigid_repo": str(Path(os.environ["FRIGID_REPO"]).resolve()),
        "frigid_commit": os.environ["EXPECTED_FRIGID_COMMIT"],
        "molforge_repo": str(Path(os.environ["MOLFORGE_REPO"]).resolve()),
        "molforge_commit": os.environ["EXPECTED_MOLFORGE_COMMIT"],
        "dirty": False,
    },
    "selection": {
        "split": "test",
        "start_index": prefix_rows,
        "max_spectra": len(suffix_names),
        "expected_total": len(expected_names),
    },
    "settings": {
        "threshold": float(os.environ["THRESHOLD"]),
        "batch_size": 1,
        "seed": 42,
        "device": "cuda:0",
        "target_fields_used_by_generation": [],
    },
    "inputs": {
        "expected_manifest": {
            "path": str(expected_manifest.resolve()),
            "sha256": os.environ["EXPECTED_MANIFEST_SHA256"],
        },
        "source_prefix": {
            "path": str(source_predictions.resolve()),
            "sha256": os.environ["EXPECTED_SOURCE_PREFIX_SHA256"],
            "row_count": prefix_rows,
        },
        "source_config": {
            "path": str(source_config.resolve()),
            "sha256": os.environ["EXPECTED_SOURCE_CONFIG_SHA256"],
        },
        "mist_checkpoint_sha256": os.environ["EXPECTED_MIST_CHECKPOINT_SHA256"],
        "molforge_checkpoint_sha256": os.environ["EXPECTED_MOLFORGE_CHECKPOINT_SHA256"],
        "suffix_config_sha256": sha256_file(suffix_config),
        "suffix_labels_sha256": sha256_file(suffix_labels),
        "suffix_split_sha256": sha256_file(suffix_split),
        "suffix_manifest_sha256": sha256_file(suffix_manifest),
    },
    "outputs": {},
}
Path(os.environ["RUN_MANIFEST"]).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

mark_failed() {
  local status="$?"
  trap - EXIT
  if [[ -f "$RUN_MANIFEST" ]]; then
    EXIT_CODE="$status" "$MOLFORGE_PYTHON" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["RUN_MANIFEST"])
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

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$MOLFORGE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
"$MOLFORGE_PYTHON" -m mist_molforge.benchmark \
  --config "$RUN_DIR/config_suffix.yaml" \
  --thresholds "$THRESHOLD" \
  --split test \
  --batch-size 1 \
  --device cuda:0 \
  --seed 42 \
  --output-dir "$RUN_DIR/output" \
  --num-workers 0

"$MOLFORGE_PYTHON" "$FRIGID_REPO/scripts/convert_molforge_predictions.py" \
  --input-jsonl "$SOURCE_PREDICTIONS" \
  --input-jsonl "$RUN_DIR/output/thr_0p172/predictions.jsonl" \
  --output-csv "$RUN_DIR/molforge_full_candidates.csv" \
  --spec-manifest "$EXPECTED_MANIFEST"

"$MOLFORGE_PYTHON" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


run_dir = Path(os.environ["RUN_DIR"])
expected = pd.read_csv(os.environ["EXPECTED_MANIFEST"], sep="\t", dtype=str).fillna("")
prefix_rows = int(os.environ["EXPECTED_PREFIX_ROWS"])
suffix_path = run_dir / "output/thr_0p172/predictions.jsonl"
suffix = [json.loads(line) for line in suffix_path.read_text().splitlines()]
suffix_names = [str(record.get("spec_name", "")) for record in suffix]
expected_suffix = expected["spec_name"].astype(str).tolist()[prefix_rows:]
if suffix_names != expected_suffix or len(suffix_names) != len(set(suffix_names)):
    raise ValueError("MolForge suffix output does not exactly cover the locked suffix")
if any(float(record.get("threshold", -1)) != float(os.environ["THRESHOLD"]) for record in suffix):
    raise ValueError("MolForge suffix threshold mismatch")

candidates_path = run_dir / "molforge_full_candidates.csv"
converter_manifest = run_dir / "molforge_full_candidates.csv.manifest.json"
converter = json.loads(converter_manifest.read_text(encoding="utf-8"))
if converter.get("query_count") != 17082:
    raise ValueError("Converted MolForge source does not cover 17,082 queries")
candidates = pd.read_csv(candidates_path)
if not set(candidates["query_spec_name"].astype(str)).issubset(
    set(expected["spec_name"].astype(str))
):
    raise ValueError("Converted MolForge candidates contain unknown query IDs")

manifest_path = Path(os.environ["RUN_MANIFEST"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["status"] = "completed"
manifest["exit_code"] = 0
manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
manifest["outputs"] = {
    "suffix_predictions": {
        "path": str(suffix_path.resolve()),
        "sha256": sha256_file(suffix_path),
        "row_count": len(suffix),
    },
    "suffix_metrics": {
        "path": str((run_dir / "output/thr_0p172/metrics.json").resolve()),
        "sha256": sha256_file(run_dir / "output/thr_0p172/metrics.json"),
    },
    "full_candidates": {
        "path": str(candidates_path.resolve()),
        "sha256": sha256_file(candidates_path),
        "row_count": len(candidates),
    },
    "converter_manifest": {
        "path": str(converter_manifest.resolve()),
        "sha256": sha256_file(converter_manifest),
    },
}
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

trap - EXIT
