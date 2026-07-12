#!/usr/bin/env python
"""Export frozen DreaMS embeddings into the RankLoop table contract."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from frigid.rankloop_corpus import _git_revision, sha256_file  # noqa: E402
from frigid.rankloop_dreams import validate_dreams_embeddings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frozen DreaMS embeddings for RankLoop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mgf", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--preparation-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--logger-path")
    parser.add_argument("--expected-dimension", type=int, default=1024)
    return parser.parse_args()


def _load_dreams_runtime():
    try:
        from dreams.api import dreams_embeddings
        from dreams.definitions import PRETRAINED
    except Exception as exc:
        raise RuntimeError(
            "Could not import the DreaMS runtime. Run this stage in the pinned "
            "DreaMS environment with its source directory on PYTHONPATH."
        ) from exc
    return dreams_embeddings, Path(PRETRAINED)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    mgf_path = Path(args.mgf).expanduser().resolve()
    metadata_path = Path(args.metadata_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_path = output_dir / "metadata.csv"
    embeddings_path = output_dir / "embeddings.npz"
    logger_path = (
        Path(args.logger_path).expanduser().resolve()
        if args.logger_path
        else output_dir / "dreams_embeddings.log"
    )

    metadata = pd.read_csv(metadata_path, dtype={"spec_name": str})
    required_columns = {"embedding_index", "spec_name"}
    if missing := sorted(required_columns.difference(metadata.columns)):
        raise ValueError(f"DreaMS metadata is missing columns: {missing}")
    expected_indices = np.arange(len(metadata), dtype=np.int64)
    if not np.array_equal(metadata["embedding_index"].to_numpy(), expected_indices):
        raise ValueError("embedding_index must be consecutive and zero-based.")
    if metadata["spec_name"].duplicated().any():
        duplicate = metadata.loc[metadata["spec_name"].duplicated(), "spec_name"].iloc[0]
        raise ValueError(f"Duplicate spectrum in DreaMS metadata: {duplicate}")

    dreams_embeddings, pretrained_dir = _load_dreams_runtime()
    embedding_checkpoint = pretrained_dir / "embedding_model.ckpt"
    backbone_checkpoint = pretrained_dir / "ssl_model.ckpt"
    for checkpoint in (embedding_checkpoint, backbone_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing pinned DreaMS checkpoint: {checkpoint}")

    embeddings = dreams_embeddings(
        str(mgf_path),
        batch_size=args.batch_size,
        logger_pth=str(logger_path),
        store_embs=False,
    )
    embedding_array = validate_dreams_embeddings(
        embeddings,
        expected_rows=len(metadata),
        expected_dimension=args.expected_dimension,
    )
    np.savez_compressed(
        embeddings_path,
        spectrum_embeddings=embedding_array,
        spec_names=metadata["spec_name"].to_numpy(dtype=str),
    )
    if metadata_path != output_metadata_path:
        shutil.copyfile(metadata_path, output_metadata_path)

    preparation_manifest = (
        Path(args.preparation_manifest).expanduser().resolve()
        if args.preparation_manifest
        else metadata_path.parent / "run_manifest.json"
    )
    if not preparation_manifest.is_file():
        preparation_manifest = None
    import dreams.api as dreams_api

    dreams_api_path = Path(dreams_api.__file__).resolve()
    dreams_repo_root = dreams_api_path.parents[2]
    revision, dirty = _git_revision(PROJECT_ROOT)
    dreams_revision, dreams_dirty = _git_revision(dreams_repo_root)
    run_manifest = {
        "schema_version": 1,
        "stage": "rankloop_dreams_embeddings",
        "repo": {"commit": revision, "dirty": dirty},
        "dreams_repo": {
            "path": str(dreams_repo_root),
            "commit": dreams_revision,
            "dirty": dreams_dirty,
        },
        "parameters": vars(args),
        "inputs": {
            "spectra_mgf": {"path": str(mgf_path), "sha256": sha256_file(mgf_path)},
            "metadata_csv": {
                "path": str(metadata_path),
                "sha256": sha256_file(metadata_path),
            },
            "preparation_manifest": (
                {
                    "path": str(preparation_manifest),
                    "sha256": sha256_file(preparation_manifest),
                }
                if preparation_manifest
                else None
            ),
            "dreams_api": {
                "path": str(dreams_api_path),
                "sha256": sha256_file(dreams_api_path),
            },
            "embedding_checkpoint": {
                "path": str(embedding_checkpoint),
                "sha256": sha256_file(embedding_checkpoint),
            },
            "backbone_checkpoint": {
                "path": str(backbone_checkpoint),
                "sha256": sha256_file(backbone_checkpoint),
            },
        },
        "outputs": {
            "metadata_csv": str(output_metadata_path),
            "metadata_sha256": sha256_file(output_metadata_path),
            "embeddings_npz": str(embeddings_path),
            "embeddings_sha256": sha256_file(embeddings_path),
            "logger_path": str(logger_path),
            "row_count": len(metadata),
            "embedding_dimension": int(embedding_array.shape[1]),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(run_manifest["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
