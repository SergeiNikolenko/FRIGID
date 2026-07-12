#!/usr/bin/env python
"""Export frozen MIST h0 embeddings for RankLoop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from benchmark_spec2mol import (  # noqa: E402
    load_config,
    load_mist_encoder,
    load_spec_data,
    merge_config_with_args,
)
from dlm.utils.benchmark_selection import (  # noqa: E402
    load_spec_manifest,
    resolve_selected_indices,
)
from frigid.rankloop_corpus import _git_revision, sha256_file  # noqa: E402
from mist.data.datasets import get_paired_loader  # noqa: E402


class _FeaturizedSubset(torch.utils.data.Subset):
    def get_featurizer(self):
        return self.dataset.get_featurizer()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frozen MIST penultimate embeddings for RankLoop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/spec2mol_benchmark_msg.yaml")
    parser.add_argument("--mist-checkpoint")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--spec-manifest")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-spectra", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dlm-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--fp-threshold", type=float, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--softmax-temp", type=float, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--randomness", type=float, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--formula-matches", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--max-attempts", type=int, default=None, help=argparse.SUPPRESS
    )
    return parser.parse_args()


def resolve_indices(split_data, args: argparse.Namespace) -> list[int]:
    names = load_spec_manifest(args.spec_manifest) if args.spec_manifest else None
    return resolve_selected_indices(
        split_data,
        names,
        args.start_index,
        args.max_spectra,
    )


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = merge_config_with_args(load_config(args.config), args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, split_data = load_spec_data(
        config["data"],
        config["mist_encoder"],
        config["evaluation"]["split"],
        shuffle=False,
    )
    selected_indices = resolve_indices(split_data, args)
    subset = _FeaturizedSubset(dataset, selected_indices)
    loader = get_paired_loader(
        subset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    encoder = load_mist_encoder(config["mist_encoder"], device)

    rows: list[dict[str, object]] = []
    embeddings: list[np.ndarray] = []
    cursor = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Exporting MIST h0 embeddings"):
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            _, auxiliary = encoder(batch)
            hidden = auxiliary["h0"].detach().cpu().numpy().astype(np.float32)
            batch_indices = selected_indices[cursor : cursor + len(hidden)]
            if len(batch_indices) != len(hidden):
                raise AssertionError("MIST batch and selected index counts diverged.")
            for row_embedding, original_index in zip(hidden, batch_indices):
                spectrum, molecule = split_data[original_index]
                rows.append(
                    {
                        "embedding_index": len(rows),
                        "split": args.split,
                        "spec_name": spectrum.get_spec_name(),
                        "formula": spectrum.get_spectra_formula(),
                        "smiles": molecule.get_smiles(),
                        "inchi_key": molecule.get_inchikey(),
                        "instrument": spectrum.get_instrument(),
                    }
                )
                embeddings.append(row_embedding)
            cursor += len(hidden)
    if cursor != len(selected_indices) or not embeddings:
        raise AssertionError("MIST embedding export did not cover the selected rows.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    embeddings_path = output_dir / "embeddings.npz"
    metadata = pd.DataFrame(rows)
    metadata.to_csv(metadata_path, index=False)
    embedding_array = np.stack(embeddings)
    np.savez_compressed(
        embeddings_path,
        spectrum_embeddings=embedding_array,
        spec_names=metadata["spec_name"].to_numpy(dtype=str),
    )

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(config["mist_encoder"]["checkpoint"]).expanduser().resolve()
    manifest_path = (
        Path(args.spec_manifest).expanduser().resolve() if args.spec_manifest else None
    )
    revision, dirty = _git_revision(PROJECT_ROOT)
    manifest = {
        "schema_version": 1,
        "repo": {"commit": revision, "dirty": dirty},
        "device": str(device),
        "parameters": vars(args),
        "inputs": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "mist_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
            "spec_manifest": (
                {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
                if manifest_path
                else None
            ),
        },
        "outputs": {
            "metadata_csv": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "embeddings_npz": str(embeddings_path),
            "embeddings_sha256": sha256_file(embeddings_path),
            "row_count": len(metadata),
            "embedding_dimension": int(embedding_array.shape[1]),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
