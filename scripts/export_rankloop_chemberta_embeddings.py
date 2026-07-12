#!/usr/bin/env python
"""Precompute frozen ChemBERTa embeddings for RankLoop candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from frigid.rankloop_corpus import (  # noqa: E402
    _git_revision,
    sha256_file,
)


DEFAULT_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
DEFAULT_MODEL_REVISION = "dc423621097c5803bd6dc0fe57f6c2f6f0de36d6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frozen ChemBERTa embeddings for all corpus candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-corpus", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if pooling == "cls":
        return hidden_states[:, 0]
    if pooling != "mean":
        raise ValueError(f"Unsupported pooling mode: {pooling}")
    weights = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (hidden_states * weights).sum(dim=1) / denominator


def require_tied_input_output_embeddings(masked_language_model) -> None:
    input_embeddings = masked_language_model.get_input_embeddings()
    output_embeddings = masked_language_model.get_output_embeddings()
    if output_embeddings is None or (
        input_embeddings.weight.data_ptr() != output_embeddings.weight.data_ptr()
    ):
        raise RuntimeError(
            "ChemBERTa input embeddings are not tied to the loaded MLM decoder."
        )


def inchi_key_first_block_from_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid candidate SMILES: {smiles!r}")
    inchi_key = Chem.MolToInchiKey(molecule)
    if not inchi_key:
        raise ValueError(f"Could not compute candidate InChIKey: {smiles!r}")
    return inchi_key.split("-", maxsplit=1)[0]


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("batch_size and max_length must be positive.")
    corpus_path = Path(args.candidate_corpus).expanduser().resolve()
    frame = pd.read_csv(corpus_path, usecols=["candidate_smiles"], dtype=str)
    smiles_values = sorted(set(frame["candidate_smiles"].dropna().astype(str)))
    if not smiles_values:
        raise ValueError("Candidate corpus contains no SMILES values.")

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        trust_remote_code=False,
    )
    masked_language_model = AutoModelForMaskedLM.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        trust_remote_code=False,
    ).to(device)
    require_tied_input_output_embeddings(masked_language_model)
    model = masked_language_model.base_model
    model.eval()
    for parameter in masked_language_model.parameters():
        parameter.requires_grad_(False)

    embedding_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(smiles_values), args.batch_size),
            desc="Exporting ChemBERTa embeddings",
        ):
            batch_smiles = smiles_values[start : start + args.batch_size]
            tokens = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            outputs = model(**tokens)
            pooled = pool_hidden_states(
                outputs.last_hidden_state,
                tokens["attention_mask"],
                args.pooling,
            )
            embedding_batches.append(pooled.cpu().numpy().astype(np.float32))
    embeddings = np.concatenate(embedding_batches, axis=0)
    if embeddings.shape[0] != len(smiles_values) or not np.isfinite(embeddings).all():
        raise AssertionError("ChemBERTa embedding export is incomplete or non-finite.")

    metadata_rows = []
    for index, smiles in enumerate(smiles_values):
        metadata_rows.append(
            {
                "embedding_index": index,
                "smiles": smiles,
                "inchi_key_first_block": inchi_key_first_block_from_smiles(smiles),
            }
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    embeddings_path = output_dir / "embeddings.npz"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    np.savez_compressed(
        embeddings_path,
        molecule_embeddings=embeddings,
        smiles=np.asarray(smiles_values, dtype=str),
    )

    revision, dirty = _git_revision(PROJECT_ROOT)
    resolved_model_revision = getattr(model.config, "_commit_hash", None)
    manifest = {
        "schema_version": 1,
        "repo": {"commit": revision, "dirty": dirty},
        "device": str(device),
        "parameters": vars(args),
        "foundation_model": {
            "name": args.model_name,
            "requested_revision": args.model_revision,
            "resolved_revision": resolved_model_revision,
            "hidden_size": int(model.config.hidden_size),
            "frozen": True,
            "loader": "AutoModelForMaskedLM.base_model",
            "tied_input_output_embeddings": True,
        },
        "inputs": {
            "candidate_corpus": str(corpus_path),
            "candidate_corpus_sha256": sha256_file(corpus_path),
        },
        "outputs": {
            "metadata_csv": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "embeddings_npz": str(embeddings_path),
            "embeddings_sha256": sha256_file(embeddings_path),
            "molecule_count": len(smiles_values),
            "embedding_dimension": int(embeddings.shape[1]),
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
