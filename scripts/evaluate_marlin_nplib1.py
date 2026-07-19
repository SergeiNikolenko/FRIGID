#!/usr/bin/env python3
"""Run clean-room MARLIN generation and evaluation on the NPLIB1 test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from dlm.utils.utils_chem import safe_to_smiles
from marlin.evaluation import load_fingerprints
from marlin.grammar import SafeGrammarMask
from marlin.mass_shell import MassShellConstraint
from marlin.model import MarlinDecoder, MarlinDecoderConfig
from marlin.sampler import MarlinSampler
from marlin.token_properties import build_token_property_table
from marlin.tokenizer import load_safe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--fingerprint-key", required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--lane", choices=("dreams", "mist"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=384)
    parser.add_argument("--diversity-dropout", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ppm-tolerance", type=float, default=10.0)
    parser.add_argument("--valence-slack", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-spectra", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ema_decoder(checkpoint_path: Path, device: torch.device) -> MarlinDecoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = MarlinDecoderConfig(**checkpoint["hyper_parameters"]["config"])
    model = MarlinDecoder(config)
    state = {
        key.removeprefix("decoder."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("decoder.")
    }
    model.load_state_dict(state, strict=True)
    ema = checkpoint.get("ema")
    if not ema:
        raise ValueError(f"checkpoint has no EMA state: {checkpoint_path}")
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    shadows = ema["shadow_params"]
    if len(parameters) != len(shadows):
        raise ValueError(
            f"EMA parameter count mismatch: model={len(parameters)} checkpoint={len(shadows)}"
        )
    with torch.no_grad():
        for parameter, shadow in zip(parameters, shadows):
            if parameter.shape != shadow.shape:
                raise ValueError(
                    f"EMA shape mismatch: model={tuple(parameter.shape)} "
                    f"checkpoint={tuple(shadow.shape)}"
                )
            parameter.copy_(shadow)
    return model.eval().to(device)


def morgan(molecule: Chem.Mol):
    return AllChem.GetMorganGenerator(radius=2, fpSize=4096).GetFingerprint(molecule)


def connectivity(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToInchiKey(molecule).split("-")[0]


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else float("nan")


def main() -> None:
    args = parse_args()
    if args.candidates <= 0:
        raise ValueError("--candidates must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata)
    if args.max_spectra is not None:
        metadata = metadata.iloc[: args.max_spectra].copy()
    fingerprints = load_fingerprints(
        args.fingerprints,
        args.fingerprint_key,
        args.threshold,
        metadata,
        allow_leading_subset=args.max_spectra is not None,
    )
    device = torch.device(args.device)
    model = load_ema_decoder(args.checkpoint, device)
    tokenizer = load_safe_tokenizer(args.tokenizer)
    special_ids = {
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.mask_token_id,
        tokenizer.pad_token_id,
    }
    token_masses, token_atoms, token_valences = build_token_property_table(
        len(tokenizer), tokenizer.convert_ids_to_tokens, special_ids
    )
    constraint = MassShellConstraint(
        token_masses,
        token_atoms,
        token_valences,
        ppm_tolerance=args.ppm_tolerance,
        valence_slack=args.valence_slack,
        eos_token_id=tokenizer.eos_token_id,
    )
    sampler = MarlinSampler(
        model,
        constraint,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        mask_token_id=tokenizer.mask_token_id,
        decode_tokens=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        safe_to_smiles=lambda safe: safe_to_smiles(safe, fix=True),
        grammar_mask=SafeGrammarMask(
            [tokenizer.convert_ids_to_tokens(index) for index in range(len(tokenizer))],
            lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
            eos_token_id=tokenizer.eos_token_id,
            special_token_ids=tuple(special_ids) + (tokenizer.unk_token_id,),
            ppm_tolerance=args.ppm_tolerance,
            valence_slack=args.valence_slack,
        ),
        forbidden_token_ids=(
            tokenizer.unk_token_id,
            tokenizer.bos_token_id,
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
        ),
    )

    predictions_path = args.output_dir / "predictions.jsonl"
    completed: set[str] = set()
    rows: list[dict] = []
    if predictions_path.exists():
        with predictions_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                completed.add(row["spec_name"])
                rows.append(row)

    with predictions_path.open("a") as output:
        for position, record in metadata.reset_index(drop=True).iterrows():
            spec_name = str(record["spec_name"])
            if spec_name in completed:
                continue
            started = time.perf_counter()
            ranked, stats = sampler.generate_ranked_with_stats(
                torch.from_numpy(fingerprints[position]),
                float(record["neutral_mass"]),
                candidates=args.candidates,
                diversity_dropout=args.diversity_dropout,
                temperature=args.temperature,
                generator=torch.Generator().manual_seed(args.seed + position),
            )
            elapsed = time.perf_counter() - started
            target_molecule = Chem.MolFromSmiles(record["smiles"])
            if target_molecule is None:
                raise ValueError(f"invalid target SMILES for {spec_name}")
            target_fingerprint = morgan(target_molecule)
            target_connectivity = str(record["inchikey_first_block"])
            candidates = []
            for candidate in ranked:
                molecule = Chem.MolFromSmiles(candidate.smiles)
                candidates.append(
                    {
                        "smiles": candidate.smiles,
                        "safe": candidate.safe,
                        "predicted_fingerprint_tanimoto": candidate.tanimoto,
                        "target_fingerprint_tanimoto": DataStructs.TanimotoSimilarity(
                            target_fingerprint, morgan(molecule)
                        ),
                        "mass_error_ppm": candidate.mass_error_ppm,
                        "exact_connectivity": connectivity(candidate.smiles)
                        == target_connectivity,
                    }
                )
            top_ten = candidates[:10]
            result = {
                "spec_name": spec_name,
                "lane": args.lane,
                "target_smiles": record["smiles"],
                "target_inchikey_first_block": target_connectivity,
                "neutral_mass": float(record["neutral_mass"]),
                "runtime_seconds": elapsed,
                "attempts": stats.attempts,
                "valid": stats.valid,
                "mass_valid": stats.mass_valid,
                "unique_mass_valid": stats.unique_mass_valid,
                "constraint_dead_ends": stats.constraint_dead_ends,
                "eos_terminated": stats.eos_terminated,
                "max_length_terminated": stats.max_length_terminated,
                "sample_terminal_safes": stats.sample_terminal_safes,
                "sample_dead_ends": stats.sample_dead_ends,
                "validity": stats.valid / stats.attempts,
                "mass_validity": stats.mass_valid / max(stats.valid, 1),
                "uniqueness": stats.unique_mass_valid / max(stats.mass_valid, 1),
                "exact_top1": bool(candidates and candidates[0]["exact_connectivity"]),
                "exact_top10": any(
                    candidate["exact_connectivity"] for candidate in top_ten
                ),
                "tanimoto_top1": candidates[0]["target_fingerprint_tanimoto"]
                if candidates
                else 0.0,
                "tanimoto_top10": max(
                    (candidate["target_fingerprint_tanimoto"] for candidate in top_ten),
                    default=0.0,
                ),
                "candidates": candidates,
            }
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            rows.append(result)
            print(
                f"{position + 1}/{len(metadata)} {spec_name} "
                f"unique={stats.unique_mass_valid} runtime={elapsed:.3f}s",
                flush=True,
            )

    metrics = {
        "lane": args.lane,
        "rows": len(rows),
        "exact_top1": mean(rows, "exact_top1"),
        "exact_top10": mean(rows, "exact_top10"),
        "tanimoto_top1": mean(rows, "tanimoto_top1"),
        "tanimoto_top10": mean(rows, "tanimoto_top10"),
        "validity": mean(rows, "validity"),
        "mass_validity": mean(rows, "mass_validity"),
        "uniqueness": mean(rows, "uniqueness"),
        "runtime_seconds_total": float(sum(row["runtime_seconds"] for row in rows)),
        "runtime_seconds_mean": mean(rows, "runtime_seconds"),
        "settings": {
            "candidates": args.candidates,
            "diversity_dropout": args.diversity_dropout,
            "temperature": args.temperature,
            "ppm_tolerance": args.ppm_tolerance,
            "valence_slack": args.valence_slack,
            "block_width": model.config.block_width,
            "ema": True,
            "grammar_mask": "inferred conservative lexical SAFE grammar",
            "grammar_decode_order": "inferred left-to-right within each block",
            "seed": args.seed,
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema_version": 1,
        "clean_room_reproduction": True,
        "author_code_available_at_start": False,
        "inferred_parameters": [
            "max_steps=100000",
            "mass Fourier frequency count",
            "conservative lexical SAFE grammar mask",
            "left-to-right token commitment within grammar-masked blocks",
        ],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "inputs": {
            str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (
                args.checkpoint,
                args.tokenizer,
                args.metadata,
                args.fingerprints,
            )
        },
        "metrics": metrics,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
