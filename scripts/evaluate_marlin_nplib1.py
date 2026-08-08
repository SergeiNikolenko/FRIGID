#!/usr/bin/env python3
"""Run clean-room MARLIN generation and evaluation on the NPLIB1 test split."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch

# Import SAFE before RDKit drawing libraries.  The FARO conda image otherwise
# resolves incompatible native expat symbols while SAFE imports wandb/IPython.
from dlm.utils.utils_chem import safe_to_smiles
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors

from marlin.expanding import ExpandingMarlinSampler
from marlin.expanding_checkpoint import expanding_model_from_checkpoint
from marlin.benchmark_selection import (
    hash_spec_names,
    load_spec_manifest,
    select_metadata,
)
from marlin.evaluation import (
    load_fingerprints,
    mass_bin_metrics,
    mean_metric,
    validate_mist_lane_provenance,
)
from marlin.grammar import SafeGrammarMask
from marlin.isotopes import theoretical_isotope_ratios
from marlin.mass_shell import MassShellConstraint
from marlin.model import MarlinDecoder, MarlinDecoderConfig
from marlin.sampler import MarlinSampler
from marlin.token_properties import (
    build_token_property_table,
    foreign_element_token_ids,
    isotope_token_ids,
)
from marlin.tokenizer import load_safe_tokenizer


def sampling_isotope_ratios(mode: str, record) -> tuple[float, float] | None:
    """Choose the isotope conditioning token supplied to the sampler.

    Training always emits this token, so ``omit`` leaves every sampling forward
    pass one conditioning token short of the training layout.
    """
    if mode == "omit":
        return None
    if mode == "zeros":
        return (0.0, 0.0)
    if mode == "oracle":
        molecule = Chem.MolFromSmiles(str(record["smiles"]))
        if molecule is None:
            raise ValueError("oracle isotope token needs a parsable target SMILES")
        return tuple(theoretical_isotope_ratios(molecule).tolist())
    raise ValueError(f"unknown isotope token mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--fingerprint-key", required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--soft-fingerprint",
        action="store_true",
        help="keep probability amplitudes on active bits while using threshold for sparsity",
    )
    parser.add_argument("--lane-provenance", type=Path)
    parser.add_argument("--formula-manifest", type=Path)
    parser.add_argument("--feature-bridge-manifest", type=Path)
    parser.add_argument("--mist-labels", type=Path)
    parser.add_argument("--lane", choices=("dreams", "mist"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=384)
    parser.add_argument("--candidate-batch-size", type=int)
    parser.add_argument("--diversity-dropout", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--generation-mode", choices=("block", "canvas"), default="block")
    parser.add_argument(
        "--architecture",
        choices=("marlin", "expanding"),
        default="marlin",
    )
    parser.add_argument("--expanding-steps", type=int, default=32)
    parser.add_argument("--ppm-tolerance", type=float, default=10.0)
    parser.add_argument("--valence-slack", type=float, default=4.0)
    parser.add_argument("--eos-boost", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-spectra", type=int)
    parser.add_argument(
        "--spec-manifest",
        type=Path,
        help="Ordered CSV/TSV panel; cannot be truncated by --max-spectra.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-grammar-mask", action="store_true")
    parser.add_argument(
        "--mass-reachability-prune",
        action="store_true",
        help=(
            "documented deviation: fold chemical mass reachability into the "
            "syntax mask instead of the paper's syntax-only support"
        ),
    )
    parser.add_argument(
        "--forbid-isotope-tokens",
        action="store_true",
        help=(
            "documented deviation: withhold support from bracket atoms carrying "
            "a mass number, which the monoisotopic mass shell can never accept"
        ),
    )
    parser.add_argument(
        "--restrict-organic-elements",
        action="store_true",
        help=(
            "documented deviation: withhold support from tokens introducing an "
            "element outside CHNOPS and the halogens, the set small-molecule MS "
            "structure elucidation works in"
        ),
    )
    parser.add_argument(
        "--per-spectrum-seconds",
        type=float,
        help=(
            "stop generating further candidates for a spectrum after this many "
            "seconds; truncation is recorded per row and counted in the metrics, "
            "never applied silently"
        ),
    )
    parser.add_argument("--disable-mass-shell", action="store_true")
    parser.add_argument("--sample-tokens", action="store_true")
    parser.add_argument("--fix-safe-decode", action="store_true")
    parser.add_argument(
        "--isotope-token",
        choices=("omit", "zeros", "oracle"),
        default="omit",
        help=(
            "conditioning parity with training, which always emits this token; "
            "'oracle' reads the target molecule and is a diagnostic only"
        ),
    )
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--layer0-long-residual-scale", type=float)
    parser.add_argument("--clearml-project")
    parser.add_argument("--clearml-task-name")
    parser.add_argument("--clearml-task-id")
    parser.add_argument("--clearml-iteration", type=int)
    parser.add_argument("--clearml-tag", action="append", default=[])
    parser.add_argument(
        "--evaluation-profile",
        choices=("screening", "paper-parity"),
        default="screening",
        help="ClearML metric namespace and evaluation-contract guard",
    )
    return parser.parse_args()


def validate_evaluation_profile(
    profile: str,
    *,
    candidates: int,
    spec_manifest: Path | None,
    max_spectra: int | None,
) -> None:
    """Reject configurations that could be mistaken for paper-parity results."""
    if profile == "screening":
        return
    if candidates != 384:
        raise ValueError("paper-parity evaluation requires exactly 384 candidates")
    if spec_manifest is None:
        raise ValueError("paper-parity evaluation requires a validation manifest")
    if max_spectra is not None:
        raise ValueError("paper-parity evaluation cannot truncate its manifest")
    if not spec_manifest.name.startswith("nplib1_val_"):
        raise ValueError(
            "paper-parity checkpoint selection requires an nplib1_val_* manifest"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def load_decoder(
    checkpoint_path: Path,
    device: torch.device,
    *,
    use_ema: bool = True,
    layer0_long_residual_scale: float | None = None,
) -> MarlinDecoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = MarlinDecoderConfig(**checkpoint["hyper_parameters"]["config"])
    checkpoint_state = checkpoint["state_dict"]
    if (
        "decoder.conditioner.fingerprint.layer_norm.weight"
        not in checkpoint_state
    ):
        config = replace(config, fingerprint_layer_norm=False)
    if layer0_long_residual_scale is not None:
        config = replace(
            config,
            layer0_long_residual_scale=layer0_long_residual_scale,
        )
    model = MarlinDecoder(config)
    state = {
        key.removeprefix("decoder."): value
        for key, value in checkpoint_state.items()
        if key.startswith("decoder.")
    }
    model.load_state_dict(state, strict=True)
    if not use_ema:
        return model.eval().to(device)
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


def add_formula_metrics(
    row: dict,
    *,
    target_formula: str | None = None,
) -> dict:
    """Add formula recall metrics to a prediction row, including resumed rows."""
    if target_formula is None:
        target_molecule = Chem.MolFromSmiles(row["target_smiles"])
        if target_molecule is None:
            raise ValueError(f"invalid target SMILES: {row['target_smiles']}")
        target_formula = rdMolDescriptors.CalcMolFormula(target_molecule)
    candidates = row.get("candidates", [])
    row["formula_top1"] = bool(
        candidates and candidates[0].get("formula") == target_formula
    )
    row["formula_top10"] = any(
        candidate.get("formula") == target_formula for candidate in candidates[:10]
    )
    return row


def formula_metric_summary(rows: list[dict]) -> dict[str, float]:
    rows_with_candidate = [row for row in rows if row["candidate_returned"]]
    return {
        "formula_top1_all": mean_metric(rows, "formula_top1"),
        "formula_top1_returned": mean_metric(
            rows_with_candidate, "formula_top1"
        ),
        "formula_top10_all": mean_metric(rows, "formula_top10"),
        "formula_top10_returned": mean_metric(
            rows_with_candidate, "formula_top10"
        ),
    }


def _clearml_candidate_table(
    rows: list[dict],
    *,
    candidates_per_spectrum: int = 3,
    max_rows: int = 100,
) -> pd.DataFrame:
    columns = [
        "spec_name",
        "rank",
        "target_smiles",
        "candidate_smiles",
        "exact",
        "formula_match",
        "target_tanimoto",
        "mass_error_ppm",
    ]
    table_rows = []
    for row in rows:
        target_molecule = Chem.MolFromSmiles(row["target_smiles"])
        if target_molecule is None:
            continue
        target_formula = rdMolDescriptors.CalcMolFormula(target_molecule)
        selected_candidates = row.get("candidates", [])[:candidates_per_spectrum]
        if not selected_candidates:
            table_rows.append(
                {
                    "spec_name": row["spec_name"],
                    "rank": None,
                    "target_smiles": row["target_smiles"],
                    "candidate_smiles": None,
                    "exact": False,
                    "formula_match": False,
                    "target_tanimoto": None,
                    "mass_error_ppm": None,
                }
            )
            if len(table_rows) >= max_rows:
                break
            continue
        for rank, candidate in enumerate(
            selected_candidates,
            start=1,
        ):
            table_rows.append(
                {
                    "spec_name": row["spec_name"],
                    "rank": rank,
                    "target_smiles": row["target_smiles"],
                    "candidate_smiles": candidate["smiles"],
                    "exact": bool(candidate["exact_connectivity"]),
                    "formula_match": candidate.get("formula") == target_formula,
                    "target_tanimoto": candidate[
                        "target_fingerprint_tanimoto"
                    ],
                    "mass_error_ppm": candidate["mass_error_ppm"],
                }
            )
            if len(table_rows) >= max_rows:
                return pd.DataFrame(table_rows, columns=columns)
    return pd.DataFrame(table_rows, columns=columns)


def _clearml_decoding_diagnostic_table(rows: list[dict]) -> pd.DataFrame:
    """Return bounded terminal decoder evidence for each evaluated spectrum."""
    records = []
    for row in rows:
        records.append(
            {
                "spec_name": row.get("spec_name"),
                "attempts": row.get("attempts", 0),
                "valid": row.get("valid", 0),
                "mass_valid": row.get("mass_valid", 0),
                "constraint_dead_ends": row.get("constraint_dead_ends", 0),
                "eos_terminated": row.get("eos_terminated", 0),
                "max_length_terminated": row.get("max_length_terminated", 0),
                "sample_terminal_safes": "\n".join(
                    row.get("sample_terminal_safes", [])[:5]
                ),
                "sample_dead_ends": json.dumps(
                    row.get("sample_dead_ends", [])[:5],
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def publish_clearml_evaluation(
    *,
    project_name: str | None,
    task_name: str | None,
    tags: list[str],
    metrics: dict,
    rows: list[dict],
    settings: dict,
    task_id: str | None = None,
    iteration: int | None = None,
    evaluation_label: str = "MARLIN",
    evaluation_profile: str | None = None,
) -> dict[str, str] | None:
    """Publish a completed evaluation only when explicitly configured."""
    if task_id is None and project_name is None and task_name is None:
        return None
    if task_id is None and (not project_name or not task_name):
        raise ValueError(
            "--clearml-project and --clearml-task-name must be provided together"
        )

    from clearml import Task

    attached = task_id is not None
    task = Task.get_task(task_id=task_id) if attached else Task.init(
        project_name=project_name, task_name=task_name,
        task_type=Task.TaskTypes.testing, tags=tags, reuse_last_task_id=False,
        output_uri=False, auto_connect_streams=False,
        auto_connect_frameworks=False, auto_resource_monitoring=False,
    )
    try:
        # Existing training tasks can already be completed when a deferred
        # evaluation attaches to them. ClearML rejects hyperparameter edits on
        # completed tasks, while scalar/table events remain valid.
        if not attached:
            task.connect(dict(settings), name="evaluation_settings")
        logger = task.get_logger()
        report_iteration = int(metrics["rows"] if iteration is None else iteration)
        metric_title = {
            "screening": "Molecular screening",
            "paper-parity": "Paper parity",
        }.get(evaluation_profile, "Molecular metrics")
        scalar_metrics = {
            "Exact@1": "exact_top1",
            "Exact@10": "exact_top10",
            "Formula@1": "formula_top1_all",
            "Formula@10": "formula_top10_all",
            "Formula@1 (returned)": "formula_top1_returned",
            "Formula@10 (returned)": "formula_top10_returned",
            "Candidate return": "candidate_return_rate",
            "Validity": "validity",
            "Completed validity": "completed_validity",
            "Mass validity": "mass_validity",
            "Uniqueness": "uniqueness",
            "Internal diversity": "internal_diversity",
            "Constraint dead ends": "constraint_dead_ends_mean",
            "EOS terminated": "eos_terminated_mean",
            "Max-length terminated": "max_length_terminated_mean",
            "Tanimoto@1 (returned)": "tanimoto_top1",
            "Tanimoto@10 (returned)": "tanimoto_top10",
        }
        for series, key in scalar_metrics.items():
            if key not in metrics:
                continue
            value = float(metrics[key])
            if not math.isfinite(value):
                continue
            logger.report_scalar(
                title=metric_title,
                series=series,
                value=value,
                iteration=report_iteration,
            )
        for mass_bin, values in metrics.get("mass_bins", {}).items():
            if "exact_top1" not in values:
                continue
            exact_top1 = float(values["exact_top1"])
            if math.isfinite(exact_top1):
                logger.report_scalar(
                    title=f"{metric_title} by mass bin",
                    series=f"{mass_bin} Exact@1",
                    value=exact_top1,
                    iteration=report_iteration,
                )
            logger.report_scalar(
                title=f"{metric_title} by mass bin",
                series=f"{mass_bin} rows",
                value=float(values["rows"]),
                iteration=report_iteration,
            )
        logger.report_table(
            title=f"{evaluation_label} molecular evaluation",
            series="Top candidates",
            iteration=report_iteration,
            table_plot=_clearml_candidate_table(rows),
        )
        logger.report_table(
            title=f"{evaluation_label} decoding diagnostics",
            series="Terminal samples",
            iteration=report_iteration,
            table_plot=_clearml_decoding_diagnostic_table(rows),
        )
        logger.report_table(
            title=f"{evaluation_label} provenance",
            series="Settings",
            iteration=report_iteration,
            table_plot=pd.DataFrame(
                [
                    {
                        "setting": key,
                        "value": json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value,
                    }
                    for key, value in sorted(settings.items())
                ]
            ),
        )
        molecules = []
        legends = []
        for row in rows:
            target = Chem.MolFromSmiles(row["target_smiles"])
            if target is not None:
                molecules.append(target)
                legends.append(f"{row['spec_name']} target")
            if row.get("candidates"):
                candidate = Chem.MolFromSmiles(row["candidates"][0]["smiles"])
                if candidate is not None:
                    molecules.append(candidate)
                    legends.append(f"{row['spec_name']} top-1")
        if molecules and hasattr(logger, "report_image"):
            from rdkit.Chem import Draw

            logger.report_image(
                title="Molecular generation",
                series="Targets and top-1 candidates",
                iteration=report_iteration,
                image=Draw.MolsToGridImage(molecules, legends=legends, molsPerRow=4),
            )
        return {
            "task_id": task.id,
            "task_name": task.name,
            "project_name": task.get_project_name()
            if hasattr(task, "get_project_name")
            else project_name,
            "web_url": task.get_output_log_web_page(),
        }
    finally:
        if not attached:
            task.close()


def main() -> None:
    args = parse_args()
    if args.candidates <= 0:
        raise ValueError("--candidates must be positive")
    if args.candidate_batch_size is not None and args.candidate_batch_size <= 0:
        raise ValueError("--candidate-batch-size must be positive")
    validate_evaluation_profile(
        args.evaluation_profile,
        candidates=args.candidates,
        spec_manifest=args.spec_manifest,
        max_spectra=args.max_spectra,
    )
    if args.clearml_task_id and (args.clearml_project or args.clearml_task_name):
        raise ValueError("--clearml-task-id is mutually exclusive with project/task name")
    if not args.clearml_task_id and bool(args.clearml_project) != bool(args.clearml_task_name):
        raise ValueError(
            "--clearml-project and --clearml-task-name must be provided together"
        )
    if args.lane == "mist" and any(
        value is None
        for value in (
            args.lane_provenance,
            args.formula_manifest,
            args.feature_bridge_manifest,
            args.mist_labels,
        )
    ):
        raise ValueError(
            "canonical MIST evaluation requires --lane-provenance, "
            "--formula-manifest, --feature-bridge-manifest, and --mist-labels"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata)
    metadata_rows = len(metadata)
    if args.lane == "mist":
        validate_mist_lane_provenance(
            args.lane_provenance,
            metadata_rows,
            args.fingerprints,
            args.metadata,
            args.formula_manifest,
            args.feature_bridge_manifest,
            args.mist_labels,
        )
    manifest_names = (
        load_spec_manifest(args.spec_manifest) if args.spec_manifest else None
    )
    metadata = select_metadata(metadata, manifest_names, args.max_spectra)
    fingerprints = load_fingerprints(
        args.fingerprints,
        args.fingerprint_key,
        args.threshold,
        metadata,
        allow_leading_subset=args.max_spectra is not None,
        preserve_probabilities=args.soft_fingerprint,
    )
    device = torch.device(args.device)
    model = load_decoder(
        args.checkpoint,
        device,
        use_ema=not args.no_ema,
        layer0_long_residual_scale=args.layer0_long_residual_scale,
    ) if args.architecture == "marlin" else None
    expanding_stage = None
    if args.architecture == "expanding":
        if args.layer0_long_residual_scale is not None:
            raise ValueError(
                "--layer0-long-residual-scale is only valid for MARLIN checkpoints"
            )
        model, expanding_stage = expanding_model_from_checkpoint(
            args.checkpoint,
            device=device,
            use_ema=not args.no_ema,
        )
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
        eos_boost=args.eos_boost,
        eos_token_id=tokenizer.eos_token_id,
    )
    token_strings = [
        tokenizer.convert_ids_to_tokens(index) for index in range(len(tokenizer))
    ]
    chemistry_forbidden_ids = tuple(
        sorted(
            set(isotope_token_ids(token_strings) if args.forbid_isotope_tokens else ())
            | set(
                foreign_element_token_ids(token_strings)
                if args.restrict_organic_elements
                else ()
            )
        )
    )
    forbidden_token_ids = (
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.mask_token_id,
        tokenizer.pad_token_id,
    ) + chemistry_forbidden_ids
    if args.architecture == "expanding":
        sampler = ExpandingMarlinSampler(
            model,
            constraint,
            stage=expanding_stage,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decode_tokens=lambda ids: tokenizer.decode(
                ids, skip_special_tokens=True
            ),
            safe_to_smiles=lambda safe: safe_to_smiles(
                safe, fix=args.fix_safe_decode
            ),
            forbidden_token_ids=forbidden_token_ids,
            mass_shell_enabled=not args.disable_mass_shell,
            steps=args.expanding_steps,
        )
    else:
        sampler = MarlinSampler(
            model,
            constraint,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            mask_token_id=tokenizer.mask_token_id,
            decode_tokens=lambda ids: tokenizer.decode(
                ids, skip_special_tokens=True
            ),
            safe_to_smiles=lambda safe: safe_to_smiles(
                safe, fix=args.fix_safe_decode
            ),
            grammar_mask=None
            if args.disable_grammar_mask
            else SafeGrammarMask(
                token_strings,
                lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
                eos_token_id=tokenizer.eos_token_id,
                mask_token_id=tokenizer.mask_token_id,
                special_token_ids=tuple(special_ids) + (tokenizer.unk_token_id,),
                forbidden_token_ids=chemistry_forbidden_ids,
                ppm_tolerance=args.ppm_tolerance,
                valence_slack=args.valence_slack,
                mass_reachability_prune=args.mass_reachability_prune,
            ),
            forbidden_token_ids=tuple(
                token_id
                for token_id in forbidden_token_ids
                if token_id != tokenizer.eos_token_id
            ),
            mass_shell_enabled=not args.disable_mass_shell,
            generation_mode=args.generation_mode,
            sample_tokens=args.sample_tokens,
        )

    current_git_commit = git_commit()
    settings = {
        "evaluation_profile": args.evaluation_profile,
        "git_commit": current_git_commit,
        "checkpoint_sha256": sha256(args.checkpoint),
        "spec_manifest_sha256": sha256(args.spec_manifest)
        if args.spec_manifest
        else None,
        "lane": args.lane,
        "fingerprint_key": args.fingerprint_key,
        "fingerprint_threshold": args.threshold,
        "soft_fingerprint": args.soft_fingerprint,
        "candidates": args.candidates,
        "diversity_dropout": args.diversity_dropout,
        "temperature": args.temperature,
        "generation_mode": args.generation_mode,
        "architecture": args.architecture,
        "expanding_stage": expanding_stage,
        "expanding_steps": (
            args.expanding_steps if args.architecture == "expanding" else None
        ),
        "ppm_tolerance": args.ppm_tolerance,
        "valence_slack": args.valence_slack,
        "eos_boost": args.eos_boost,
        "block_width": model.config.block_width,
        "layer0_long_residual_scale": model.config.layer0_long_residual_scale,
        "ema": not args.no_ema,
        "weights": "ema" if not args.no_ema else "raw",
        "grammar_mask": (
            not args.disable_grammar_mask
            if args.architecture == "marlin"
            else False
        ),
        "mass_shell_constraint": not args.disable_mass_shell,
        "mass_reachability_prune": args.mass_reachability_prune,
        "per_spectrum_seconds": args.per_spectrum_seconds,
        "forbid_isotope_tokens": args.forbid_isotope_tokens,
        "restrict_organic_elements": args.restrict_organic_elements,
        "chemistry_forbidden_token_count": len(chemistry_forbidden_ids),
        "safe_decode_fix": args.fix_safe_decode,
        "isotope_token": args.isotope_token,
        "token_selection": "multinomial" if args.sample_tokens else "argmax",
        "seed": args.seed,
        "max_spectra": args.max_spectra,
        "spec_manifest": str(args.spec_manifest.resolve())
        if args.spec_manifest
        else None,
        "ordered_spec_names_sha256": hash_spec_names(
            metadata["spec_name"].astype(str).tolist()
        ),
    }
    signature = {
        "schema_version": 2,
        "git_commit": current_git_commit,
        "settings": settings,
        "inputs": {
            str(path.resolve()): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (
                args.checkpoint,
                args.tokenizer,
                args.metadata,
                args.fingerprints,
                *([args.spec_manifest] if args.spec_manifest else []),
                *([args.lane_provenance] if args.lane_provenance else []),
            )
        },
    }
    signature_path = args.output_dir / "run_signature.json"
    if signature_path.exists():
        existing_signature = json.loads(signature_path.read_text())
        if existing_signature != signature:
            raise ValueError(
                f"output directory contains an incompatible run: {signature_path}"
            )
    else:
        predictions_path = args.output_dir / "predictions.jsonl"
        if predictions_path.exists() and predictions_path.stat().st_size:
            raise ValueError(
                "refusing to resume predictions without a matching run_signature.json"
            )
        signature_path.write_text(
            json.dumps(signature, indent=2, sort_keys=True) + "\n"
        )

    predictions_path = args.output_dir / "predictions.jsonl"
    completed: set[str] = set()
    rows: list[dict] = []
    if predictions_path.exists():
        with predictions_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["spec_name"] in completed:
                    raise ValueError(
                        f"duplicate spectrum in predictions: {row['spec_name']}"
                    )
                add_formula_metrics(row)
                completed.add(row["spec_name"])
                rows.append(row)

    candidate_batch_size = args.candidate_batch_size
    if args.per_spectrum_seconds is not None and candidate_batch_size is None:
        # The deadline is only observable between batches, so a single batch of
        # every candidate would make the budget unenforceable.
        candidate_batch_size = min(8, args.candidates)

    with predictions_path.open("a") as output:
        for position, record in metadata.reset_index(drop=True).iterrows():
            spec_name = str(record["spec_name"])
            if spec_name in completed:
                continue
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            ranked, stats = sampler.generate_ranked_with_stats(
                torch.from_numpy(fingerprints[position]),
                float(record["neutral_mass"]),
                candidates=args.candidates,
                diversity_dropout=args.diversity_dropout,
                temperature=args.temperature,
                generator=torch.Generator(device=device).manual_seed(
                    args.seed + position
                ),
                candidate_batch_size=candidate_batch_size,
                isotope_ratios=sampling_isotope_ratios(args.isotope_token, record),
                time_budget_seconds=args.per_spectrum_seconds,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            target_molecule = Chem.MolFromSmiles(record["smiles"])
            if target_molecule is None:
                raise ValueError(f"invalid target SMILES for {spec_name}")
            target_fingerprint = morgan(target_molecule)
            target_formula = rdMolDescriptors.CalcMolFormula(target_molecule)
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
                        "formula": rdMolDescriptors.CalcMolFormula(molecule),
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
                "truncated": stats.truncated,
                "valid": stats.valid,
                "mass_valid": stats.mass_valid,
                "unique_mass_valid": stats.unique_mass_valid,
                "constraint_dead_ends": stats.constraint_dead_ends,
                "eos_terminated": stats.eos_terminated,
                "max_length_terminated": stats.max_length_terminated,
                "sample_terminal_safes": stats.sample_terminal_safes,
                "sample_dead_ends": stats.sample_dead_ends,
                "validity": stats.valid / stats.attempts,
                # validity divides by every attempt, so a branch the constraints
                # killed before it emitted anything is scored as an invalid
                # molecule. Report the completed branches separately, otherwise a
                # decoder defect and a constraint defect look identical.
                "completed_validity": stats.valid
                / max(stats.attempts - stats.constraint_dead_ends, 1),
                "mass_validity": stats.mass_valid / max(stats.valid, 1),
                "uniqueness": stats.unique_mass_valid / max(stats.mass_valid, 1),
                "candidate_returned": bool(candidates),
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
            add_formula_metrics(result, target_formula=target_formula)
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            rows.append(result)
            print(
                f"{position + 1}/{len(metadata)} {spec_name} "
                f"unique={stats.unique_mass_valid} runtime={elapsed:.3f}s",
                flush=True,
            )

    rows_with_candidate = [row for row in rows if row["candidate_returned"]]
    within_spectrum_diversities = []
    for row in rows:
        candidate_fingerprints = []
        for candidate in row.get("candidates", []):
            molecule = Chem.MolFromSmiles(candidate["smiles"])
            if molecule is not None:
                candidate_fingerprints.append(morgan(molecule))
        pairwise = []
        for left in range(len(candidate_fingerprints)):
            for right in range(left + 1, len(candidate_fingerprints)):
                pairwise.append(
                    1.0 - DataStructs.TanimotoSimilarity(
                        candidate_fingerprints[left], candidate_fingerprints[right]
                    )
                )
        if pairwise:
            within_spectrum_diversities.append(sum(pairwise) / len(pairwise))
    metrics = {
        "lane": args.lane,
        "rows": len(rows),
        "exact_top1": mean_metric(rows, "exact_top1"),
        "exact_top10": mean_metric(rows, "exact_top10"),
        "candidate_return_rate": len(rows_with_candidate) / max(len(rows), 1),
        "tanimoto_top1": mean_metric(rows_with_candidate, "tanimoto_top1"),
        "tanimoto_top10": mean_metric(rows_with_candidate, "tanimoto_top10"),
        **formula_metric_summary(rows),
        "mass_bins": mass_bin_metrics(rows),
        "validity": mean_metric(rows, "validity"),
        "completed_validity": mean_metric(rows, "completed_validity"),
        "mass_validity": mean_metric(rows, "mass_validity"),
        "uniqueness": mean_metric(rows, "uniqueness"),
        "internal_diversity": float(
            sum(within_spectrum_diversities) / len(within_spectrum_diversities)
        ) if within_spectrum_diversities else 0.0,
        "constraint_dead_ends_mean": mean_metric(rows, "constraint_dead_ends"),
        "eos_terminated_mean": mean_metric(rows, "eos_terminated"),
        "max_length_terminated_mean": mean_metric(
            rows, "max_length_terminated"
        ),
        "truncated_spectra": int(sum(1 for row in rows if row.get("truncated"))),
        "attempts_total": int(sum(row.get("attempts", 0) for row in rows)),
        "runtime_seconds_total": float(sum(row["runtime_seconds"] for row in rows)),
        "runtime_seconds_mean": mean_metric(rows, "runtime_seconds"),
        "metric_denominators": {
            "exact_top1": "all rows",
            "exact_top10": "all rows",
            "candidate_return_rate": "all rows",
            "tanimoto_top1": "rows with a returned candidate",
            "tanimoto_top10": "rows with a returned candidate",
            "formula_top1_all": "all rows",
            "formula_top1_returned": "rows with a returned candidate",
            "formula_top10_all": "all rows",
            "formula_top10_returned": "rows with a returned candidate",
            "validity": "all rows",
            "completed_validity": "all rows, branches that were not killed by a constraint",
            "mass_validity": "all rows",
            "uniqueness": "all rows",
            "internal_diversity": "mean within-spectrum pairwise Morgan distance",
            "constraint_dead_ends_mean": "all rows",
            "eos_terminated_mean": "all rows",
            "max_length_terminated_mean": "all rows",
            "runtime_seconds_mean": "all rows",
        },
        "settings": {
            **settings,
            "grammar_mask": "disabled diagnostic validity-gate lane"
            if args.disable_grammar_mask
            else "inferred conservative lexical SAFE grammar",
            "grammar_decode_order": "paper-specified confidence order"
            if args.disable_grammar_mask
            else "paper-specified confidence order with inferred hole handling",
            "mass_shell_constraint": "disabled diagnostic validity-gate lane"
            if args.disable_mass_shell
            else "enabled",
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
            "partial-block grammar handling for confidence-order commitment",
        ],
        "git_commit": signature["git_commit"],
        "inputs": signature["inputs"],
        "metrics": metrics,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    clearml_task = publish_clearml_evaluation(
        project_name=args.clearml_project,
        task_name=args.clearml_task_name,
        tags=args.clearml_tag,
        metrics=metrics,
        rows=rows,
        settings=settings,
        task_id=args.clearml_task_id,
        iteration=args.clearml_iteration,
        evaluation_label=(
            "MARLIN paper parity"
            if args.evaluation_profile == "paper-parity"
            else "MARLIN screening"
        ),
        evaluation_profile=args.evaluation_profile,
    )
    if clearml_task is not None:
        clearml_task["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
        (args.output_dir / "clearml_task.json").write_text(
            json.dumps(clearml_task, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
