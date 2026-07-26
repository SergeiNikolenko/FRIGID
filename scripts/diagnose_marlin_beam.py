#!/usr/bin/env python3
"""Run a bounded constrained-beam gate on oracle-conditioned MARLIN rows."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Draw

from audit_marlin_safe_oracle import encode_audit_sequence
from diagnose_marlin_production_prefix import (
    build_production_sampler,
    oracle_condition,
)
from evaluate_marlin_nplib1 import git_commit, load_decoder, sha256
from marlin.tokenizer import load_safe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "raw"), default="raw")
    parser.add_argument("--beam-width", type=int, action="append", required=True)
    parser.add_argument("--branch-factor", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-model-batch-size", type=int, default=64)
    parser.add_argument("--max-completed-paths", type=int)
    parser.add_argument("--expected-block-width", type=int, required=True)
    parser.add_argument("--clearml-project")
    parser.add_argument("--clearml-task-name")
    parser.add_argument("--clearml-tag", action="append", default=[])
    return parser.parse_args()


def connectivity(smiles: str | None) -> str | None:
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        return None
    return Chem.MolToInchiKey(molecule).split("-")[0]


def summarize_width(rows: list[dict]) -> dict[str, float | int]:
    count = len(rows)
    returned = sum(bool(row["candidate_returned"]) for row in rows)
    rows_with_valid_path = sum(
        int(row["search_stats"]["valid_paths"] > 0) for row in rows
    )
    rows_with_strict_valid_path = sum(
        int(row["search_stats"]["strict_valid_paths"] > 0) for row in rows
    )
    rows_with_mass_valid_path = sum(
        int(row["search_stats"]["mass_valid_paths"] > 0) for row in rows
    )
    return {
        "rows": count,
        "candidate_return_rate": returned / max(count, 1),
        "exact_top1": sum(bool(row["exact_top1"]) for row in rows) / max(count, 1),
        "exact_top10": sum(bool(row["exact_top10"]) for row in rows) / max(count, 1),
        "tanimoto_top1": sum(float(row["tanimoto_top1"]) for row in rows)
        / max(count, 1),
        "row_valid_recovery_rate": rows_with_valid_path / max(count, 1),
        "row_strict_valid_recovery_rate": rows_with_strict_valid_path / max(count, 1),
        "row_mass_valid_recovery_rate": rows_with_mass_valid_path / max(count, 1),
        "rows_with_valid_path": rows_with_valid_path,
        "rows_with_strict_valid_path": rows_with_strict_valid_path,
        "rows_with_mass_valid_path": rows_with_mass_valid_path,
        "unique_candidates": sum(len(row["candidates"]) for row in rows),
        "expanded_hypotheses": sum(
            int(row["search_stats"]["expanded_hypotheses"]) for row in rows
        ),
        "expanded_tokens": sum(
            int(row["search_stats"]["expanded_tokens"]) for row in rows
        ),
        "completed_paths": sum(
            int(row["search_stats"]["completed_paths"]) for row in rows
        ),
        "constraint_dead_ends": sum(
            int(row["search_stats"]["constraint_dead_ends"]) for row in rows
        ),
        "eos_terminated": sum(
            int(row["search_stats"]["eos_terminated"]) for row in rows
        ),
        "block_terminated": sum(
            int(row["search_stats"]["block_terminated"]) for row in rows
        ),
        "max_length_terminated": sum(
            int(row["search_stats"]["max_length_terminated"]) for row in rows
        ),
        "pruned_hypotheses": sum(
            int(row["search_stats"]["pruned_hypotheses"]) for row in rows
        ),
        "backtrack_recoveries": sum(
            int(row["search_stats"]["backtrack_recoveries"]) for row in rows
        ),
        "max_committed_tokens": max(
            (int(row["search_stats"]["max_committed_tokens"]) for row in rows),
            default=0,
        ),
        "runtime_seconds": sum(float(row["runtime_seconds"]) for row in rows),
    }


def start_clearml(
    *,
    project_name: str | None,
    task_name: str | None,
    tags: list[str],
    settings: dict,
) -> tuple[Any | None, dict[str, str] | None]:
    if project_name is None and task_name is None:
        return None, None
    if not project_name or not task_name:
        raise ValueError(
            "--clearml-project and --clearml-task-name must be provided together"
        )

    from clearml import Task

    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type=Task.TaskTypes.testing,
        tags=tags,
        reuse_last_task_id=False,
        output_uri=False,
        auto_connect_streams=False,
        auto_connect_frameworks=False,
        auto_resource_monitoring=False,
    )
    task.connect(dict(settings), name="beam_search_settings")
    task.get_logger().report_text(
        "MARLIN constrained beam diagnostic started",
        print_console=False,
    )
    return task, {
        "task_id": task.id,
        "task_name": task.name,
        "project_name": project_name,
        "web_url": task.get_output_log_web_page(),
    }


def publish_clearml(task: Any | None, widths: list[dict]) -> None:
    if task is None:
        return
    logger = task.get_logger()
    scalar_keys = (
        "candidate_return_rate",
        "exact_top1",
        "exact_top10",
        "tanimoto_top1",
        "row_valid_recovery_rate",
        "row_strict_valid_recovery_rate",
        "row_mass_valid_recovery_rate",
        "rows_with_valid_path",
        "rows_with_strict_valid_path",
        "rows_with_mass_valid_path",
        "unique_candidates",
        "completed_paths",
        "constraint_dead_ends",
        "eos_terminated",
        "block_terminated",
        "max_length_terminated",
        "pruned_hypotheses",
        "backtrack_recoveries",
        "max_committed_tokens",
        "runtime_seconds",
    )
    table_rows = []
    for width_result in widths:
        width = int(width_result["beam_width"])
        metrics = width_result["metrics"]
        for key in scalar_keys:
            value = float(metrics[key])
            if math.isfinite(value):
                logger.report_scalar(
                    title="MARLIN constrained beam metrics",
                    series=key,
                    value=value,
                    iteration=width,
                )
        molecules = []
        legends = []
        for row in width_result["rows"]:
            table_rows.append(
                {
                    "beam_width": width,
                    "metadata_row": row["metadata_row"],
                    "target_smiles": row["target_smiles"],
                    "top1_smiles": row["top1_smiles"],
                    "candidate_returned": row["candidate_returned"],
                    "exact_top1": row["exact_top1"],
                    "tanimoto_top1": row["tanimoto_top1"],
                    "mass_valid_paths": row["search_stats"]["mass_valid_paths"],
                    "constraint_dead_ends": row["search_stats"]["constraint_dead_ends"],
                    "expanded_hypotheses": row["search_stats"]["expanded_hypotheses"],
                    "backtrack_recoveries": row["search_stats"]["backtrack_recoveries"],
                    "best_completed_log_probability": row["search_stats"][
                        "best_completed_log_probability"
                    ],
                }
            )
            target = Chem.MolFromSmiles(row["target_smiles"])
            if target is not None:
                molecules.append(target)
                legends.append(
                    f"row {row['metadata_row']} target\n{row['target_smiles']}"
                )
            generated = Chem.MolFromSmiles(row["top1_smiles"] or "")
            if generated is not None:
                molecules.append(generated)
                legends.append(
                    f"row {row['metadata_row']} beam={width} "
                    f"T={row['tanimoto_top1']:.3f}\n{row['top1_smiles']}"
                )
        if molecules:
            image = Draw.MolsToGridImage(
                molecules,
                legends=legends,
                molsPerRow=2,
                subImgSize=(420, 300),
                useSVG=False,
            )
            logger.report_image(
                title="MARLIN constrained beam molecules",
                series=f"beam_width_{width}",
                iteration=width,
                image=image,
                max_image_history=-1,
            )
    logger.report_table(
        title="MARLIN constrained beam evaluation",
        series="target vs generated",
        iteration=max(int(result["beam_width"]) for result in widths),
        table_plot=pd.DataFrame(table_rows),
    )
    logger.report_text(
        "MARLIN constrained beam diagnostic completed",
        print_console=False,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_search(
    *,
    sampler,
    conditions: list[dict],
    beam_widths: list[int],
    branch_factor: int,
    temperature: float,
    max_model_batch_size: int,
    max_completed_paths: int | None,
    device: torch.device,
    task: Any | None,
    partial_path: Path,
    result_base: dict,
) -> list[dict]:
    width_results = []
    for beam_width in beam_widths:
        rows = []
        for condition in conditions:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            ranked, stats = sampler.generate_beam_ranked_with_stats(
                condition["fingerprint"],
                condition["target_mass"],
                beam_width=beam_width,
                branch_factor=branch_factor,
                temperature=temperature,
                max_model_batch_size=max_model_batch_size,
                max_completed_paths=max_completed_paths,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            candidates = [asdict(candidate) for candidate in ranked]
            top1 = candidates[0] if candidates else None
            exact_flags = [
                connectivity(candidate["smiles"]) == condition["target_connectivity"]
                for candidate in candidates
            ]
            rows.append(
                {
                    "metadata_row": condition["metadata_row"],
                    "spec_name": condition["spec_name"],
                    "target_smiles": condition["target_smiles"],
                    "target_safe": condition["target_safe"],
                    "target_mass": condition["target_mass"],
                    "candidate_returned": bool(candidates),
                    "top1_smiles": top1["smiles"] if top1 else None,
                    "exact_top1": bool(exact_flags and exact_flags[0]),
                    "exact_top10": any(exact_flags[:10]),
                    "tanimoto_top1": float(top1["tanimoto"]) if top1 else 0.0,
                    "runtime_seconds": time.perf_counter() - started,
                    "search_stats": asdict(stats),
                    "candidates": candidates,
                }
            )
            if task is not None:
                task.get_logger().report_scalar(
                    title="MARLIN constrained beam progress",
                    series=f"beam_width_{beam_width}_rows_completed",
                    value=len(rows),
                    iteration=len(rows),
                )
        width_results.append(
            {
                "beam_width": beam_width,
                "metrics": summarize_width(rows),
                "rows": rows,
            }
        )
        atomic_json(
            partial_path,
            {**result_base, "widths": width_results},
        )
    return width_results


def main() -> None:
    args = parse_args()
    if args.row < 0 or args.rows <= 0:
        raise ValueError("row must be non-negative and rows must be positive")
    if args.branch_factor <= 0:
        raise ValueError("branch factor must be positive")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if args.max_model_batch_size <= 0:
        raise ValueError("max model batch size must be positive")
    if args.max_completed_paths is not None and args.max_completed_paths <= 0:
        raise ValueError("max completed paths must be positive")
    beam_widths = sorted(set(args.beam_width))
    if any(width <= 0 for width in beam_widths):
        raise ValueError("beam widths must be positive")
    if bool(args.clearml_project) != bool(args.clearml_task_name):
        raise ValueError(
            "--clearml-project and --clearml-task-name must be provided together"
        )

    device = torch.device(args.device)
    model = load_decoder(
        args.checkpoint,
        device,
        use_ema=args.weights == "ema",
    )
    if model.config.block_width != args.expected_block_width:
        raise ValueError(
            f"checkpoint block width is {model.config.block_width}; "
            f"expected {args.expected_block_width}"
        )
    tokenizer = load_safe_tokenizer(args.tokenizer)
    tokenizer_contract = {
        "vocab_size": (len(tokenizer), model.config.vocab_size),
        "bos_token_id": (tokenizer.bos_token_id, model.config.bos_token_id),
        "eos_token_id": (tokenizer.eos_token_id, model.config.eos_token_id),
        "mask_token_id": (tokenizer.mask_token_id, model.config.mask_token_id),
        "pad_token_id": (tokenizer.pad_token_id, model.config.pad_token_id),
    }
    mismatches = {
        name: {"tokenizer": actual, "checkpoint": expected}
        for name, (actual, expected) in tokenizer_contract.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"tokenizer/checkpoint contract mismatch: {mismatches}")
    sampler = build_production_sampler(model, tokenizer)
    metadata = pd.read_csv(args.metadata).iloc[args.row : args.row + args.rows]
    if len(metadata) != args.rows:
        raise ValueError(
            f"requested {args.rows} rows at offset {args.row}; found {len(metadata)}"
        )
    conditions = []
    for metadata_index, record in metadata.iterrows():
        safe, _ = encode_audit_sequence(str(record["smiles"]), tokenizer)
        fingerprint, target_mass = oracle_condition(safe, model.config.fingerprint_bits)
        conditions.append(
            {
                "metadata_row": int(metadata_index),
                "spec_name": (
                    str(record["spec_name"]) if "spec_name" in record else None
                ),
                "target_smiles": str(record["smiles"]),
                "target_safe": safe,
                "target_connectivity": connectivity(str(record["smiles"])),
                "target_mass": target_mass,
                "fingerprint": fingerprint,
            }
        )

    settings = {
        "weights": args.weights,
        "beam_widths": beam_widths,
        "branch_factor": args.branch_factor,
        "temperature": args.temperature,
        "max_model_batch_size": args.max_model_batch_size,
        "max_completed_paths": args.max_completed_paths,
        "row_start": args.row,
        "row_count": args.rows,
        "block_width": model.config.block_width,
        "conditioning": "oracle Morgan radius=2 fingerprint and exact molecular mass",
        "constraints": "SafeGrammarMask plus MassShellConstraint",
        "search_score": "cumulative constrained token log probability",
    }
    result_base = {
        "kind": "MARLIN constrained beam diagnostic",
        "git_commit": git_commit(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": sha256(args.tokenizer),
        "metadata": str(args.metadata),
        "metadata_sha256": sha256(args.metadata),
        "settings": settings,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.output_dir / "run_status.json"
    clearml_path = args.output_dir / "clearml_task.json"
    partial_path = args.output_dir / "beam_results.partial.json"
    output_path = args.output_dir / "beam_results.json"
    started_at = utc_now()
    status = {
        "status": "initializing",
        "started_at_utc": started_at,
        "updated_at_utc": started_at,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": result_base["git_commit"],
    }
    atomic_json(status_path, status)

    task = None
    clearml_info = None
    try:
        task, clearml_info = start_clearml(
            project_name=args.clearml_project,
            task_name=args.clearml_task_name,
            tags=args.clearml_tag,
            settings=settings,
        )
        status.update(status="running", updated_at_utc=utc_now())
        atomic_json(status_path, status)
        if clearml_info is not None:
            clearml_info.update(
                status="running",
                started_at_utc=started_at,
                updated_at_utc=utc_now(),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                git_commit=result_base["git_commit"],
            )
            atomic_json(clearml_path, clearml_info)

        width_results = run_search(
            sampler=sampler,
            conditions=conditions,
            beam_widths=beam_widths,
            branch_factor=args.branch_factor,
            temperature=args.temperature,
            max_model_batch_size=args.max_model_batch_size,
            max_completed_paths=args.max_completed_paths,
            device=device,
            task=task,
            partial_path=partial_path,
            result_base=result_base,
        )
        result = {**result_base, "widths": width_results}
        atomic_json(output_path, result)
        publish_clearml(task, width_results)
    except BaseException as error:
        failure = {
            "exception_type": type(error).__name__,
            "message": str(error)[:2000],
        }
        status.update(
            status="failed",
            updated_at_utc=utc_now(),
            failure=failure,
        )
        atomic_json(status_path, status)
        if clearml_info is not None:
            clearml_info.update(
                status="failed",
                updated_at_utc=utc_now(),
                failure=failure,
            )
            atomic_json(clearml_path, clearml_info)
        if task is not None:
            try:
                task.get_logger().report_text(
                    f"MARLIN constrained beam diagnostic failed: {failure}",
                    print_console=False,
                )
                task.flush(wait_for_uploads=True)
                task.mark_failed(
                    status_reason=failure["exception_type"],
                    status_message=failure["message"],
                )
            except Exception:
                pass
        raise
    else:
        if task is not None:
            task.close()
        status.update(status="completed", updated_at_utc=utc_now())
        atomic_json(status_path, status)
        if clearml_info is not None:
            clearml_info.update(status="completed", updated_at_utc=utc_now())
            atomic_json(clearml_path, clearml_info)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "widths": [
                    {
                        "beam_width": width_result["beam_width"],
                        "metrics": width_result["metrics"],
                    }
                    for width_result in width_results
                ],
                "clearml_task": clearml_info,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
