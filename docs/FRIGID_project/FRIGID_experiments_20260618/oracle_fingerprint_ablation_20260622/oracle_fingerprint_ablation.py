#!/usr/bin/env python3
"""Oracle fingerprint ablation for FRIGID Spec2Mol.

This script replaces the MIST-predicted fingerprint with the ground-truth
Morgan fingerprint computed from the target molecule. It is an upper-bound
test for the encoder bottleneck because the same fingerprint is used for
generation conditioning and formula-candidate ranking.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dlm-checkpoint", required=True)
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--formula-matches", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--sigma-lambda", type=float, default=3.0)
    parser.add_argument("--use-shared-cross-attention", action="store_true")
    return parser.parse_args()


def configure_imports(project_root: Path) -> None:
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))


def build_predictions(matched_smiles, matched_sims, fp_bits, fp_radius):
    from dlm.utils.benchmark_utils import build_prediction_entry

    sim_records = defaultdict(list)
    for smi, sim in zip(matched_smiles, matched_sims):
        sim_records[smi].append(sim)

    predictions = []
    for smi, sims in sim_records.items():
        entry = build_prediction_entry(smi, np.mean(sims), len(sims), "oracle_formula", fp_bits, fp_radius)
        if entry:
            predictions.append(entry)
    predictions.sort(key=lambda x: (x["frequency"], x["similarity"]), reverse=True)
    return predictions


def save_predictions_csv(results, output_dir: Path) -> None:
    max_preds = max((len(r.get("all_matched_smiles", [])) for r in results), default=0)
    rows = []
    for r in results:
        row = {"true_smiles": r.get("target_smiles", ""), "name": r.get("spec_name", "")}
        for idx in range(max_preds):
            matched = r.get("all_matched_smiles", [])
            row[f"pred_smiles_{idx + 1}"] = matched[idx] if idx < len(matched) else ""
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "oracle_predictions.csv", index=False)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    configure_imports(project_root)

    from scripts.benchmark_spec2mol import load_config, load_spec_data, load_dlm_sampler
    from dlm.utils.benchmark_utils import (
        compute_aggregate_statistics,
        compute_morgan_fingerprint,
        compute_tanimoto_similarity,
        evaluate_predictions,
        generate_with_formula_filter,
        get_inchikey_first_block,
        load_token_model,
        normalize_formula,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    config["data"]["datadir"] = args.data_dir
    config["data"]["labels_file"] = os.path.join(args.data_dir, "labels.tsv")
    config["data"]["split_file"] = os.path.join(args.data_dir, "split.tsv")
    config["data"]["spec_folder"] = os.path.join(args.data_dir, "spec_files")
    config["data"]["subform_folder"] = os.path.join(args.data_dir, "subformulae/default_subformulae")
    config["dlm"]["checkpoint"] = args.dlm_checkpoint
    config["generation"]["batch_size"] = args.batch_size
    config["formula_filter"]["n_required"] = args.formula_matches
    config["formula_filter"]["max_attempts"] = args.max_attempts
    config["evaluation"]["split"] = "test"

    token_model = None
    token_features = None
    is_ngboost = False
    if args.token_model:
        token_model, token_features, is_ngboost = load_token_model(args.token_model)

    sample = pd.read_csv(args.sample_csv)
    requested_names = list(sample["spec_name"])
    sample_by_name = sample.set_index("spec_name")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Requested spectra: {len(requested_names)}")

    dataset, split_data = load_spec_data(
        config["data"],
        config["mist_encoder"],
        split="test",
        shuffle=False,
    )
    del dataset

    name_to_idx = {spec.get_spec_name(): idx for idx, (spec, _mol) in enumerate(split_data)}
    missing = [name for name in requested_names if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Sample names not found in split_data: {missing[:10]}")

    sampler = load_dlm_sampler(config["dlm"], args.use_shared_cross_attention)

    fp_cfg = config["fingerprint"]
    gen_cfg = config["generation"]
    filter_cfg = config["formula_filter"]
    fp_bits = fp_cfg["bits"]
    fp_radius = fp_cfg["radius"]
    top_k = gen_cfg.get("top_k", 100)

    results = []
    start = time.time()
    for ordinal, spec_name in enumerate(requested_names, start=1):
        idx = name_to_idx[spec_name]
        spec, mol = split_data[idx]
        target_smiles = mol.get_smiles()
        target_inchi_key = get_inchikey_first_block(mol.get_inchikey())
        target_mol = Chem.MolFromSmiles(target_smiles)
        target_formula = normalize_formula(rdMolDescriptors.CalcMolFormula(target_mol))
        target_fp = compute_morgan_fingerprint(target_smiles, fp_bits, fp_radius)
        if target_fp is None:
            print(f"Skipping {spec_name}: could not compute target fingerprint")
            continue

        case_start = time.time()
        matched_smiles, matched_sims, total_gen, total_valid, total_matched, _counter, last_valid, gen_time = (
            generate_with_formula_filter(
                sampler=sampler,
                fingerprint_array=target_fp,
                target_formula=target_formula,
                target_smiles=target_smiles,
                n_required=filter_cfg["n_required"],
                max_attempts=filter_cfg["max_attempts"],
                batch_size=gen_cfg["batch_size"],
                softmax_temp=gen_cfg["softmax_temp"],
                randomness=gen_cfg["randomness"],
                fp_bits=fp_bits,
                fp_radius=fp_radius,
                token_model=token_model,
                token_features=token_features,
                is_ngboost=is_ngboost,
                sigma_lambda=args.sigma_lambda,
            )
        )

        predictions = build_predictions(matched_smiles, matched_sims, fp_bits, fp_radius)
        source = "oracle_formula" if predictions else "fallback"
        if not predictions and last_valid:
            from dlm.utils.benchmark_utils import build_prediction_entry

            fallback_fp = compute_morgan_fingerprint(last_valid, fp_bits, fp_radius)
            sim = compute_tanimoto_similarity(target_fp, fallback_fp) if fallback_fp is not None else 0.0
            entry = build_prediction_entry(last_valid, sim, 1, "fallback", fp_bits, fp_radius)
            if entry:
                predictions.append(entry)

        preds_eval = predictions[:top_k] if top_k and top_k > 0 else predictions
        result = evaluate_predictions(preds_eval, target_smiles, target_inchi_key, target_fp, fp_bits, fp_radius)
        result["spec_name"] = spec.get_spec_name()
        result["fingerprint_source"] = "oracle_ground_truth"
        result["oracle_conditioning_tanimoto"] = 1.0
        result["baseline_mist_tanimoto"] = float(sample_by_name.loc[spec_name, "mist_tanimoto"])
        result["baseline_exact_match_top1"] = float(sample_by_name.loc[spec_name, "exact_match_top1"])
        result["baseline_exact_match_top10"] = float(sample_by_name.loc[spec_name, "exact_match_top10"])
        result["baseline_tanimoto_top1"] = float(sample_by_name.loc[spec_name, "tanimoto_top1"])
        result["baseline_tanimoto_top10"] = float(sample_by_name.loc[spec_name, "tanimoto_top10"])
        result["baseline_outcome_bucket"] = sample_by_name.loc[spec_name, "outcome_bucket"]
        result["baseline_shard"] = sample_by_name.loc[spec_name, "shard"]
        result["proposal_smiles"] = predictions[0]["smiles"] if predictions else None
        result["proposal_source"] = source
        result["formula_matches_collected"] = len(matched_smiles)
        result["total_generated"] = total_gen
        result["total_valid"] = total_valid
        result["total_formula_matched"] = total_matched
        result["generation_time"] = gen_time
        result["wall_time"] = time.time() - case_start
        result["all_matched_smiles"] = matched_smiles
        results.append(result)
        print(
            f"[{ordinal:03d}/{len(requested_names):03d}] {spec_name} "
            f"base_top10={result['baseline_exact_match_top10']:.0f} "
            f"oracle_top10={result['exact_match_top10']:.0f} "
            f"oracle_tan10={result['tanimoto_top10']:.3f} "
            f"gen={total_gen} matched={total_matched} time={gen_time:.1f}s",
            flush=True,
        )

    elapsed = time.time() - start
    aggregate = compute_aggregate_statistics(results, elapsed)
    aggregate["total_generation_time"] = float(sum(r.get("generation_time", 0.0) for r in results))
    aggregate["generation_time_percentage"] = (
        aggregate["total_generation_time"] / elapsed * 100 if elapsed > 0 else 0.0
    )
    aggregate["total_elapsed_time_seconds"] = elapsed
    aggregate["proposal_source_counts"] = Counter(
        [r["proposal_source"] for r in results if r.get("proposal_source")]
    ).most_common()
    aggregate["fingerprint_source"] = "oracle_ground_truth"
    aggregate["sample_csv"] = str(args.sample_csv)
    aggregate["baseline_exact_match_top1"] = float(sample["exact_match_top1"].mean())
    aggregate["baseline_exact_match_top10"] = float(sample["exact_match_top10"].mean())
    aggregate["baseline_tanimoto_top1_mean"] = float(sample["tanimoto_top1"].mean())
    aggregate["baseline_tanimoto_top10_mean"] = float(sample["tanimoto_top10"].mean())
    aggregate["baseline_mist_tanimoto_mean"] = float(sample["mist_tanimoto"].mean())

    save_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ("top_predictions", "all_matched_smiles")}
        row["oracle_top_prediction_smiles"] = [p["smiles"] for p in r.get("top_predictions", [])]
        save_rows.append(row)
    detailed = pd.DataFrame(save_rows)
    detailed.to_csv(output_dir / "oracle_detailed_results.csv", index=False)
    save_predictions_csv(results, output_dir)

    comparison = detailed[
        [
            "spec_name",
            "baseline_shard",
            "baseline_outcome_bucket",
            "baseline_mist_tanimoto",
            "baseline_exact_match_top1",
            "baseline_exact_match_top10",
            "baseline_tanimoto_top1",
            "baseline_tanimoto_top10",
            "exact_match_top1",
            "exact_match_top10",
            "tanimoto_top1",
            "tanimoto_top10",
            "formula_matches_collected",
            "total_generated",
            "total_formula_matched",
            "generation_time",
            "target_smiles",
            "proposal_smiles",
        ]
    ].copy()
    comparison = comparison.rename(
        columns={
            "exact_match_top1": "oracle_exact_match_top1",
            "exact_match_top10": "oracle_exact_match_top10",
            "tanimoto_top1": "oracle_tanimoto_top1",
            "tanimoto_top10": "oracle_tanimoto_top10",
        }
    )
    comparison["delta_exact_top1"] = comparison["oracle_exact_match_top1"] - comparison["baseline_exact_match_top1"]
    comparison["delta_exact_top10"] = comparison["oracle_exact_match_top10"] - comparison["baseline_exact_match_top10"]
    comparison["delta_tanimoto_top10"] = comparison["oracle_tanimoto_top10"] - comparison["baseline_tanimoto_top10"]
    comparison.to_csv(output_dir / "comparison_vs_baseline.csv", index=False)

    with open(output_dir / "oracle_aggregate_statistics.json", "w") as f:
        json.dump(aggregate, f, indent=2)
    with open(output_dir / "oracle_run_config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
