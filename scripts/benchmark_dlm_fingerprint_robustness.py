#!/usr/bin/env python
"""
Paired DLM robustness benchmark for clean vs MIST-predicted fingerprints.

For each spectrum-molecule pair, this script evaluates the same DLM decoder under
multiple fingerprint sources:
- ground_truth: Morgan fingerprint computed from the target structure.
- mist_binary: MIST probabilities thresholded with the benchmark threshold.
- mist_probs: raw MIST probabilities, useful for soft-fingerprint ablations.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from benchmark_spec2mol import (  # noqa: E402
    load_config,
    load_dlm_sampler,
    load_mist_encoder,
    load_spec_data,
    merge_config_with_args,
)
from dlm.utils.benchmark_utils import (  # noqa: E402
    binarize_fingerprint,
    build_prediction_entry,
    compute_aggregate_statistics,
    compute_morgan_fingerprint,
    compute_tanimoto_similarity,
    evaluate_predictions,
    generate_with_formula_filter,
    get_inchikey_first_block,
    load_token_model,
    normalize_formula,
)
from mist.data.datasets import get_paired_loader  # noqa: E402

RDLogger.DisableLog('rdApp.*')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate DLM robustness to MIST-predicted fingerprints.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, default='configs/spec2mol_benchmark_msg.yaml')
    parser.add_argument('--mist-checkpoint', type=str, help='MIST encoder checkpoint')
    parser.add_argument('--dlm-checkpoint', type=str, help='DLM decoder checkpoint')
    parser.add_argument('--data-dir', type=str, help='Spec data directory')
    parser.add_argument('--fp-threshold', type=float, help='MIST FP binarization threshold')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'])
    parser.add_argument('--max-spectra', type=int, default=None)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--softmax-temp', type=float)
    parser.add_argument('--randomness', type=float)
    parser.add_argument('--formula-matches', type=int, help='Required formula matches per spectrum')
    parser.add_argument('--max-attempts', type=int, help='Max generation attempts per spectrum')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-shared-cross-attention', action='store_true')
    parser.add_argument('--token-model', type=str, default=None)
    parser.add_argument('--sigma-lambda', type=float, default=3.0)
    parser.add_argument(
        '--partial-save-every',
        type=int,
        default=100,
        help='Write partial CSV/JSON outputs every N processed spectra. Use 0 to disable.',
    )
    parser.add_argument(
        '--fingerprint-sources',
        nargs='+',
        default=['ground_truth', 'mist_binary'],
        choices=['ground_truth', 'mist_binary', 'mist_probs'],
    )
    return parser.parse_args()


def fingerprint_error_stats(target_fp: np.ndarray, mist_binary: np.ndarray, mist_probs: np.ndarray) -> Dict[str, Any]:
    false_positive = np.logical_and(mist_binary == 1, target_fp == 0)
    false_negative = np.logical_and(mist_binary == 0, target_fp == 1)
    return {
        'mist_tanimoto': compute_tanimoto_similarity(target_fp, mist_binary),
        'mist_l1_mean': float(np.mean(np.abs(mist_probs - target_fp))),
        'gt_active_bits': int(np.sum(target_fp)),
        'mist_active_bits': int(np.sum(mist_binary)),
        'mist_false_positive_bits': int(np.sum(false_positive)),
        'mist_false_negative_bits': int(np.sum(false_negative)),
    }


def compute_aggregate(
    results: List[Dict[str, Any]],
    fingerprint_sources: List[str],
    elapsed: float,
) -> Dict[str, Any]:
    aggregate = {}
    for source in fingerprint_sources:
        source_results = [r for r in results if r['fingerprint_source'] == source]
        aggregate[source] = compute_aggregate_statistics(source_results, elapsed)
        aggregate[source]['total_generation_time'] = float(sum(r.get('generation_time', 0.0) for r in source_results))

    if 'ground_truth' in aggregate and 'mist_binary' in aggregate:
        gt = aggregate['ground_truth']
        mist = aggregate['mist_binary']
        aggregate['mist_binary_vs_ground_truth_delta'] = {
            key: float(mist.get(key, 0.0) - gt.get(key, 0.0))
            for key in ('exact_match_top1', 'exact_match_top10', 'tanimoto_top1_mean', 'tanimoto_top10_mean')
        }

    return aggregate


def write_json(path: str, payload: Dict[str, Any]):
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def build_predictions(
    matched_smiles: List[str],
    matched_sims: List[float],
    last_valid: Optional[str],
    ranking_fp: np.ndarray,
    fp_bits: int,
    fp_radius: int,
) -> Tuple[List[Dict[str, Any]], str]:
    sim_records = defaultdict(list)
    for smi, sim in zip(matched_smiles, matched_sims):
        sim_records[smi].append(sim)

    predictions = []
    if sim_records:
        for smi, sims in sim_records.items():
            entry = build_prediction_entry(smi, np.mean(sims), len(sims), 'formula', fp_bits, fp_radius)
            if entry:
                predictions.append(entry)
        predictions.sort(key=lambda x: (x['frequency'], x['similarity']), reverse=True)
        return predictions, 'formula'

    if last_valid:
        fallback_fp = compute_morgan_fingerprint(last_valid, fp_bits, fp_radius)
        sim = compute_tanimoto_similarity(ranking_fp, fallback_fp) if fallback_fp is not None else 0.0
        entry = build_prediction_entry(last_valid, sim, 1, 'fallback', fp_bits, fp_radius)
        if entry:
            predictions.append(entry)
    return predictions, 'fallback'


def evaluate_one_source(
    sampler,
    fingerprint_source: str,
    fingerprint_array: np.ndarray,
    target_smiles: str,
    target_inchi_key: str,
    target_formula: str,
    target_fp: np.ndarray,
    spec_name: str,
    config: dict,
    token_model=None,
    token_features=None,
    is_ngboost: bool = False,
    sigma_lambda: float = 3.0,
) -> Dict[str, Any]:
    gen_cfg = config['generation']
    fp_cfg = config['fingerprint']
    filter_cfg = config['formula_filter']
    fp_bits = fp_cfg['bits']
    fp_radius = fp_cfg['radius']
    top_k = gen_cfg.get('top_k', 100)

    (
        matched_smiles,
        matched_sims,
        total_gen,
        total_valid,
        total_matched,
        _counter,
        last_valid,
        gen_time,
    ) = generate_with_formula_filter(
        sampler=sampler,
        fingerprint_array=fingerprint_array,
        target_formula=target_formula,
        target_smiles=target_smiles,
        n_required=filter_cfg['n_required'],
        max_attempts=filter_cfg['max_attempts'],
        batch_size=gen_cfg['batch_size'],
        softmax_temp=gen_cfg['softmax_temp'],
        randomness=gen_cfg['randomness'],
        fp_bits=fp_bits,
        fp_radius=fp_radius,
        token_model=token_model,
        token_features=token_features,
        is_ngboost=is_ngboost,
        sigma_lambda=sigma_lambda,
    )

    predictions, proposal_source = build_predictions(
        matched_smiles,
        matched_sims,
        last_valid,
        fingerprint_array,
        fp_bits,
        fp_radius,
    )
    preds_eval = predictions[:top_k] if top_k and top_k > 0 else predictions
    result = evaluate_predictions(preds_eval, target_smiles, target_inchi_key, target_fp, fp_bits, fp_radius)
    result.update({
        'fingerprint_source': fingerprint_source,
        'spec_name': spec_name,
        'proposal_smiles': predictions[0]['smiles'] if predictions else None,
        'proposal_source': proposal_source if predictions else None,
        'formula_matches_collected': len(matched_smiles),
        'total_generated': total_gen,
        'total_valid': total_valid,
        'total_formula_matched': total_matched,
        'generation_time': gen_time,
        'all_matched_smiles': matched_smiles,
    })
    return result


def run_paired_benchmark(
    mist_encoder,
    sampler,
    dataset,
    split_data,
    config: dict,
    device: torch.device,
    fingerprint_sources: List[str],
    max_spectra: Optional[int] = None,
    token_model=None,
    token_features=None,
    is_ngboost: bool = False,
    sigma_lambda: float = 3.0,
    output_dir: Optional[str] = None,
    partial_save_every: int = 100,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    fp_cfg = config['fingerprint']
    fp_bits = fp_cfg['bits']
    fp_radius = fp_cfg['radius']
    fp_threshold = fp_cfg['threshold']

    dataloader = get_paired_loader(dataset, shuffle=False, batch_size=1, num_workers=0)
    num_to_process = min(len(dataset), max_spectra) if max_spectra else len(dataset)
    results = []
    fp_drift_rows = []
    start_time = time.time()

    for idx, batch in enumerate(tqdm(dataloader, total=num_to_process, desc='Processing spectra')):
        if max_spectra and idx >= max_spectra:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        spec, mol = split_data[idx]
        target_smiles = mol.get_smiles()
        target_inchi_key = get_inchikey_first_block(mol.get_inchikey())
        target_formula = normalize_formula(rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(target_smiles)))
        target_fp = compute_morgan_fingerprint(target_smiles, fp_bits, fp_radius)
        if target_fp is None:
            print(f"Warning: Could not compute FP for {target_smiles}, skipping")
            continue

        with torch.no_grad():
            mist_probs, _ = mist_encoder(batch)
            mist_probs = mist_probs.cpu().numpy()[0]
        mist_binary = binarize_fingerprint(mist_probs, fp_threshold)

        fp_stats = fingerprint_error_stats(target_fp, mist_binary, mist_probs)
        fp_stats.update({
            'spec_name': spec.get_spec_name(),
            'target_smiles': target_smiles,
            'target_inchi_key': target_inchi_key,
        })
        fp_drift_rows.append(fp_stats)

        source_to_fp = {
            'ground_truth': target_fp,
            'mist_binary': mist_binary,
            'mist_probs': mist_probs.astype(np.float32),
        }
        for source in fingerprint_sources:
            result = evaluate_one_source(
                sampler=sampler,
                fingerprint_source=source,
                fingerprint_array=source_to_fp[source],
                target_smiles=target_smiles,
                target_inchi_key=target_inchi_key,
                target_formula=target_formula,
                target_fp=target_fp,
                spec_name=spec.get_spec_name(),
                config=config,
                token_model=token_model,
                token_features=token_features,
                is_ngboost=is_ngboost,
                sigma_lambda=sigma_lambda,
            )
            result.update(fp_stats)
            results.append(result)

        processed = idx + 1
        if output_dir and partial_save_every > 0 and processed % partial_save_every == 0:
            elapsed = time.time() - start_time
            partial_aggregate = compute_aggregate(results, fingerprint_sources, elapsed)
            save_outputs(
                output_dir,
                partial_aggregate,
                results,
                fp_drift_rows,
                config,
                run_state={
                    'completed': False,
                    'processed_spectra': processed,
                    'total_spectra': num_to_process,
                    'elapsed_time_seconds': elapsed,
                },
            )
            print(f"\nPartial results saved after {processed}/{num_to_process} spectra to: {output_dir}/")

    elapsed = time.time() - start_time
    aggregate = compute_aggregate(results, fingerprint_sources, elapsed)

    return aggregate, results, fp_drift_rows


def save_outputs(
    output_dir: str,
    aggregate: Dict[str, Any],
    results: List[Dict[str, Any]],
    fp_drift_rows: List[Dict[str, Any]],
    config: dict,
    run_state: Optional[Dict[str, Any]] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    write_json(os.path.join(output_dir, 'aggregate_statistics.json'), aggregate)

    save_rows = []
    for row in results:
        save_rows.append({k: v for k, v in row.items() if k not in ('top_predictions', 'all_matched_smiles')})
    details = pd.DataFrame(save_rows)
    details.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    pd.DataFrame(fp_drift_rows).to_csv(os.path.join(output_dir, 'fingerprint_drift.csv'), index=False)

    for source in sorted(details['fingerprint_source'].unique()):
        source_rows = []
        for row in results:
            if row['fingerprint_source'] != source:
                continue
            matched = row.get('all_matched_smiles', [])
            out_row = {
                'true_smiles': row.get('target_smiles', ''),
                'name': row.get('spec_name', ''),
            }
            for idx, smiles in enumerate(matched, start=1):
                out_row[f'pred_smiles_{idx}'] = smiles
            source_rows.append(out_row)
        pd.DataFrame(source_rows).to_csv(
            os.path.join(output_dir, f'predictions_{source}.csv'),
            index=False,
        )

    metric_cols = ['exact_match_top1', 'exact_match_top10', 'tanimoto_top1', 'tanimoto_top10', 'total_formula_matched']
    paired = details.pivot(index='spec_name', columns='fingerprint_source', values=metric_cols)
    paired.columns = [f'{metric}_{source}' for metric, source in paired.columns]
    if 'tanimoto_top1_ground_truth' in paired and 'tanimoto_top1_mist_binary' in paired:
        paired['delta_tanimoto_top1_mist_binary_minus_ground_truth'] = (
            paired['tanimoto_top1_mist_binary'] - paired['tanimoto_top1_ground_truth']
        )
    paired.reset_index().to_csv(os.path.join(output_dir, 'paired_comparison.csv'), index=False)

    write_json(os.path.join(output_dir, 'config.json'), config)
    if run_state is not None:
        write_json(os.path.join(output_dir, 'run_state.json'), run_state)


def print_summary(aggregate: Dict[str, Any]):
    print("\nDLM FINGERPRINT ROBUSTNESS SUMMARY")
    for source, stats in aggregate.items():
        if source.endswith('_delta'):
            continue
        print(
            f"{source}: exact@1={stats.get('exact_match_top1', 0.0):.4f}, "
            f"exact@10={stats.get('exact_match_top10', 0.0):.4f}, "
            f"tan@1={stats.get('tanimoto_top1_mean', 0.0):.4f}, "
            f"tan@10={stats.get('tanimoto_top10_mean', 0.0):.4f}"
        )
    if 'mist_binary_vs_ground_truth_delta' in aggregate:
        print(f"mist_binary - ground_truth: {aggregate['mist_binary_vs_ground_truth_delta']}")


def main():
    args = parse_args()
    config = merge_config_with_args(load_config(args.config), args)
    if args.output_dir:
        config['output']['results_dir'] = args.output_dir
    else:
        config['output']['results_dir'] = os.path.join(config['output']['results_dir'], 'dlm_fingerprint_robustness')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    token_model = None
    token_features = None
    is_ngboost = False
    if args.token_model:
        token_model, token_features, is_ngboost = load_token_model(args.token_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset, split_data = load_spec_data(
        config['data'],
        config['mist_encoder'],
        config['evaluation']['split'],
        shuffle=False,
    )
    mist_encoder = load_mist_encoder(config['mist_encoder'], device)
    sampler = load_dlm_sampler(config['dlm'], args.use_shared_cross_attention)

    aggregate, results, fp_drift_rows = run_paired_benchmark(
        mist_encoder=mist_encoder,
        sampler=sampler,
        dataset=dataset,
        split_data=split_data,
        config=config,
        device=device,
        fingerprint_sources=args.fingerprint_sources,
        max_spectra=args.max_spectra or config.get('evaluation', {}).get('max_spectra'),
        token_model=token_model,
        token_features=token_features,
        is_ngboost=is_ngboost,
        sigma_lambda=args.sigma_lambda,
        output_dir=config['output']['results_dir'],
        partial_save_every=args.partial_save_every,
    )

    save_outputs(
        config['output']['results_dir'],
        aggregate,
        results,
        fp_drift_rows,
        config,
        run_state={
            'completed': True,
            'processed_spectra': len(fp_drift_rows),
            'total_spectra': len(fp_drift_rows),
            'elapsed_time_seconds': max(
                aggregate.get(source, {}).get('elapsed_time_seconds', 0.0)
                for source in args.fingerprint_sources
            ),
        },
    )
    print_summary(aggregate)
    print(f"\nResults saved to: {config['output']['results_dir']}/")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nBenchmark interrupted.')
        sys.exit(1)
