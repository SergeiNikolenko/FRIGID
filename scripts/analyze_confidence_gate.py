#!/usr/bin/env python
"""Retrospective single-feature confidence gate analysis for FRIGID runs."""

import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    'mist_prob_entropy_norm',
    'mist_prob_entropy_mean',
    'mist_prob_high_confidence_ratio',
    'mist_prob_top16_mass',
    'mist_prob_top32_mass',
    'mist_prob_top64_mass',
    'mist_prob_top128_mass',
    'mist_prob_top256_mass',
    'mist_prob_bits_ge_0p10',
    'mist_prob_bits_ge_0p30',
    'mist_prob_bits_ge_0p50',
    'mist_prob_mean',
    'mist_prob_std',
    'mist_prob_max',
    'mist_prob_p95',
    'mist_prob_p99',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Find retrospective no-oracle feature gates between two completed FRIGID runs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--diagnostics-csv', type=str, required=True)
    parser.add_argument('--baseline-details', type=str, required=True)
    parser.add_argument('--challenger-details', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--baseline-name', type=str, default='baseline')
    parser.add_argument('--challenger-name', type=str, default='challenger')
    parser.add_argument('--features', nargs='*', default=None)
    parser.add_argument('--quantiles', nargs='*', type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    return parser.parse_args()


def load_metric_table(path: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [
        'spec_name',
        'tanimoto_top1',
        'tanimoto_top10',
        'exact_match_top1',
        'exact_match_top10',
        'total_formula_matched',
    ]
    missing = [col for col in keep if col not in df.columns]
    if missing:
        raise ValueError(f'{path} is missing columns: {missing}')
    return df[keep].rename(columns={col: f'{prefix}_{col}' for col in keep if col != 'spec_name'})


def policy_metrics(
    merged: pd.DataFrame,
    use_challenger: np.ndarray,
    baseline_name: str,
    challenger_name: str,
) -> Dict[str, float]:
    metrics = {}
    for metric in ('tanimoto_top1', 'tanimoto_top10', 'exact_match_top1', 'exact_match_top10'):
        base = merged[f'{baseline_name}_{metric}'].to_numpy(dtype=float)
        challenger = merged[f'{challenger_name}_{metric}'].to_numpy(dtype=float)
        selected = np.where(use_challenger, challenger, base)
        metrics[f'policy_{metric}'] = float(np.mean(selected))
        metrics[f'policy_minus_baseline_{metric}'] = float(np.mean(selected - base))
        metrics[f'policy_minus_challenger_{metric}'] = float(np.mean(selected - challenger))
    return metrics


def evaluate_rules(
    merged: pd.DataFrame,
    features: Iterable[str],
    quantiles: Iterable[float],
    baseline_name: str,
    challenger_name: str,
) -> List[Dict[str, float]]:
    rows = []
    for feature in features:
        if feature not in merged.columns:
            continue
        values = merged[feature].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < 2:
            continue
        thresholds = sorted({float(np.quantile(values[finite], q)) for q in quantiles})
        for threshold in thresholds:
            for direction, mask in (
                ('le', values <= threshold),
                ('ge', values >= threshold),
            ):
                if mask.sum() == 0 or mask.sum() == len(mask):
                    continue
                row = {
                    'feature': feature,
                    'direction': direction,
                    'threshold': threshold,
                    'challenger_fraction': float(np.mean(mask)),
                    'challenger_count': int(np.sum(mask)),
                    'n': int(len(mask)),
                }
                row.update(policy_metrics(merged, mask, baseline_name, challenger_name))
                rows.append(row)
    rows.sort(
        key=lambda row: (
            row['policy_minus_baseline_tanimoto_top1'],
            row['policy_tanimoto_top1'],
            row['policy_tanimoto_top10'],
        ),
        reverse=True,
    )
    return rows


def summarize_pair(merged: pd.DataFrame, baseline_name: str, challenger_name: str) -> Dict[str, float]:
    summary = {'n': int(len(merged))}
    for metric in ('tanimoto_top1', 'tanimoto_top10', 'exact_match_top1', 'exact_match_top10'):
        base = merged[f'{baseline_name}_{metric}'].to_numpy(dtype=float)
        challenger = merged[f'{challenger_name}_{metric}'].to_numpy(dtype=float)
        delta = challenger - base
        summary[f'{baseline_name}_{metric}_mean'] = float(np.mean(base))
        summary[f'{challenger_name}_{metric}_mean'] = float(np.mean(challenger))
        summary[f'{challenger_name}_minus_{baseline_name}_{metric}_mean'] = float(np.mean(delta))
        summary[f'{challenger_name}_wins_{metric}'] = int(np.sum(delta > 0))
        summary[f'{challenger_name}_losses_{metric}'] = int(np.sum(delta < 0))
        summary[f'{challenger_name}_ties_{metric}'] = int(np.sum(delta == 0))
    return summary


def write_json(path: str, payload: Dict):
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    diagnostics = pd.read_csv(args.diagnostics_csv)
    baseline = load_metric_table(args.baseline_details, args.baseline_name)
    challenger = load_metric_table(args.challenger_details, args.challenger_name)
    merged = diagnostics.merge(baseline, on='spec_name').merge(challenger, on='spec_name')
    if merged.empty:
        raise ValueError('No overlapping spec_name rows across diagnostics and detailed result files')

    features = args.features or DEFAULT_FEATURES
    rules = evaluate_rules(merged, features, args.quantiles, args.baseline_name, args.challenger_name)
    rules_df = pd.DataFrame(rules)
    rules_df.to_csv(os.path.join(args.output_dir, 'confidence_gate_rules.tsv'), sep='\t', index=False)
    merged.to_csv(os.path.join(args.output_dir, 'confidence_gate_joined.csv'), index=False)

    summary = summarize_pair(merged, args.baseline_name, args.challenger_name)
    summary['best_rule'] = rules[0] if rules else None
    summary['features_evaluated'] = [feature for feature in features if feature in merged.columns]
    write_json(os.path.join(args.output_dir, 'summary.json'), summary)

    print(f'Joined rows: {len(merged)}')
    print(f'Rules evaluated: {len(rules)}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
