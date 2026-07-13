#!/usr/bin/env python
"""
Export MIST-predicted fingerprints for DLM fine-tuning.

The output contains:
- metadata.csv: input SAFE/SMILES identifiers and split labels.
- fingerprints.npz: MIST probability, binary MIST, and ground-truth fingerprints.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
scripts_path = os.path.join(project_root, 'scripts')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from dlm.utils.benchmark_selection import load_spec_manifest, resolve_selected_indices  # noqa: E402

RDLogger.DisableLog('rdApp.*')


class _FeaturizedSubset(torch.utils.data.Subset):
    def get_featurizer(self):
        return self.dataset.get_featurizer()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export MIST fingerprints for DLM adaptation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, default='configs/spec2mol_benchmark_msg.yaml')
    parser.add_argument('--mist-checkpoint', type=str, help='MIST encoder checkpoint')
    parser.add_argument('--dlm-checkpoint', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--data-dir', type=str, help='Spec data directory')
    parser.add_argument('--fp-threshold', type=float, help='MIST FP binarization threshold')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--max-spectra', type=int, default=None)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--softmax-temp', type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--randomness', type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--formula-matches', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--max-attempts', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        '--spec-manifest',
        type=str,
        default=None,
        help="Ordered CSV/TSV subset with a unique 'spec_name' column.",
    )
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def resolve_export_indices(
    split_data,
    spec_manifest: str | None,
    start_index: int,
    max_spectra: int | None,
) -> list[int]:
    spec_names = load_spec_manifest(spec_manifest) if spec_manifest else None
    return resolve_selected_indices(
        split_data,
        spec_names,
        start_index,
        max_spectra,
    )


def build_export_subset(dataset, split_data, spec_manifest: str | None, start_index: int, max_spectra: int | None):
    selected_indices = resolve_export_indices(split_data, spec_manifest, start_index, max_spectra)
    return _FeaturizedSubset(dataset, selected_indices), selected_indices


def main():
    from benchmark_spec2mol import (
        load_config,
        load_mist_encoder,
        load_spec_data,
        merge_config_with_args,
    )
    from dlm.utils.benchmark_utils import (
        binarize_fingerprint,
        compute_morgan_fingerprint,
        get_inchikey_first_block,
    )
    from dlm.utils.utils_chem import smiles_to_safe
    from mist.data.datasets import get_paired_loader

    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be positive.')
    if args.num_workers < 0:
        raise ValueError('--num-workers cannot be negative.')
    config = merge_config_with_args(load_config(args.config), args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f'Output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset, split_data = load_spec_data(
        config['data'],
        config['mist_encoder'],
        config['evaluation']['split'],
        shuffle=False,
    )
    encoder = load_mist_encoder(config['mist_encoder'], device)
    dataset_subset, selected_indices = build_export_subset(
        dataset,
        split_data,
        args.spec_manifest,
        args.start_index,
        args.max_spectra,
    )
    dataloader = get_paired_loader(
        dataset_subset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    fp_cfg = config['fingerprint']
    fp_bits = fp_cfg['bits']
    fp_radius = fp_cfg['radius']
    fp_threshold = fp_cfg['threshold']
    num_to_process = len(selected_indices)

    rows = []
    mist_probs = []
    mist_binary = []
    ground_truth = []

    cursor = 0
    progress = tqdm(total=num_to_process, desc='Exporting fingerprints')
    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            batch_probs, _ = encoder(batch)
            batch_probs = batch_probs.detach().cpu().numpy().astype(np.float32)
        batch_indices = selected_indices[cursor : cursor + len(batch_probs)]
        if len(batch_indices) != len(batch_probs):
            raise AssertionError('MIST batch and selected index counts diverged.')
        for original_idx, pred_probs in zip(batch_indices, batch_probs):
            spec, mol = split_data[original_idx]
            smiles = mol.get_smiles()
            gt_fp = compute_morgan_fingerprint(smiles, fp_bits, fp_radius)
            if gt_fp is None:
                raise ValueError(f'Could not compute target fingerprint for {spec.get_spec_name()}')
            pred_binary = binarize_fingerprint(pred_probs, fp_threshold).astype(np.float32)
            rows.append({
                'fingerprint_index': len(rows),
                'split': args.split,
                'spec_name': spec.get_spec_name(),
                'smiles': smiles,
                'input': smiles_to_safe(smiles),
                'inchi_key': mol.get_inchikey(),
                'inchi_key_first_block': get_inchikey_first_block(mol.get_inchikey()),
            })
            mist_probs.append(pred_probs)
            mist_binary.append(pred_binary)
            ground_truth.append(gt_fp.astype(np.float32))
        cursor += len(batch_probs)
        progress.update(len(batch_probs))
    progress.close()
    if cursor != len(selected_indices) or len(rows) != len(selected_indices):
        raise AssertionError('MIST export did not cover every selected spectrum.')

    metadata_path = output_dir / 'metadata.csv'
    fp_path = output_dir / 'fingerprints.npz'
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    np.savez_compressed(
        fp_path,
        mist_probs=np.stack(mist_probs),
        mist_binary=np.stack(mist_binary),
        ground_truth=np.stack(ground_truth),
    )

    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved fingerprints to: {fp_path}")
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(config['mist_encoder']['checkpoint']).expanduser().resolve()
    data_dir = Path(config['data']['datadir']).expanduser().resolve()
    manifest_path = (
        Path(args.spec_manifest).expanduser().resolve() if args.spec_manifest else None
    )

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    commit = subprocess.run(
        ['git', '-C', project_root, 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ['git', '-C', project_root, 'status', '--porcelain', '--untracked-files=all'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip())
    run_manifest = {
        'schema_version': 1,
        'purpose': 'mist_fingerprint_export',
        'status': 'completed',
        'code': {'commit': commit, 'dirty': dirty},
        'device': str(device),
        'parameters': vars(args),
        'inputs': {
            'config': {'path': str(config_path), 'sha256': sha256_file(config_path)},
            'mist_checkpoint': {
                'path': str(checkpoint_path),
                'sha256': sha256_file(checkpoint_path),
            },
            'labels': {
                'path': str(data_dir / 'labels.tsv'),
                'sha256': sha256_file(data_dir / 'labels.tsv'),
            },
            'split': {
                'path': str(data_dir / 'split.tsv'),
                'sha256': sha256_file(data_dir / 'split.tsv'),
            },
            'spec_manifest': (
                {'path': str(manifest_path), 'sha256': sha256_file(manifest_path)}
                if manifest_path
                else None
            ),
        },
        'target_fields_used_by_model': [],
        'target_fields_exported_for_metrics_only': [
            'smiles', 'inchi_key', 'ground_truth_fingerprint'
        ],
        'outputs': {
            'metadata_csv': {
                'path': str(metadata_path),
                'sha256': sha256_file(metadata_path),
                'row_count': len(rows),
            },
            'fingerprints_npz': {
                'path': str(fp_path),
                'sha256': sha256_file(fp_path),
                'shape': [len(rows), fp_bits],
                'keys': ['mist_probs', 'mist_binary', 'ground_truth'],
            },
        },
    }
    (output_dir / 'run_manifest.json').write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(f"Rows: {len(rows)}")
    print(json.dumps(run_manifest['outputs'], indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
