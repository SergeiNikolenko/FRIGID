#!/usr/bin/env python
"""
Export MIST-predicted fingerprints for DLM fine-tuning.

The output contains:
- metadata.csv: input SAFE/SMILES identifiers and split labels.
- fingerprints.npz: MIST probability, binary MIST, and ground-truth fingerprints.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from benchmark_spec2mol import load_config, load_mist_encoder, load_spec_data, merge_config_with_args  # noqa: E402
from dlm.utils.benchmark_utils import binarize_fingerprint, compute_morgan_fingerprint, get_inchikey_first_block  # noqa: E402
from dlm.utils.benchmark_selection import load_spec_manifest, resolve_selected_indices  # noqa: E402
from dlm.utils.utils_chem import smiles_to_safe  # noqa: E402
from mist.data.datasets import get_paired_loader  # noqa: E402

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
    parser.add_argument('--batch-size', type=int, default=None, help=argparse.SUPPRESS)
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
    args = parse_args()
    config = merge_config_with_args(load_config(args.config), args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
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
    dataloader = get_paired_loader(dataset_subset, shuffle=False, batch_size=1, num_workers=0)

    fp_cfg = config['fingerprint']
    fp_bits = fp_cfg['bits']
    fp_radius = fp_cfg['radius']
    fp_threshold = fp_cfg['threshold']
    num_to_process = len(selected_indices)

    rows = []
    mist_probs = []
    mist_binary = []
    ground_truth = []

    for original_idx, batch in zip(
        selected_indices,
        tqdm(dataloader, total=num_to_process, desc='Exporting fingerprints'),
    ):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        spec, mol = split_data[original_idx]
        smiles = mol.get_smiles()
        gt_fp = compute_morgan_fingerprint(smiles, fp_bits, fp_radius)
        if gt_fp is None:
            print(f"Warning: Could not compute FP for {smiles}, skipping")
            continue

        with torch.no_grad():
            pred_probs, _ = encoder(batch)
            pred_probs = pred_probs.cpu().numpy()[0].astype(np.float32)
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

    metadata_path = os.path.join(args.output_dir, 'metadata.csv')
    fp_path = os.path.join(args.output_dir, 'fingerprints.npz')
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    np.savez_compressed(
        fp_path,
        mist_probs=np.stack(mist_probs),
        mist_binary=np.stack(mist_binary),
        ground_truth=np.stack(ground_truth),
    )

    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved fingerprints to: {fp_path}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
