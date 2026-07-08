#!/usr/bin/env python
"""Export MIST probability confidence diagnostics without DLM generation."""

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, Optional

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

from benchmark_dlm_fingerprint_robustness import fingerprint_error_stats  # noqa: E402
from benchmark_spec2mol import load_config, load_mist_encoder, load_spec_data  # noqa: E402
from dlm.utils.benchmark_utils import (  # noqa: E402
    compute_morgan_fingerprint,
    get_inchikey_first_block,
    normalize_formula,
    sparsify_fingerprint,
)
from mist.data.datasets import get_paired_loader  # noqa: E402

RDLogger.DisableLog('rdApp.*')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export MIST confidence diagnostics for confidence-gated sparsification analysis.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, default='configs/spec2mol_benchmark_msg.yaml')
    parser.add_argument('--mist-checkpoint', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--max-spectra', type=int, default=None)
    parser.add_argument('--start-index', type=int, default=0, help='Start offset within the selected split.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp-threshold', type=float, default=0.5)
    parser.add_argument(
        '--fp-sparsify-mode',
        choices=['threshold', 'topk', 'quantile'],
        default='threshold',
    )
    parser.add_argument('--fp-top-k', type=int, default=None)
    parser.add_argument('--fp-quantile', type=float, default=None)
    parser.add_argument('--fp-min-threshold', type=float, default=None)
    parser.add_argument('--fp-max-threshold', type=float, default=None)
    return parser.parse_args()


def configure(args) -> Dict[str, Any]:
    config = load_config(args.config)
    config['data']['datadir'] = args.data_dir
    config['data']['labels_file'] = os.path.join(args.data_dir, 'labels.tsv')
    config['data']['split_file'] = os.path.join(args.data_dir, 'split.tsv')
    config['data']['spec_folder'] = os.path.join(args.data_dir, 'spec_files')
    config['data']['subform_folder'] = os.path.join(args.data_dir, 'subformulae/default_subformulae')
    config['mist_encoder']['checkpoint'] = args.mist_checkpoint
    config['evaluation']['split'] = args.split
    config['fingerprint']['threshold'] = args.fp_threshold
    config['fingerprint']['sparsify_mode'] = args.fp_sparsify_mode
    config['fingerprint']['top_k'] = args.fp_top_k
    config['fingerprint']['quantile'] = args.fp_quantile
    config['fingerprint']['min_threshold'] = args.fp_min_threshold
    config['fingerprint']['max_threshold'] = args.fp_max_threshold
    return config


def write_json(path: str, payload: Dict[str, Any]):
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def formula_from_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return normalize_formula(rdMolDescriptors.CalcMolFormula(mol))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = configure(args)
    os.makedirs(args.output_dir, exist_ok=True)
    write_json(os.path.join(args.output_dir, 'config.json'), config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    dataset, split_data = load_spec_data(
        config['data'],
        config['mist_encoder'],
        args.split,
        shuffle=False,
    )
    mist_encoder = load_mist_encoder(config['mist_encoder'], device)
    dataloader = get_paired_loader(dataset, shuffle=False, batch_size=1, num_workers=0)

    fp_cfg = config['fingerprint']
    fp_bits = fp_cfg['bits']
    fp_radius = fp_cfg['radius']
    num_to_process = min(len(dataset), args.max_spectra) if args.max_spectra else len(dataset)
    if args.start_index < 0 or args.start_index >= len(dataset):
        raise ValueError(f'start_index must be in [0, {len(dataset) - 1}]')
    available = len(dataset) - args.start_index
    num_to_process = min(available, args.max_spectra) if args.max_spectra else available
    rows = []
    start_time = time.time()

    progress = tqdm(total=num_to_process, desc='Exporting diagnostics')
    for idx, batch in enumerate(dataloader):
        if idx < args.start_index:
            continue
        if len(rows) >= num_to_process:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        spec, mol = split_data[idx]
        target_smiles = mol.get_smiles()
        target_fp = compute_morgan_fingerprint(target_smiles, fp_bits, fp_radius)
        if target_fp is None:
            print(f'Warning: Could not compute FP for {target_smiles}, skipping')
            continue

        with torch.no_grad():
            mist_probs, _ = mist_encoder(batch)
            mist_probs = mist_probs.cpu().numpy()[0]

        mist_binary = sparsify_fingerprint(
            mist_probs,
            threshold=fp_cfg['threshold'],
            mode=fp_cfg.get('sparsify_mode', 'threshold'),
            top_k=fp_cfg.get('top_k'),
            quantile=fp_cfg.get('quantile'),
            min_threshold=fp_cfg.get('min_threshold'),
            max_threshold=fp_cfg.get('max_threshold'),
        )
        row = fingerprint_error_stats(target_fp, mist_binary, mist_probs)
        row.update({
            'row_index': idx,
            'spec_name': spec.get_spec_name(),
            'target_smiles': target_smiles,
            'target_inchi_key': get_inchikey_first_block(mol.get_inchikey()),
            'target_formula': formula_from_smiles(target_smiles),
        })
        rows.append(row)
        progress.update(1)
    progress.close()

    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(os.path.join(args.output_dir, 'mist_confidence_diagnostics.csv'), index=False)
    write_json(
        os.path.join(args.output_dir, 'run_state.json'),
        {
            'completed': True,
            'processed_spectra': len(rows),
            'requested_spectra': num_to_process,
            'elapsed_time_seconds': time.time() - start_time,
        },
    )
    print(f'Wrote {len(rows)} diagnostics rows to {args.output_dir}/mist_confidence_diagnostics.csv')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nExport interrupted.')
        sys.exit(1)
