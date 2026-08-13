"""Paired conditioning metrics on the NPLIB1 axis, CPU only.

Convention (calibrated to reproduce DECODER_PROGRAM.md:118-120):
  pool  = the 7,144 adaptation targets (train 6,748 + val 396 rows, undeduped)
  shell = |ExactMolWt(candidate) - ExactMolWt(gold)| <= 0.5 Da
  gold is prepended to the shell; top-1 = argmax Tanimoto lands on the gold
  connectivity block.
"""
import argparse, json
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog('rdApp.*')

GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
_cache = {}

def feat(smi):
    if smi not in _cache:
        m = Chem.MolFromSmiles(smi)
        fp = np.zeros(4096, np.uint8)
        fp[list(GEN.GetFingerprint(m).GetOnBits())] = 1
        _cache[smi] = (ExactMolWt(m), fp)
    return _cache[smi]

def tan(p, M):
    inter = (M & p).sum(1).astype(np.float32)
    un = (M | p).sum(1).astype(np.float32)
    return np.where(un > 0, inter / np.maximum(un, 1e-9), 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', required=True)
    ap.add_argument('--probs-key', default='probs')
    ap.add_argument('--ids-key', default='spectrum_ids')
    ap.add_argument('--split', default='test')
    ap.add_argument('--panel', default=None)
    ap.add_argument('--thresholds', default='0.95')
    ap.add_argument('--label', default='')
    a = ap.parse_args()

    md = {s: pd.read_csv(f'runs/mist/{s}/metadata.csv') for s in ('train', 'val', 'test')}
    tv = pd.concat([md['train'], md['val']], ignore_index=True)
    pm, pf, pb = [], [], []
    for smi, blk in zip(tv.smiles, tv.inchi_key_first_block):
        m, f = feat(smi); pm.append(m); pf.append(f); pb.append(blk)
    pm = np.array(pm); pf = np.stack(pf); pb = np.array(pb)

    d = np.load(a.probs, allow_pickle=True)
    probs = d[a.probs_key].astype(np.float32)
    ids = ([str(x) for x in d[a.ids_key]] if a.ids_key in d
           else list(md[a.split].spec_name.values))
    meta = pd.concat(md.values(), ignore_index=True).drop_duplicates('spec_name').set_index('spec_name')
    if a.panel:
        keep = set(pd.read_csv(a.panel, sep='\t').spec_name)
        sel = [i for i, s in enumerate(ids) if s in keep]
        probs = probs[sel]; ids = [ids[i] for i in sel]

    gold = [feat(meta.loc[s, 'smiles']) for s in ids]
    gblk = [meta.loc[s, 'inchi_key_first_block'] for s in ids]
    gfp = np.stack([g[1] for g in gold])

    out = {'label': a.label, 'source': a.probs, 'n': len(ids), 'pool_rows': int(len(pm))}
    for thr in [float(x) for x in a.thresholds.split(',')]:
        binp = (probs >= thr).astype(np.uint8)
        active = binp.sum(1)
        inter = (binp & gfp).sum(1).astype(np.float32)
        un = (binp | gfp).sum(1).astype(np.float32)
        t = np.where(un > 0, inter / np.maximum(un, 1e-9), 0.0)
        rec = np.where(gfp.sum(1) > 0, inter / np.maximum(gfp.sum(1), 1), 0.0)
        prec = np.where(active > 0, inter / np.maximum(active, 1), 0.0)
        win = 0; strict = 0; comps = []
        for k, s in enumerate(ids):
            gm = gold[k][0]
            idxs = np.where(np.abs(pm - gm) <= 0.5)[0]
            cand = np.vstack([gfp[k][None, :], pf[idxs]])
            cb = [gblk[k]] + list(pb[idxs])
            sc = tan(binp[k], cand); comps.append(len(cb) - 1)
            if cb[int(np.argmax(sc))] == gblk[k]:
                win += 1
            other = np.array([sc[i] for i in range(len(cb)) if cb[i] != gblk[k]])
            if len(other) == 0 or sc[0] > other.max() + 1e-9:
                strict += 1
        out[f'thr={thr}'] = {
            'median_tanimoto': round(float(np.median(t)), 4),
            'mean_tanimoto': round(float(t.mean()), 4),
            'mean_recall': round(float(rec.mean()), 4),
            'mean_precision': round(float(prec.mean()), 4),
            'active_bits_mean': round(float(active.mean()), 2),
            'active_bits_median': float(np.median(active)),
            'active_bits_p5_p95': [float(np.percentile(active, 5)), float(np.percentile(active, 95))],
            'active_bits_min_max': [int(active.min()), int(active.max())],
            'zero_bit_rows': int((active == 0).sum()),
            'mass_shell_top1': round(win / len(ids), 4),
            'mass_shell_top1_n': win,
            'mass_shell_top1_strict': round(strict / len(ids), 4),
            'mass_shell_top1_strict_n': strict,
            'median_competitors': float(np.median(comps)),
        }
    print(json.dumps(out, indent=2))

main()
