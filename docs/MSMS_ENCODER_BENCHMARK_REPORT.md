# MS/MS Encoder Benchmark: Decision Report and Reproducible Protocol

Date: 2026-07-15

## Executive decision

MIST remains the FRIGID production encoder. The existing DreaMS replacement
line is closed: its strongest full fine-tune reached approximately 0.258 mean
fingerprint Tanimoto, while the best MIST+DreaMS residual improved MIST by only
0.000684, below the 0.005 promotion gate.

The next comparison wave is:

1. MSBERT, MS2DeepScore, JESTR, and SpecEmbedding with the same frozen-backbone
   `LayerNorm -> Linear(d, 4096)` probe.
2. IDSL_MINT as the first direct Morgan-4096 challenger trained from scratch.
3. DiffMS and MSFlow in a separate end-to-end track. Their decoder gains must
   not be reported as encoder gains.

This repository now contains an ID-safe evaluator that rejects incompatible
rows, fingerprint dimensions, thresholds, and leakage evidence instead of
silently producing a score.

## What the task is about

FRIGID currently relies on a spectrum encoder that predicts the 4096-bit
fingerprint consumed by the molecular decoder. The research question is not
whether another model can produce a vector of length 4096. It is whether an
alternative encoder predicts the same ordered Morgan bits more accurately on
the same spectra, without train/evaluation leakage, and whether that gain
survives the DLM decoder.

The task therefore has four separate deliverables:

- inventory and classify available MS/MS encoders;
- lock one comparison contract and one evidence format;
- reproduce the existing MIST baseline and compare candidates pairwise;
- run end-to-end DLM only for encoders that pass the encoder-level gate.

## Evidence inventory

The required historical data are already available on `spectrum`; waiting for
an additional handoff is not necessary.

| Item | Evidence |
|---|---|
| MSG labels | `/home/nikolenko/work/Projects/FRIGID/repro_cache/msg/labels.tsv`; 231,104 rows; SHA-256 `feb8c9c58be5ced28c2e370f196638610c013cc656d77eebfbe5d42d215ae8fb` |
| MSG split | `/home/nikolenko/work/Projects/FRIGID/repro_cache/msg/split.tsv`; SHA-256 `5b049fd52855eb491719bd3038fb1643c662fa0071367b1fd36a280b268a21e4` |
| Eligible rows | train 191,216; validation 19,043; test 17,082 |
| MIST checkpoint | `/home/nikolenko/work/Projects/FRIGID/repro_cache/mist_msg.pt`; SHA-256 `09b4e93e8d5d35e7472f4d187d72ca316b125368232bb69148df101ec211b92a` |
| DLM checkpoint | `/home/nikolenko/work/Projects/FRIGID/repro_cache/DLM.ckpt`; SHA-256 `b6177c2d43448380aba80ff41c01461ea34ca2ca93b213986954c5afb7f0f457` |
| Clean historical code | `dreams-fingerprint-head`, commit `5fb8fd62715b9b779761fa945af3f80c2a191081` |
| New baseline export | Slurm job `172`, completed in 3m07s with exit `0:0`; `/home/nikolenko/work/Projects/FRIGID_encoder_benchmark_runs/20260715T182006Z_mist_val` |

The older full-validation prediction NPZ files were no longer present. Metrics
JSON and checkpoints survived, but they are insufficient for paired error
analysis. The completed baseline export restores the missing per-spectrum
evidence and records input and output hashes:

- `fingerprints.npz`: 19,043 rows, 4096 bits, 558 MiB, SHA-256
  `181eff747b01fb15568d9bfc7bdf44f5394a9ed85fc8dce5ee10a1c4ed5c053a`;
- `metadata.csv`: SHA-256
  `91a43b3fda96a5b1310ef18a6c5b63a9de6e62cfdf859ab383b5186f6ce9e8ff`;
- `summary.json`: SHA-256
  `f58f7df65ba4b763a2f69041909d79fa72ba5063dcf209197e62fb90209b8785`.

## Existing results, kept in separate lanes

### Encoder-level validation

The historical MIST sweep used 19,043 eligible validation spectra, Morgan
radius 2 with 4096 bits, and selected threshold 0.25.

| Model or intervention | Mean fingerprint Tanimoto | Decision |
|---|---:|---|
| MIST, threshold 0.25 | 0.542043 | Baseline |
| DreaMS frozen head | 0.123805 | Closed |
| DreaMS calibrated | 0.234203 | Closed |
| DreaMS distillation | 0.240354 | Closed |
| DreaMS full fine-tune | 0.2580 | Closed; strong overfit |
| DreaMS staged adapter | 0.232862 | Closed |
| DreaMS spectral JEPA | 0.192590 | Closed |
| MIST + DreaMS residual | 0.542726 | Gain 0.000684; fails 0.005 gate |
| SpecEmbedding historical probe | 0.3184 | Below MIST; no DLM run |

The historical MIST threshold was selected on the same full validation surface
used for reporting. It remains the continuity reference, but it is not the
prospective decision surface. New comparisons must freeze each threshold on a
calibration partition before evaluating the disjoint decision partition.

### End-to-end test

The full MIST+DLM test used 17,082 eligible spectra and threshold 0.187. Its
mean input-fingerprint Tanimoto was 0.540670; exact top-1/top-10 were
0.109706/0.123932 and generated-structure Tanimoto top-1/top-10 were
0.459838/0.484233. These values must not be mixed with the validation encoder
table because both the split and threshold differ.

A separate 1,400-spectrum paired diagnostic showed the decoder upper bound:

| Fingerprint input | Exact top-1 | Molecular Tanimoto top-1 | Formula success |
|---|---:|---:|---:|
| Ground truth | 0.4879 | 0.8130 | 0.7643 |
| MIST binary | 0.1386 | 0.5677 | 0.6936 |

The run was partial (`completed: false`), so it is diagnostic evidence rather
than a final benchmark. Two attempted DLM adaptations (2,500 and 10,000 steps)
regressed and are closed.

## Locked benchmark contract

### Data and information boundary

- Train: eligible official MSG train rows only.
- Calibration: a deterministic molecule-clustered partition of official MSG
  validation, used only to freeze thresholds and probe hyperparameters.
- Evaluation: the remaining molecule-disjoint validation partition.
- Final confirmation: official MSG test, exactly once after a model passes the
  validation gate.
- Group identity: first block of the InChIKey.
- Allowed encoder inputs must be declared: peaks, precursor, formula, adduct,
  instrument, and subformula annotations.
- Forbidden inputs: SMILES, InChIKey, ground-truth fingerprint, target active
  bit count, or any statistic derived from evaluation labels.
- External pretraining overlap is `unknown` until audited; it is never silently
  labeled leakage-safe.

### Fingerprint identity

- Type: RDKit Morgan fingerprint.
- Radius: 2.
- Bits: 4096.
- `useChirality`: false.
- Bit numbering must match the reference exactly.

A custom 4096-dimensional vector or a dense embedding is not automatically a
compatible fingerprint. Dense encoders enter the fingerprint track only through
the same train-only probe. They may also enter a separate retrieval/reranking
track, where raw embedding metrics are appropriate.

### Metrics and promotion gate

Primary metric: per-spectrum binary fingerprint Tanimoto.

Required secondary evidence:

- median and molecule-balanced mean Tanimoto;
- soft Tanimoto and BCE;
- bit precision, recall, F1, FP bits, FN bits, and active-bit counts;
- paired wins/losses/ties versus MIST;
- molecule-cluster paired bootstrap 95% interval;
- latency when the exporter supplies it;
- train/evaluation structure overlap status.

A candidate passes the encoder gate only when all conditions hold:

1. paired mean Tanimoto gain is at least 0.005;
2. the lower bound of the molecule-cluster bootstrap interval is above zero;
3. declared training structures do not overlap evaluation structures;
4. the exact prediction bundle, manifest, code revision, checkpoint, and data
   hashes are retained.

Only a passing encoder is connected to DLM. The end-to-end comparison must use
the same formula source, generation budget, seeds, filtering, and ranking.

## Candidate landscape

| Candidate | Native output | Public implementation status | FRIGID treatment | Priority |
|---|---|---|---|---|
| MIST | Morgan fingerprint | Official code/checkpoint available | Direct baseline | Baseline |
| MSBERT | 512D dense embedding | MIT code and released weights | Shared frozen 512->4096 probe | A |
| MS2DeepScore 2.x | 500D similarity embedding | Apache-2.0 code and pretrained models | Shared frozen 500->4096 probe; separate retrieval test | A |
| JESTR | 512D joint spectrum/molecule embedding | MIT code and checkpoints | Shared probe plus reranking track | A |
| SpecEmbedding | 512D contrastive embedding | Code and published checkpoint bundle | Shared probe; license must be confirmed for the exact bundle | A |
| IDSL_MINT | Sequence of active fingerprint bits | MIT code; no general pretrained checkpoint | Train directly with exact Morgan-4096 settings | B |
| CMSSP | 2048D cross-modal embedding | Apache-2.0 code and weights | Shared 2048->4096 probe | B |
| ChemEmbed | Dense chemical embedding | Code and trained models reported | Shared probe/retrieval track | B |
| DiffMS | Formula-aware encoder inside a diffusion generator | MIT code and released artifacts | Inspect raw 4096 output and full generator separately | B |
| MSFlow | 512D continuous representation plus flow decoder | MIT code; weights published separately | End-to-end track, not direct fingerprint claim | B |
| CLERMS | 200D contrastive embedding | Code; no stable weights/license evidence | Defer | C |
| MetFID | Custom 5618-bit fingerprint | Code and weights; training partly depends on closed NIST data | Incompatible without a new head and retraining | C |
| MSAlign | DreaMS/ChemBERTa aligned embedding | Public 2026 code | Retrieval watchlist; not another DreaMS fingerprint replacement | Watch |

Primary references:

- MIST: <https://doi.org/10.1038/s42256-023-00708-3>
- MSBERT: <https://doi.org/10.1021/acs.analchem.4c02426>
- MS2DeepScore: <https://doi.org/10.1186/s13321-021-00558-4>
- JESTR: <https://doi.org/10.1093/bioinformatics/btaf354>
- SpecEmbedding: <https://doi.org/10.1021/acs.analchem.5c02655>
- IDSL_MINT: <https://doi.org/10.1186/s13321-024-00804-5>
- CMSSP: <https://doi.org/10.1021/acs.analchem.4c03724>
- DiffMS: <https://arxiv.org/abs/2502.09571>
- MSFlow: <https://arxiv.org/abs/2602.19912>
- MSAlign: <https://openreview.net/forum?id=ZoBAklPA7R>

## Reproducible evaluator

The evaluator is `scripts/benchmark_encoder_predictions.py`; validation and
metric primitives are in `src/frigid/encoder_benchmark.py`.

New candidate bundle contract:

- `probs`: finite float array `[N, 4096]` in `[0, 1]`;
- `spectrum_ids`: unique one-dimensional IDs for the same rows;
- `inference_seconds`: optional non-negative per-spectrum latency.

Historical bundles without embedded IDs are supported only with an explicit
companion metadata file. The evaluator rejects duplicate, missing, or extra
IDs and reorders candidate rows by identity rather than position.

Example:

```bash
python scripts/benchmark_encoder_predictions.py \
  --reference-metadata runs/mist_val/metadata.csv \
  --reference-fingerprints runs/mist_val/fingerprints.npz \
  --reference-model mist=mist_probs \
  --prediction msbert=runs/msbert_val/predictions.npz \
  --threshold mist=0.25 \
  --threshold msbert=<frozen-calibration-threshold> \
  --baseline mist \
  --minimum-gain 0.005 \
  --training-identifiers msbert=runs/msbert_train/inchikeys.txt \
  --code-revision <git-commit> \
  --output-dir runs/encoder_comparison
```

The output directory is immutable: the evaluator refuses to overwrite a
non-empty directory. It writes:

- `aggregate_metrics.csv`;
- `per_spectrum_metrics.csv`;
- `paired_deltas.csv`;
- optional `stratified_metrics.csv`;
- `benchmark_summary.json` with hashes, versions, thresholds, ranking, gate
  decisions, and the ordered spectrum-ID hash.

## Execution plan

### Wave 0: baseline restoration

1. Export MIST probabilities and ground-truth fingerprints for all 19,043
   eligible validation spectra from the clean historical commit.
2. Record MSG, config, and checkpoint hashes before inference.
3. Run the new evaluator and retain per-spectrum metrics and checksums.
4. Create the prospective calibration/evaluation manifests and freeze the MIST
   threshold without reading evaluation labels.

### Wave 1: cheap pretrained probes

Run MSBERT, MS2DeepScore, JESTR, and SpecEmbedding with identical rules:

- frozen backbone;
- one `LayerNorm -> Linear(d, 4096)` head;
- class-weighted BCE;
- identical train/calibration/evaluation structure manifests;
- the same optimizer budget and early-stopping rule;
- no model-specific hidden MLP;
- public checkpoint and preprocessing hashes retained.

Fine-tune only the best two probes, with the same limited budget.

### Wave 2: direct and end-to-end challengers

- Train IDSL_MINT with exact Morgan radius 2, 4096-bit, no-chirality targets.
- Evaluate DiffMS raw fingerprint-like output separately from its generator.
- Evaluate MSFlow and DiffMS as full pipelines against the same DLM/generation
  budget only in the end-to-end table.

### Wave 3: DLM confirmation

For any encoder that passes the 0.005 gate, run the paired DLM benchmark on the
locked test set. Report encoder gain, exact top-k, generated-structure Tanimoto,
formula success, wall time, and failure rate as separate columns.

## Final interpretation

The current evidence does not support replacing MIST with DreaMS or a generic
dense embedding. It does support a focused, inexpensive pretrained-probe wave
and one direct IDSL_MINT training run. The new evaluator turns that next wave
from a collection of incomparable experiments into one falsifiable decision:
either a candidate beats MIST by at least 0.005 with a positive paired interval
and clean overlap evidence, or it does not proceed to DLM.
