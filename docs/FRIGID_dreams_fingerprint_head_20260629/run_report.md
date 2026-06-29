# DreaMS Fingerprint-Head Full Spectrum Run

Date: 2026-06-29

## Objective

Run the DreaMS fingerprint-head workflow as a real remote training job on
`spectrum`, evaluate it on the MassSpecGym validation split, and decide whether
it is strong enough to replace the current MIST spectrum-to-fingerprint backend
inside FRIGID.

This was not a local smoke test. The run prepared the full train and validation
packages, extracted DreaMS embeddings for both splits, trained the supervised
fingerprint head for 20 epochs, predicted validation fingerprints, and computed
real full-validation fingerprint metrics.

## Code and Runtime

- Local branch: `dreams-fingerprint-head`
- Remote repository target: `https://github.com/SergeiNikolenko/FRIGID.git`
- Remote checkout: `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head`
- Code commit used by the remote run:
  `aff6e67882dfb8166af18d5d9c128d3ace2b8924`
- Remote run directory:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_full_20260629T110040Z`
- Run script:
  `docs/FRIGID_dreams_fingerprint_head_20260629/run_full_dreams_pipeline.sh`

The remote checkout reported branch `main`, but the commit was explicitly reset
to `aff6e67882dfb8166af18d5d9c128d3ace2b8924` from the user's fork before the
run. The existing upstream `coleygroup/FRIGID` checkout was used only as a data
and checkpoint provenance source, not as the Git target for this work.

## Inputs

- MSG data root:
  `/home/nikolenko/work/Projects/FRIGID/repro_cache/msg`
- DLM checkpoint for the optional downstream test:
  `/home/nikolenko/work/Projects/FRIGID/repro_cache/DLM.ckpt`
- Fingerprint target: 4096-bit Morgan fingerprint, radius 2
- DreaMS embedding size: 1024

## Runtime Setup Notes

The DreaMS environment on `spectrum` needed dependency repair before the
embedding API could run. The following packages were installed into the
DreaMS/MolForge virtual environment:

- `umap-learn`
- `igraph`
- `plotly`
- `pandarallel`
- `huggingface_hub`
- `ase`
- `wandb`
- `pyopenms`
- `msml_legacy_architectures` from
  `git+https://github.com/roman-bushuiev/msml_legacy_architectures.git@main`

The first real embedding extraction downloaded the DreaMS pretrained
checkpoints into the DreaMS source tree:

- `embedding_model.ckpt`
- `ssl_model.ckpt`

## Dataset and Embeddings

| Split | Rows | Fingerprint bits | Embedding dim | Local summary |
| --- | ---: | ---: | ---: | --- |
| train | 191,216 | 4,096 | 1,024 | `train_dataset/summary.json`, `train_embeddings/summary.json` |
| val | 19,043 | 4,096 | 1,024 | `val_dataset/summary.json`, `val_embeddings/summary.json` |

Large artifacts such as `.mgf`, `.hdf5`, `.npz` embeddings, target
fingerprints, predictions, and model checkpoints remain on `spectrum`. This
repository stores the compact summaries and metrics only.

## Training

The supervised head was trained for 20 epochs:

```text
DreaMS embedding (1024) -> MLP -> 4096 Morgan fingerprint logits
```

Training used the full train split and an explicit full validation split. The
best and final epoch was epoch 20 by validation loss, mean Tanimoto, and bit F1.

Key training settings and artifacts:

- Positive class weight: `95.89204431233892`
- Threshold for binary metrics: `0.5`
- Model checkpoint:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_full_20260629T110040Z/fingerprint_head/dreams_fingerprint_head.pt`
- Training metrics:
  `docs/FRIGID_dreams_fingerprint_head_20260629/fingerprint_head/training_metrics.json`

## Full Validation Fingerprint Metrics

The saved full-validation metrics are in
`docs/FRIGID_dreams_fingerprint_head_20260629/val_evaluation/fingerprint_metrics.json`.

| Metric | Value |
| --- | ---: |
| validation rows | 19,043 |
| mean Tanimoto | 0.1238048323 |
| median Tanimoto | 0.1048951074 |
| bit accuracy | 0.9300574225 |
| bit precision | 0.1238794961 |
| bit recall | 0.5741983662 |
| bit F1 | 0.2037921789 |
| mean predicted active bits | 295.9601440430 |
| mean target active bits | 63.8514404297 |

The high bit accuracy is not a success signal because the target fingerprints
are sparse. The more informative metrics are Tanimoto, precision, recall, F1,
and the active-bit count mismatch.

## Downstream DLM Attempt

A 200-spectrum downstream DLM benchmark using the DreaMS-head fingerprints was
started, but it was not treated as a valid completed result. The run reached
about 59 / 200 spectra after roughly 41 minutes and did not produce
`aggregate_statistics.json`.

The DLM stage was stopped after the full validation fingerprint metrics were
available because the encoder result was already far below the decision
threshold. Running the slow DLM generation to completion would not answer the
main question more clearly: this frozen-DreaMS plus shallow-head encoder is not
competitive enough to promote into the DLM backend.

## Interpretation

This experiment did complete the serious first test: full remote training on
`spectrum` and real validation metrics on 19,043 spectra.

The result is negative:

- The DreaMS-head mean fingerprint Tanimoto is `0.1238`.
- The documented MIST reference target for this gate is about `0.52`.
- The model predicts too many active bits: about `296` predicted active bits
  versus about `64` target active bits.
- Precision is very low (`0.1239`) while recall is moderate (`0.5742`), which
  means the head learned broad active-bit coverage but not a selective
  fingerprint representation.

Conclusion: frozen DreaMS embeddings plus a supervised MLP fingerprint head are
not a sufficient replacement for MIST in FRIGID.

## Next Technical Steps

The next serious DreaMS path should not be another shallow-head run with the
same setup. The priority should be:

1. Recalibrate the current head as a cheap diagnostic: tune thresholds, test
   top-k active-bit selection, and inspect per-bit frequency calibration.
2. If calibration cannot move validation Tanimoto materially, fine-tune DreaMS
   itself on the fingerprint objective instead of keeping it frozen.
3. Only after the encoder exceeds the MIST fingerprint-quality gate should
   DreaMS-head fingerprints be pushed through a full DLM benchmark.
4. If the encoder improves but DLM does not, then adapt DLM on predicted
   fingerprints as a separate decoder-robustness experiment.

## Local Evidence Package

This directory stores compact, reviewable evidence:

- `artifact_manifest.json`
- `run_full_dreams_pipeline.sh`
- `train_dataset/summary.json`
- `val_dataset/summary.json`
- `train_embeddings/summary.json`
- `val_embeddings/summary.json`
- `fingerprint_head/training_metrics.json`
- `val_predictions/summary.json`
- `val_evaluation/fingerprint_metrics.json`

