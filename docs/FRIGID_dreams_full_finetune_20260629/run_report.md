# FRIGID DreaMS Full Fine-Tune Run

Date: 2026-06-29

## Objective

Train DreaMS as a real replacement candidate for the current MIST
spectrum-to-fingerprint backend. This run is intentionally not a smoke test and
not another frozen-embedding head: the DreaMS spectrum encoder is part of the
training graph and is fine-tuned together with a 4096-bit Morgan fingerprint
head.

## Why this run exists

Earlier DreaMS experiments used frozen DreaMS embeddings or DreaMS embeddings as
side information:

- Frozen DreaMS head: substantially below MIST on validation fingerprint
  Tanimoto.
- MIST + DreaMS linear blend: best validation point was pure MIST.
- MIST + DreaMS residual: anchored residuals gave only a very small gain, below
  the replacement gate.
- DreaMS distilled replacement head: DreaMS-only frozen-embedding student was
  still far below MIST in the first completed variants.

Those results do not answer whether the DreaMS encoder itself can be adapted to
the FRIGID/MassSpecGym fingerprint task. This experiment addresses that gap.

## Remote run

- Host: `spectrum`
- FRIGID checkout:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head`
- DreaMS/MolForge checkout:
  `/home/nikolenko/work/Projects/mist_molforge_autoresearch`
- Run directory:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_full_finetune_20260629T174403Z`
- tmux session: `frigid_dreams_full_finetune`
- Source FRIGID commit:
  `d2c48dd865476ddc59a8b2e1acefeac301bb0086`

## Data

The run reuses the previously exported full MSG/FRIGID DreaMS dataset artifacts:

- Train spectra:
  `runs/dreams_fingerprint_head_msg_full_20260629T110040Z/train_dataset/spectra.hdf5`
- Validation spectra:
  `runs/dreams_fingerprint_head_msg_full_20260629T110040Z/val_dataset/spectra.hdf5`
- Train fingerprints:
  `runs/dreams_fingerprint_head_msg_full_20260629T110040Z/train_dataset/fingerprints.npz`
- Validation fingerprints:
  `runs/dreams_fingerprint_head_msg_full_20260629T110040Z/val_dataset/fingerprints.npz`

Dataset sizes confirmed at launch:

- Train: 191,216 spectra, 24,294 unique SMILES, 4096-bit Morgan targets.
- Validation: 19,043 spectra, 3,270 unique SMILES, 4096-bit Morgan targets.

The run directory contains symlinks to the HDF5 spectra files and generated
`joblib` files with the exact `smiles` order plus `fingerprints_map`, matching
the DreaMS full-training pipeline interface.

## Model and training setup

Training script:

- Base script:
  `/home/nikolenko/work/Projects/mist_molforge_autoresearch/scripts/dreams_molforge_pipeline/3.train_full.py`
- Run-local patched copy:
  `runs/dreams_full_finetune_20260629T174403Z/train_full_patched.py`

The run-local patch removes the deprecated `verbose=True` argument from
`torch.optim.lr_scheduler.ReduceLROnPlateau`, which is incompatible with the
installed PyTorch version. The initial attempt reached dataset loading and then
failed only on this scheduler API mismatch; data loading itself was valid.

Training configuration from the DreaMS full-training script:

- Backbone: pretrained DreaMS encoder via `PreTrainedModel.from_name(DREAMS_EMBEDDING)`.
- Head: MLP fingerprint predictor, 1024 input dimension to 4096 output bits.
- Loss: 0.5 binary cross-entropy plus 0.5 soft Tanimoto loss.
- Batch size: 512.
- Epoch budget: 100.
- Optimizer: AdamW with separate encoder/head learning rates.
- Encoder learning rate: 1e-4.
- Head learning rate: 1e-3.
- Validation: full 19,043-spectrum validation split each epoch.

Because `tune_encoder_epochs` is set to `0` in the script and the training loop
unfreezes the encoder when `epoch >= tune_encoder_epochs`, this run fine-tunes
the DreaMS encoder from epoch 0.

## Launch status

The first launch failed before training because of the PyTorch scheduler API
mismatch:

```text
TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'
```

After patching the run-local script, the full run was restarted in tmux. The
latest checked status showed:

```text
Overall Progress: 0/100 epochs
Epoch 0 [Train]: 2/373 batches, loss=0.6645
```

This confirms that the full DreaMS encoder plus fingerprint head training loop
started on the complete train split.

## Success criteria

The run should not be judged only by training loss. The replacement decision
requires validation and downstream FRIGID metrics:

1. Validation fingerprint Tanimoto must beat the current MIST fingerprint
   baseline by a meaningful margin, not just match a weak frozen DreaMS head.
2. The best checkpoint must be exported to the FRIGID precomputed-fingerprint
   handoff format: `fingerprints.npz` with probability rows and matching
   `metadata.csv`.
3. The exported predictions must be evaluated through the existing FRIGID/DLM
   robustness path, not only the DreaMS training script's internal validation
   loop.
4. A replacement should be accepted only if it improves the real downstream DLM
   metrics or clearly improves fingerprint quality in MIST failure regions
   without regressing the overall validation set.

## Next actions

- Monitor `frigid_dreams_full_finetune` until at least the first full validation
  epoch finishes.
- Save the best checkpoint and validation metrics into this documentation
  folder.
- Add an export/evaluation step for the best full-fine-tuned DreaMS checkpoint.
- Compare against MIST using the same validation split and the existing
  `sweep_fingerprint_predictions.py` / DLM robustness tooling.
