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
- Superseded offline run directory:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_full_finetune_20260629T174403Z`
- Active online ClearML run directory:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_full_finetune_clearml_20260629T175115Z`
- tmux session: `frigid_dreams_full_finetune`
- Source FRIGID commit:
  `d2c48dd865476ddc59a8b2e1acefeac301bb0086`
- ClearML web server:
  `https://app.clearai.innopolis.university/`
- ClearML project: `FRIGID/DreaMS Replacement`
- ClearML task name: `FRIGID DreaMS full fine-tune MSG 20260629`
- ClearML task id: `588fc21748004c08a0d878662ac05301`
- ClearML results page:
  `https://app.clearai.innopolis.university/projects/3b1b7bc52b61435aab86b06ecafaa508/experiments/588fc21748004c08a0d878662ac05301/output/log`

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

The active ClearML run directory contains symlinks to the HDF5 spectra files and
the generated `joblib` files with the exact `smiles` order plus
`fingerprints_map`, matching the DreaMS full-training pipeline interface.

## Model and training setup

Training script:

- Base script:
  `/home/nikolenko/work/Projects/mist_molforge_autoresearch/scripts/dreams_molforge_pipeline/3.train_full.py`
- Active online ClearML run-local patched copy:
  `runs/dreams_full_finetune_clearml_20260629T175115Z/train_full_clearml.py`

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

## Final status

The first launch failed before training because of the PyTorch scheduler API
mismatch:

```text
TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'
```

After patching the run-local script, the full run was first restarted in an
offline ClearML mode. That offline run was then superseded because it would not
appear in the Innopolis ClearML UI. The canonical run is now the online ClearML
run in `dreams_full_finetune_clearml_20260629T175115Z`.

The online ClearML run created task id `588fc21748004c08a0d878662ac05301` and
completed. Early stopping triggered at epoch 15 after 16 logged epochs.

Best validation checkpoint:

- Best epoch by validation loss and validation Tanimoto: epoch 8.
- Train loss at epoch 8: 0.1253.
- Train Tanimoto at epoch 8: 0.7468.
- Validation loss at epoch 8: 0.3912.
- Validation Tanimoto at epoch 8: 0.2580.

Last epoch before early stopping:

- Epoch: 15.
- Train loss: 0.0775.
- Train Tanimoto: 0.8438.
- Validation loss: 0.4072.
- Validation Tanimoto: 0.2436.

The final script test pass reused the validation split for compatibility with
the original DreaMS full-training script and reported:

```text
Final Test Results:
Loss: 0.3912
Tanimoto: 0.2580
```

The checkpoint files were produced in the active run directory:

- `model/best_model.pth`
- `model/last.pth`

## Interpretation

This full DreaMS encoder fine-tune did not produce a MIST replacement. The best
validation Tanimoto was 0.2580, while the MIST validation baseline from the
blend/residual experiments was approximately 0.5420 on the same fingerprint
handoff target. The model also overfit strongly: train Tanimoto continued rising
to 0.8438, while validation Tanimoto peaked at epoch 8 and then declined.

The result is still useful because it closes the main question left open by the
frozen-embedding experiments: simply unfreezing and fine-tuning the pretrained
DreaMS encoder with the current BCE plus soft-Tanimoto fingerprint objective is
not enough to replace MIST.

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

- Do not promote this checkpoint as a FRIGID backend replacement.
- If continuing with DreaMS, change the training objective or data strategy
  before another long run. The current setup overfits the train split without
  approaching MIST validation quality.
- The next useful experiment should focus on retrieval/contrastive supervision,
  hard negative spectra, or a hybrid objective tied to downstream DLM success,
  rather than repeating the same plain fingerprint BCE plus Tanimoto loss.
