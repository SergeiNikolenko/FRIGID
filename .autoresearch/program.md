# FRIGID Throughput And Quality Program

Goal: reduce FRIGID training and inference wall time while preserving
MassSpecGym spectrum-to-molecule quality.

The campaign should optimize throughput with explicit quality guards. Speed-only
changes are not accepted unless `exact_top1`, `tanimoto_top1`, and
`formula_success` remain within the scorer contract.

Primary surfaces:

- Base MSG inference in `scripts/benchmark_spec2mol.py`.
- Formula-filtered generation in `src/dlm/utils/benchmark_utils.py`.
- DLM generation in `src/dlm/sampler.py`.
- ICEBERG scaling in `scripts/spec2mol_scaling.py` and `src/dlm/iceberg_sampler.py`.
- Training harness in `scripts/train.py`, `configs/fp2mol_pretraining.yaml`, and
  `src/dlm/utils/utils_data.py`.

Do not optimize directly on the full test set during iteration. Use the scorer
subset for throughput and quality guard checks, then confirm on larger runs.
