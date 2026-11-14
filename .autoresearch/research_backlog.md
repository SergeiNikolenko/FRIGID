# Research Backlog

## P0 Correctness And Comparability

- Fix or avoid the built-in multi-GPU base path before relying on it. The worker
  path assigns `generation_time = gen_time`, but `gen_time` is not defined in
  `_worker_process_spectra`.
- Keep the optimization target single-GPU. Multi-GPU sharding is only a
  correctness/parallel throughput option and must not hide per-case latency or
  single-device underutilization.
- Ensure every scorer run records the exact command, commit, host, GPU, model
  checkpoints, `formula_matches`, `max_attempts`, `batch_size`, and output JSON.
- Keep smoke settings separate from paper-like settings. Smoke uses only for
  integration; paper-like base uses `formula_matches=10` and `max_attempts=100`.

## P0 Inference Throughput

- Profile `generate_with_formula_filter`: separate DLM generation time from RDKit
  validation, formula calculation, InChI key generation, Morgan fingerprint, and
  CSV/result assembly.
- Test larger generation batches on one GPU. Current benchmark uses
  `batch_size=16`; on A100 80GB it uses only about 4 GB VRAM, and on RTX 4090
  free memory should still allow controlled 32/64 tests. This is a material
  throughput hypothesis only if quality guards are held fixed.
- Add per-case incremental metrics output so long runs can be resumed and
  analyzed without waiting for final CSV/JSON.
- Cache repeated formula parsing, InChI key, formula, and fingerprint computation
  for duplicate generated SMILES within and across cases.
- Evaluate whether sorting by `(frequency, predicted-fingerprint similarity)`
  can be done without recomputing fingerprints in `build_prediction_entry`.

## P1 Base Inference Architecture

- Batch MIST encoder calls over multiple spectra instead of `batch_size=1` in the
  benchmark loop. Current code loads the dataloader with `batch_size=1`.
- Investigate vectorized multi-sample formula filtering: generate candidates for
  multiple spectra per DLM call when formulas/fingerprints can be grouped.
- Test `torch.inference_mode()` and matmul precision settings for inference.
- Test `torch.compile` only as a controlled hypothesis with warmup and comparable
  scorer timing; compile overhead must be excluded or amortized.

## P1 ICEBERG Scaling

- Reduce subprocess overhead in `_run_iceberg_batch`. It creates a new temporary
  batch directory and calls `iceberg_prediction` repeatedly per CE/instrument
  group.
- Prefer larger ICEBERG batches and fewer subprocess launches if memory permits.
- Persist ICEBERG prediction cache across scorer iterations where inputs match,
  but do not let cached results contaminate comparable timing unless explicitly
  measured as a cache-enabled mode.
- Diagnose empty hallucinated-peak traces before investing heavily in scaling
  throughput. If masking degenerates to full masking, speeding it up does not
  improve useful inference.

## P1 Training Throughput

- Fix the official Lightning batch-transfer failure for `formula: list[str]`
  before training-speed campaigns. Training smoke has shown the model can
  forward/backward manually, but the harness is not clean.
- Audit `training_step`: it uses `torch.amp.autocast('cuda', dtype=torch.float32)`
  even though the trainer config requests `precision: bf16`. This may block bf16
  speedups.
- Profile data loading and RDKit conversion inside `Collator`, especially
  SAFE-to-SMILES, formula extraction, InChI exclusion, and Morgan fingerprint
  generation.
- Test precomputed formula/fingerprint fields or an HDF5/Arrow cache for training
  rather than recomputing RDKit features in every collate call.

## P2 Quality-Preserving Improvements

- Use current partial baseline as a guard: about 13.1% Exact Top-1 and 0.476
  Tanimoto Top-1 over completed MSG base shards.
- Stratify scorer samples by MIST quality. Throughput improvements should not
  accidentally overfit to easy high-MIST examples.
- Add separate scorer modes for base inference and ICEBERG scaling; they should
  not share one primary metric until both are stable.
