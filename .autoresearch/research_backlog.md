# Research Backlog

## Current Resume Priority

- Do not use multi-GPU for this campaign. The current target is global
  single-GPU throughput measured as `seconds_per_case`.
- Use `kolmogorov-conditioning-cache-sigma0-batch32-32cases` as the latest
  valid speed reference: `15.017504721879959 sec/case`,
  `tanimoto_top1 = 0.6486558560281992`, `formula_success = 1.0`.
- Treat the `extended_attention_mask` cache as unvalidated until a complete
  comparable 32-case scorer run finishes.
- Before another scored optimization, add generation micro-profiling and
  per-case anatomy so the next patch targets a measured hot path.
- Do not spend more comparable scorer slots on batch-size-only, sigma-only,
  multi-GPU, or naive bf16 changes unless new profiling evidence justifies it.
- The `num_tokens_unmask = 2` probe is a negative signal relative to the
  earlier `num_tokens_unmask = 1` micro-profile. Do not assume higher unmask
  counts are a free throughput gain; revisit only if a larger-profile diagnostic
  changes the picture.
- The current generation probes all predict token length `120` with zero
  estimated padding, so length grouping is not yet evidence-backed. Do not burn
  a scorer slot on outer length scheduling until a mixed-length sample proves
  there is real padding waste to remove.
- The 32-case aggregate shows only about `8.6%` of valid generations hit the
  target formula (`7.53125` formula matches out of `87.09375` valid samples on
  average). That is enough to justify a logging-only formula-aware pruning
  audit before any further outer scheduling idea.
- The pruning experiments now have a clearer split:
  `formula_pruning_chunk_size = 8` is a negative result because it reduced
  wall time but also dropped `formula_match_success_rate` to `66.67%` on the
  3-spectrum diagnostic. `formula_pruning_chunk_size = 16` is the first real
  speed candidate because it kept `formula_match_success_rate = 1.0` on the
  same diagnostic and reduced wall time to about `10.8s`, but it also pulled
  `tanimoto_top1` down from about `0.525` to `0.498`. Do not promote either as
  a final solution yet.
- `formula_pruning_chunk_size = 12` did not find a better trade-off. It was
  slower than `16` and slightly worse on `tanimoto_top1`, so the pruning sweep
  should not keep probing the middle blindly.
- On the current 32-case comparison, `formula_pruning_chunk_size = 16` beat the
  no-pruning control on throughput (`105.0s` vs `152.8s`) while keeping
  `formula_success = 1.0`. The quality trade-off is mild but real:
  `tanimoto_top1` went from `0.4742` to `0.4682`. Keep `16` as the best fixed
  pruning candidate, but do not present it as quality-neutral.
- `num_tokens_unmask = 2` on the same 32-case path is also a negative result:
  slower than control, `formula_success = 96.88%`, and `tanimoto_top1 = 0.4552`.
  Do not spend another scorer slot on larger unmask counts for this branch.
- `encoder_batch_size = 8` did not help either. The 32-case run was slightly
  slower than control (`154.7s` vs `152.8s`) and kept the same quality numbers,
  so MIST batching is not the next likely speed lever in this exact path.
- `formula_pruning_chunk_size = 24` is now the best compromise point on the
  current code path: `132.4s` total, `tanimoto_top1 = 0.4975`, and
  `formula_success = 93.75%`. It beats the control on both throughput and
  Tanimoto, but it still loses one case on formula success, so it is not a
  final answer yet.
- `formula_pruning_chunk_size = 28` is dominated by `24`: slower, lower
  `tanimoto_top1 = 0.4590`, and no better success than `24`. Do not probe
  fixed-size chunks above `24` on this path.
- The pruning frontier is now fixed-size bounded: `24` is the best fixed chunk
  so far, and `28` adds no value. Any further pruning work on this branch
  should be adaptive or should be replaced by a different generation-side
  mechanism. The reconciler also marked the `24` run as `quarantine` because
  its `tanimoto_top1` regressed too far versus the baseline guard, so there is
  no accepted supervisor candidate yet.
- The next pruning hypothesis should be adaptive, not another fixed sweep.
  Use the existing diagnostic runs as evidence: `8` reached the first unique
  formula match earlier but needed more pruning batches, `16` reduced batches
  and preserved formula success, and `12` was intermediate. A sensible next
  check is a stop rule keyed to first unique formula hit plus a small tail,
  measured against the same 32-case guard set.
- Use `scripts/audit_formula_waste.py` on every meaningful scorer run so formula
  waste is captured from `detailed_results.csv` immediately instead of being
  recomputed manually.
- Remote execution is currently usable via direct `ssh kolmogorov`, but
  `rsync`/`scp` are flaky from this workstation and may fail DNS resolution.
  Prefer direct SSH commands for runs and avoid assuming the file-transfer
  helpers are reliable until the route stabilizes.

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

- Profile the current accepted generation path. The profile should separate
  conditioning setup, backbone forward calls, masking/sampling, formula
  filtering, RDKit validation, fingerprint scoring, and result assembly with
  CUDA synchronization around GPU regions.
- The latest 3-spectrum diagnostic profile already shows the backbone encoder
  dominates the generation path (`13.32s` of `14.17s` generation time) while
  conditioning, decode, sampling, and padding are tiny. The next patch should
  target backbone-call reduction or decode-step reduction, not padding or
  length grouping.
- The `torch.inference_mode()` swap was neutral on the same 3-spectrum
  diagnostic profile. Do not spend another scorer slot on `no_grad` vs
  `inference_mode`; the bottleneck is elsewhere.
- `num_tokens_unmask = 4` was a negative signal on the same 3-spectrum
  diagnostic profile: slower wall time, worse Tanimoto, and formula success
  dropped below 100%. Do not treat larger unmask counts as a free reduction in
  backbone work.
- `num_tokens_unmask = 3` also failed to improve on the `1`-token setting:
  speed was not better, Tanimoto worsened, and the duplicate-valid rate rose.
  Keep the step-loop unmask knob closed for now.
- The current pruning audit shows `avg_first_unique_formula_match_at = 11.67`
  and `avg_wasted_generated_after_first_unique_formula_match = 19.0` on the
  recent 3-spectrum diagnostic. That is the next meaningful signal to use for a
  formula-aware pruning or earlier batch-stop audit.
- The new pruning chunking result says the next experiment should not be another
  blind speed sweep. Either recover the lost Tanimoto with a smarter stop rule or
  abandon this branch and move to a backbone-call reduction idea.
- Because `12` is dominated, the next pruning attempt should be adaptive rather
  than another fixed chunk-size point. If no adaptive rule is available, stop
  spending scorer slots on this branch.
- If pruning is continued, the next experiment should try to preserve
  Tanimoto while retaining the 32-case speed gain from `16`. Otherwise, move
  the next scorer slot to a different hot path instead of squeezing fixed chunk
  sizes further.
- Because the step-count knob is now closed, the next measurable lever should
  come from a different mechanism: adaptive pruning, backbone-call reduction,
  or a new cache path.
- Because encoder batching did not move the metric, the next lever should stay
  on the generation/backbone side rather than more encoder batching.
- Since `24` improved both speed and Tanimoto, the next pruning action should
  be to recover the missing formula success case rather than continuing a blind
  fixed-size sweep. If that is not practical, keep `24` as the best known
  frontier and move to a different hot path.
- Fixed-size pruning is now bounded: `24` is the winner, `28` is dominated.
  Any further work in this branch should be adaptive, not another fixed chunk
  interpolation.
- Add per-case anatomy output: attempts, valid candidates, unique valid
  candidates, duplicate candidates, formula matches, stop reason, generated
  lengths, padding estimate, and wall time.
- Validate or discard the `extended_attention_mask` cache with a full comparable
  32-case run before stacking additional sampler changes on top of it.
- Profile `generate_with_formula_filter`: separate DLM generation time from RDKit
  validation, formula calculation, InChI key generation, Morgan fingerprint, and
  CSV/result assembly.
- Audit formula-aware pruning next if a generation-side change is needed; the
  current length data does not justify a padding-scheduling detour.
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
