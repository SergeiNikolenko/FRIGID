# FRIGID Agent Instructions

## Compact MSG evaluation

- Treat `configs/benchmarks/msg_compact_*_v1.tsv` as locked manifests. Do not
  edit them or tune model parameters against their outcomes.
- Use 16-32 spectra for runtime smoke tests and the existing diverse development
  panel for exploratory tuning.
- Freeze the candidate before compact evaluation.
- Use micro128 only as a futility gate. A positive micro128 result is not enough
  to promote a candidate.
- Require concordant micro256 and molecule-disjoint macro64 results for compact
  promotion. Use micro512 only for a predeclared borderline decision.
- Compare micro panels with molecule-cluster bootstrap using
  `scripts/compare_paired_benchmark_runs.py --bootstrap-unit molecule`.
- Record manifest hashes, code revision, checkpoint hashes, seed, generation
  budget, and complete paired outputs for every run.
- Run the locked 1,024-spectrum gate and then the full benchmark only for compact
  winners. Compact panels never replace final evidence.
- Regenerate panels only with `scripts/build_msg_compact_benchmark.py`; retain
  its JSON quality report and stop if any acceptance or overlap check fails.

## Experiment orchestration

- Read `docs/FRIGID_MODEL_STATUS.md` before proposing or launching an
  architecture experiment. Do not repeat a rejected branch without a materially
  different hypothesis that addresses the recorded failure.
- Distinguish `audit`, `prepared`, `running`, `rejected`, and `confirmed`.
  Repository integration or checkpoint inspection is not benchmark evidence.
- Keep one Linear issue per promoted hypothesis. Record the exact host, worktree,
  branch, commit, manifest hash, checkpoint hash, seed, command, run directory,
  and decision. Close rejected and completed issues; create a new issue for the
  next gate rather than silently extending scope.
- Use the local machine only for orchestration, documentation, lightweight
  validation, and paired analysis. Run model inference and training on a
  configured remote host in a persistent session.
- Never update a dirty server checkout in place. Fetch the required commit into
  a dedicated clean worktree and preserve active run directories and user edits.
- A candidate may enter the production union only after target-blind paired
  evaluation. Target labels may be used for metrics and molecule-disjointness,
  never for generation, selection, or ranking.
- Prefer complementary candidate recall over replacing the complete pipeline.
  The confirmed reference is DLM control + DLM temperature 0.8 + train-only
  retrieval + MolForge 0.172.

## Architecture stop rules

- Do not retry frozen, calibrated, distilled, blended, residual, or full-finetuned
  DreaMS fingerprint heads as direct MIST replacements. All failed their gates.
- Do not promote NGBoost as a quality improvement at a matched generation
  budget. It is a speed option; no-NGBoost DLM is the quality reference.
- Keep temperature 0.8 as a complementary candidate source, not a standalone
  ranking winner.
- Do not claim ICEBERG, DiffMS, MBGen, DualLGD, GEMS-style refinement, or
  selective TTT as improvements until an identical-subset paired gate passes.
- Do not run MBGen without public compatible weights and a complete loader.
- Do not adapt DualLGD until a train-only test proves a valid Morgan-4096 to
  Morgan-2048 interface without target leakage.
- Treat selective TTT as prepared only. It requires train-neighbor provenance,
  a frozen selection policy, and a paired compact result before promotion.
