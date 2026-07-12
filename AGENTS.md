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
