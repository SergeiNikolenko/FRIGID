Adaptive interpolation run with `formula_pruning_chunk_size=20` on Kolmogorov.

Result: `seconds_per_case=3.532234065234661`, `tanimoto_top1=0.4688765509054065`,
`formula_success=1.0`.

This is the fastest fixed pruning point so far, but the quality guard still fails
because Tanimoto regressed too far versus the baseline. Keep it as a quarantine
signal, not a promoted candidate.
