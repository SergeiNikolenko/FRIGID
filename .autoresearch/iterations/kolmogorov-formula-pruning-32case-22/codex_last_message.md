Adaptive interpolation run with `formula_pruning_chunk_size=22` on Kolmogorov.

Result: `seconds_per_case=3.834425263106823`, `tanimoto_top1=0.5015192916616797`,
`formula_success=1.0`.

This point is better than the 24-chunk run on both speed and Tanimoto, but it
still misses the guard band by a wide margin. Keep it as the best compromise
within the pruning sweep so far, but do not promote it as a final answer.
