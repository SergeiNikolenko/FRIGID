# FRIGID Theory Diary

This diary consolidates the confirmed observations and next-step hypotheses
for the FRIGID throughput/quality campaign.

## Confirmed observations

- `batch_size=32` on Kolmogorov produced `17.05796180665493 sec/case` on the
  smoke scorer.
- `batch_size=64` on Kolmogorov produced `17.29562647640705 sec/case` on the
  same smoke scorer.
- A locally measured bf16 inference pass was later found not to be authoritative
  because the scorer had not synced local source changes to Kolmogorov.
- After source sync was added, the bf16 path failed generation: `Categorical`
  rejected bf16 probability tensors, all formula guards failed, and the run was
  quarantined as `kolmogorov-bf16-synced-invalid`.
- The float32 diagnostic run
  `kolmogorov-dedup-diagnostics-batch32-smoke` produced
  `17.24069844186306 sec/case`, `tanimoto_top1 = 0.6561510339379311`, and
  `formula_success = 1.0`.
- The valid float32 single-GPU runs kept the quality guards green:
  - `exact_top1 = 0.0`
  - `tanimoto_top1` around `0.656`
  - `formula_success = 1.0`
- The dedup diagnostic run reported `avg_unique_valid_smiles = 73.5625`,
  `avg_duplicate_valid_smiles = 19.1875`, `avg_valid_duplicate_rate =
  0.2072613995134333`, and `avg_formula_duplicate_matches = 0.0`.
- The early dedup implementation
  `kolmogorov-early-dedup-batch32-smoke` produced
  `17.14091181755066 sec/case`, `tanimoto_top1 = 0.6561510339379311`, and
  `formula_success = 1.0`. This is a small speed improvement over the
  diagnostic run, but still close to ordinary smoke-run variance.
- The fixed-length NGBoost run
  `kolmogorov-fixed-length-sigma0-batch32-smoke` with
  `sigma_lambda = 0.0` produced `15.825038358569145 sec/case`,
  `tanimoto_top1 = 0.6496169790625572`, and `formula_success = 1.0`. This is
  the strongest single-GPU speed signal so far, but it trades away some
  Tanimoto versus the baseline guard.
- The 32-case confirmation run
  `kolmogorov-fixed-length-sigma0-batch32-32cases` produced
  `15.097882315516472 sec/case`, `tanimoto_top1 = 0.6486558560281992`, and
  `formula_success = 1.0`. The speed signal held on a larger smoke set, while
  the Tanimoto trade-off also persisted.
- The conditioning-cache run
  `kolmogorov-conditioning-cache-sigma0-batch32-32cases` produced
  `15.017504721879959 sec/case`, `tanimoto_top1 = 0.6486558560281992`, and
  `formula_success = 1.0`. This is a small but real global speed gain over the
  earlier fixed-length 32-case run, and it did not change the quality trade-off.
- The first intermediate variance run
  `kolmogorov-sigma0p5-batch32-32cases` produced
  `17.15382981300354 sec/case`, `tanimoto_top1 = 0.6809012647718191`, and
  `formula_success = 1.0`. This is a quality-improving point, but it gives back
  most of the speed gain from fixed-length decoding.
- The intermediate sweep point
  `kolmogorov-sigma0p25-batch32-32cases` produced
  `16.73140214383602 sec/case`, `tanimoto_top1 = 0.6734022311866283`,
  `exact_top1 = 0.03125`, `exact_top10 = 0.03125`, and `formula_success = 1.0`.
  This looks like the first real speed/quality compromise: it is slower than
  fixed-length `sigma_lambda = 0.0`, but materially better than the default
  smoke quality and still faster than the ordinary float32 smoke path.
- The lower sweep point
  `kolmogorov-sigma0p1-batch32-32cases` produced
  `16.67725083231926 sec/case`, `tanimoto_top1 = 0.6822530776262283`,
  `exact_top1 = 0.0`, `exact_top10 = 0.0`, and `formula_success = 1.0`.
  This improved quality over `sigma_lambda = 0.25`, but it did not recover the
  fixed-length speed signal.
- The mid sweep point
  `kolmogorov-sigma0p2-batch32-32cases` produced
  `15.791235215961933 sec/case`, `tanimoto_top1 = 0.6642945446074009`,
  `exact_top1 = 0.0`, `exact_top10 = 0.0`, and `formula_success = 1.0`.
  This is close to the fixed-length speed point, but it still trails
  `sigma_lambda = 0.0` on throughput and does not create a clear new optimum.
- `generation_time_percentage` is consistently about `99%`, so the dominant
  bottleneck is inside generation, not post-processing.
- Increasing the outer batch size alone did not move the metric materially.
- Multi-GPU is out of scope for this campaign; the optimization target is
  single-GPU behavior on a controlled remote GPU.
- The latest 3-spectrum diagnostic profile on Kolmogorov (`batch_size = 32`,
  `formula_matches = 1`, `max_attempts = 32`, `sigma_lambda = 0.0`,
  `num_tokens_unmask = 1`) produced `14.463257551193237s` total and
  `generation_time_percentage = 97.98252317270118%`. The breakdown kept
  pointing at backbone work inside generation: `model_forward_time =
  13.319700931198895s`, `sampling_step_time = 0.40032156370580196s`,
  `decode_time = 0.3904726840555668s`, `conditioning_setup_time =
  0.03468880895525217s`, and estimated padding stayed at `0.0`. Quality stayed
  green (`formula_success = 1.0`), but the case-level anatomy still showed only
  `4.333333333333333` unique valid SMILES and `31` attempts per match on
  average.
- Switching the generation entrypoints from `torch.no_grad()` to
  `torch.inference_mode()` did not produce a measurable shift on the same
  3-spectrum diagnostic profile: wall time stayed at about `14.4s`,
  `generation_time_percentage` stayed at about `98%`, and the backbone-forward
  breakdown remained the same. Treat `inference_mode` as a confirmed
  non-lever for this campaign unless a different code path is identified.
- Raising `num_tokens_unmask` to `4` on the same 3-spectrum diagnostic made
  things worse: wall time grew to about `16.0s`, generation time rose to about
  `97.4%`, model-forward time grew to about `14.5s`, `formula_success` dropped
  to `66.67%`, and `tanimoto_top1` fell to `0.4568`. Treat larger unmask counts
  as a negative signal for this path, not a free throughput win.
- `num_tokens_unmask = 3` also failed to beat the `1`-token setting on the same
  3-spectrum diagnostic. Wall time was about `14.8s`, `generation_time_percentage`
  stayed around `97.8%`, `formula_success` remained `100%`, but `tanimoto_top1`
  fell to `0.4306` and duplicate valid rate rose to about `24%`. So the
  per-step unmask knob is not the next throughput lever either.
- The latest pruning-oriented audit on the current 3-spectrum diagnostic shows
  `avg_first_unique_formula_match_at = 11.67` and
  `avg_wasted_generated_after_first_unique_formula_match = 19.0`. That is a
  direct signal that many generated candidates are still spent after the first
  useful unique formula hit inside the batch, so formula-aware pruning or
  earlier batch termination is now the next concrete hypothesis.
- The first active pruning chunk probe with `formula_pruning_chunk_size = 8`
  is not safe as-is: it cut `seconds_per_case` on the 3-spectrum diagnostic but
  dropped `formula_success` to `66.67%` and changed the sampling path enough to
  lose the control outcome on one spectrum. Treat this as a negative signal for
  naive sub-batching, not as a candidate.
- A larger `formula_pruning_chunk_size = 16` preserved `formula_success = 1.0`
  on the same 3-spectrum diagnostic and reduced `seconds_per_case` from about
  `14.6` to `10.8`, but it also pulled `tanimoto_top1` down from about `0.525`
  to `0.498`. That is the first real speed win from the pruning experiment, but
  it is not yet an output-preserving optimization.
- An intermediate `formula_pruning_chunk_size = 12` did not improve the picture.
  It stayed above the no-pruning control on throughput, but it was slower than
  `16` and also slightly worse on `tanimoto_top1`. Treat `16` as the current
  best pruning candidate and `12` as a dominated point.
- The 32-case comparison on the same current code path sharpened the picture:
  `formula_pruning_chunk_size = 16` finished in `105.0s` total versus
  `152.8s` for the no-pruning control, which is a real throughput gain. The
  trade-off is still present, though: `tanimoto_top1` dropped from `0.4742` to
  `0.4682`, while `formula_success` stayed at `1.0`. So pruning is now a valid
  speed candidate, but not yet a clean chemistry-preserving improvement.
- The 32-case `num_tokens_unmask = 2` run was a negative result. It took
  `154.9s` versus `152.8s` for the current control, `formula_success` fell to
  `96.88%`, and `tanimoto_top1` dropped to `0.4552`. Treat larger unmask counts
  as closed for this path unless a new model change changes the picture.
- Batched MIST encoding at `encoder_batch_size = 8` was also a negative
  result. The 32-case run finished in `154.7s` versus `152.8s` for control,
  with the same `tanimoto_top1 = 0.4742` and `formula_success = 1.0`. So the
  encoder was not the bottleneck we hoped to expose; the generation path still
  dominates.
- A larger pruning point, `formula_pruning_chunk_size = 24`, gave the best
  overall compromise so far on the current code path. It finished in `132.4s`
  total versus `152.8s` for the control, while improving `tanimoto_top1` to
  `0.4975` and keeping `formula_success = 0.9375`. This is a better Pareto
  point than `16` if we care about chemistry quality, but it is still not fully
  guard-clean because one case never matched.

## High-priority theories

### 1. Inference cache inside the decoder path

Hypothesis: cache shared conditioning state, prefix state, or decode state
inside `src/dlm/sampler.py` / `src/dlm/model.py` so repeated work is not
recomputed for every candidate in a case.

Why it matters: the measured bottleneck is still almost entirely inside the
generation call, and batch scaling already stalled.

Expected effect: material reduction in `seconds_per_case` without changing the
quality guards if the candidate set stays stable.

First check: compare wall time and candidate overlap between baseline and a
cached-path smoke run on a fixed set of cases.

### 2. Dedup before validation and ranking

Hypothesis: remove duplicate molecules earlier in the candidate path so the
same structure is not scored, validated, or reranked repeatedly.

Why it matters: repeated sampling can waste a lot of time when the search keeps
returning the same structures.

Expected effect: modest throughput gain, stronger on cases with a high duplicate
rate.

First check: measure unique-vs-raw candidate ratios and replay scoring on the
unique set.

Status: implemented as an early canonical-SMILES skip in
`generate_with_formula_filter`. The scorer kept quality guards green and moved
from `17.24069844186306` to `17.14091181755066 sec/case` versus the diagnostic
run, but `generation_time_percentage` stayed at about `99.1%`. Treat this as a
minor cleanup, not the main optimization path.

### 3. ICEBERG worker persistence

Hypothesis: keep ICEBERG-style scoring warm between requests instead of paying
subprocess startup and reload cost for every batch.

Why it matters: the scaling path is still expected to depend on repeated
orchestration overhead.

Expected effect: large gain for ICEBERG-heavy runs; neutral to quality if state
does not leak between cases.

First check: profile cold-start vs persistent-worker timings for the same
request set.

### 4. Formula-aware pruning

Hypothesis: reject branches earlier when they cannot still reach the target
formula.

Why it matters: formula success is already perfect, so we should look for waste
before the final formula check.

Expected effect: less wasted decoding work, with quality preserved if the
pruning logic is exact.

First check: logging-only audit of how many branches could be pruned without
changing outputs.

### 5. BF16 inference path

Hypothesis: the inference path is still running with float32 autocast inside the
model forward path and can be moved to bf16 for eval-only runs.

Why it matters: this is a direct way to make the GPU path cheaper without
changing the outer scorer contract.

Status: rejected as currently unsafe. The synced run produced invalid sampling
probabilities and failed the quality guards.

Next check: only revisit if probability normalization/sampling is made explicitly
float32 around `Categorical`.

### 6. Case-level scheduling

Hypothesis: use a smarter per-case scheduler instead of only increasing the
outer batch size.

Why it matters: batch64 already showed that bigger batches alone do not solve
the bottleneck.

Expected effect: better GPU utilization on heterogeneous cases.

First check: trace-based estimate of padding/idle waste on the current decode
mix.

### 7. Intermediate length variance sweep

Hypothesis: `sigma_lambda = 0.0` is too aggressive but shows that length
variance/padding is a real speed lever. Intermediate values may recover some
Tanimoto while keeping most of the speed gain.

Why it matters: the 32-case confirmation run improved speed from roughly
`17.1 sec/case` to `15.1 sec/case`, but `tanimoto_top1` dropped from about
`0.656` to `0.6487`.

Expected effect: identify a better speed/quality trade-off than both default
`sigma_lambda = 3.0` and fixed-length `sigma_lambda = 0.0`.

First check: run single-GPU smoke sweeps for lower intermediate values such as
`sigma_lambda = 0.25`, because `0.5` recovered quality but mostly lost the
speed gain.

## Current priority order

1. Decide the unvalidated `extended_attention_mask` cache by completing a
   comparable 32-case scorer run or explicitly leaving it unpromoted.
2. Add a 2-3 case micro-profiler for the generation hot path with CUDA
   synchronization.
3. Add per-case generation anatomy: attempts, valid candidates, unique valid
   candidates, duplicates, formula matches, stop reason, generated lengths,
   padding estimate, and wall time.
4. Run formula-aware pruning as a logging-only audit before changing outputs.
5. Audit length and padding waste, then test length grouping or exact pruning
   only if the profiler supports it.
6. Keep ICEBERG worker persistence as a separate scaling-path campaign after
   the base generator bottleneck is understood.
7. Do not spend more scorer slots on batch-size-only, multi-GPU, or naive bf16
   changes in this campaign.

## Resume note from 2026-06-18

The latest valid speed point is
`kolmogorov-conditioning-cache-sigma0-batch32-32cases` with
`15.017504721879959 sec/case`, `tanimoto_top1 = 0.6486558560281992`, and
`formula_success = 1.0`. Compare future speed candidates against this result.

The `extended_attention_mask` cache is not validated. Its scorer run was
interrupted before completing 32 cases, so it must not be treated as an accepted
improvement.

The next useful work is not another sigma sweep. The campaign needs a
generation micro-profile and per-case anatomy first, then a scored patch chosen
from measured hot paths.

Profiler instrumentation has been added behind an opt-in flag:
`scripts/benchmark_spec2mol.py --profile-generation`, or
`FRIGID_SCORER_PROFILE_GENERATION=1` through the scorer. The scorer also syncs
`src/dlm/sampler.py` and `src/dlm/utils/spec2mol.py`, so remote runs use the
current hot-path code. Treat profiler runs as diagnostic because CUDA
synchronization changes timing overhead.

The first diagnostic probe
`kolmogorov-profile-generation-probe` used 2 spectra on Kolmogorov with
`batch_size = 32`, `formula_matches = 1`, `max_attempts = 32`, and
`sigma_lambda = 0.0`. It is not a comparable candidate. It showed
`generation_time_percentage = 97.83771326722832`, with
`total_generation_profile_model_forward_time = 8.801494488492608`,
`total_generation_profile_sampling_step_time = 0.24337605107575655`,
`total_generation_profile_decode_time = 0.23280819039791822`, and
`total_generation_profile_conditioning_setup_time = 0.01683588419109583`.
This points the next optimization at reducing model forward work across
diffusion steps, not RDKit post-processing or conditioning setup.

The synchronized follow-up probe
`kolmogorov-profile-generation-breakdown-probe-v2` fixed the forward timing
measurement and confirmed the bottleneck more precisely. On 1 spectrum it
showed `generation_profile_model_forward_time = 4.455873496830463`, of which
`generation_profile_model_forward_backbone_time = 4.403329693712294`.
`generation_profile_model_forward_build_embeddings_time`,
`generation_profile_model_forward_formula_conditioning_time`, and
`generation_profile_model_forward_fingerprint_conditioning_time` were all tiny.
So the next mechanism is not conditioning prep; it is the backbone encoder
work executed on every diffusion step.

The code now exposes an opt-in `num_tokens_unmask` knob through the scorer and
benchmark path. This is the next throughput hypothesis to test because it can
reduce the number of backbone passes per spectrum without changing the model
itself. A fresh probe with `num_tokens_unmask = 2` is already in
`.autoresearch/iterations/kolmogorov-num-tokens-unmask-2-probe/`, but it
regressed versus the earlier `num_tokens_unmask = 1` profile on both
throughput and quality. Treat `2` as a negative result, not a candidate.

The same generation profiles also show no useful length variance yet: the
stored `predicted_token_length` is flat at `120` and `estimated_padding_tokens`
is `0`. That means length grouping is not evidence-backed on the current
sample, so the next mechanism should stay focused on the inner generation path
or exact pruning rather than outer scheduling.

The 32-case aggregate gives a concrete formula waste signal too:
`avg_total_valid = 87.09375` and `avg_formula_matches = 7.53125`, which is
about an `8.6%` formula-hit rate among valid generations. That is the first
hard number that makes a logging-only formula-aware pruning audit worth doing
before another scheduling experiment.

The local helper `scripts/audit_formula_waste.py` now turns that into a stable
snapshot from `detailed_results.csv`. On the current 32-case run it reports
`avg_formula_match_fraction_among_valid = 10.37%` and
`avg_unique_valid_fraction_among_valid = 80.84%`, which is the cleaner summary
to carry into the next remote audit.

The current workstation cannot reach the remote GPU hosts right now. `ssh`
attempts to `kolmogorov` fail on DNS resolution, and `spectrum`, `faro`, and
`bern` fail with `Operation not permitted` on both the normal SSH port and
`46522`. The next remote audit step should only be retried after the network
path changes.

## Notes from ideation threads

- The local ideation thread suggested the same direction: shared encoder/KV
  cache, dedup, formula-aware pruning, persistent ICEBERG workers, cheap-first
  validation, and smarter case scheduling.
- The stronger model pass reinforced the same priorities and did not surface a
  contradictory mechanism.
- The synced bf16 smoke run showed that naive eval-only bf16 is invalid for this
  sampler path.
- The new sigma sweep points suggest a monotonic trade-off: lower sigma values
  improve speed toward fixed-length decoding, but the only point that clearly
  changes the frontier is `sigma_lambda = 0.0`; intermediate values do not beat
  it on throughput and do not create a better quality/speed Pareto point yet.
- The conditioning cache removed a repeated decode-step setup from the hot
  path and produced a small but measurable global speedup on the 32-case fixed
  length benchmark. That makes it a valid optimization, but not the main lever
  for the remaining generation bottleneck.
