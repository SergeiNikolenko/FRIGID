# FRIGID Autoresearch Continuation Brief

Date: 2026-06-18

This file is the handoff point for resuming the local Autoresearch/Codex goal
for FRIGID. The campaign is not complete. It is paused at a safe boundary.

## Operating Mode

- Keep Codex and the Autoresearch supervisor local.
- Run heavy FRIGID work over SSH on remote GPU hosts.
- Primary active host for the latest speed campaign: `ssh:kolmogorov`.
- Do not use multi-GPU for this campaign. The target is global single-GPU
  throughput, not sharding across devices.
- Optimize `seconds_per_case` as the primary metric.
- Track `exact_top1`, `tanimoto_top1`, and `formula_success` as quality guards.
- Do not treat a speed-only result as a chemistry-quality improvement.

## Dashboard And State

- Dashboard URL when the local server is running: `http://127.0.0.1:8765/`.
- Render command:

```bash
python3 /Users/nikolenko/.agents/plugins/autoresearch/scripts/autoresearch.py render-dashboard --repo /Users/nikolenko/.codex/worktrees/f017/FRIGID
```

- Source of truth:
  - `.autoresearch/state.json`
  - `.autoresearch/results.tsv`
  - `.autoresearch/research_backlog.md`
  - `.autoresearch/theory_diary.md`
  - `.autoresearch/iterations/*`

## Current Best Valid Speed Point

Best observed valid candidate:

- Iteration: `kolmogorov-conditioning-cache-sigma0-batch32-32cases`
- `seconds_per_case`: `15.017504721879959`
- `elapsed_time_seconds`: `480.5601511001587`
- `exact_top1`: `0.0`
- `tanimoto_top1`: `0.6486558560281992`
- `tanimoto_top10`: `0.6531608942896128`
- `formula_success`: `1.0`
- `avg_unique_valid_smiles`: `70.25`
- `avg_duplicate_valid_smiles`: `16.84375`
- `avg_valid_duplicate_rate`: `0.19158275187727353`
- `generation_time_percentage`: `99.13670617366856`
- Artifacts:
  `.autoresearch/iterations/kolmogorov-conditioning-cache-sigma0-batch32-32cases/`

Interpretation: the conditioning cache is a valid global speed improvement over
the fixed-length `sigma_lambda=0.0` 32-case run, but the gain is small. The main
bottleneck is still the generation path.

## Key Valid Results

| Iteration | Cases | sec/case | exact_top1 | tanimoto_top1 | formula_success | Decision |
|---|---:|---:|---:|---:|---:|---|
| `kolmogorov-single-gpu-baseline` | smoke | `18.50810819864273` | `0.0` | `0.6610685214400291` | `1.0` | baseline |
| `kolmogorov-batch32-smoke` | smoke | `17.05796180665493` | `0.0` | `0.6561510339379311` | `1.0` | candidate |
| `kolmogorov-batch64-smoke` | smoke | `17.29562647640705` | `0.0` | `0.656480857121758` | `1.0` | candidate |
| `kolmogorov-dedup-diagnostics-batch32-smoke` | 16 | `17.24069844186306` | `0.0` | `0.6561510339379311` | `1.0` | diagnostic |
| `kolmogorov-early-dedup-batch32-smoke` | 16 | `17.14091181755066` | `0.0` | `0.6561510339379311` | `1.0` | candidate |
| `kolmogorov-fixed-length-sigma0-batch32-smoke` | 16 | `15.825038358569145` | `0.0` | `0.6496169790625572` | `1.0` | candidate |
| `kolmogorov-fixed-length-sigma0-batch32-32cases` | 32 | `15.097882315516472` | `0.0` | `0.6486558560281992` | `1.0` | candidate |
| `kolmogorov-sigma0p5-batch32-32cases` | 32 | `17.15382981300354` | `0.0` | `0.6809012647718191` | `1.0` | candidate |
| `kolmogorov-sigma0p25-batch32-32cases` | 32 | `16.73140214383602` | `0.03125` | `0.6734022311866283` | `1.0` | candidate |
| `kolmogorov-sigma0p1-batch32-32cases` | 32 | `16.67725083231926` | `0.0` | `0.6822530776262283` | `1.0` | candidate |
| `kolmogorov-sigma0p2-batch32-32cases` | 32 | `15.791235215961933` | `0.0` | `0.6642945446074009` | `1.0` | candidate |
| `kolmogorov-conditioning-cache-sigma0-batch32-16cases` | 16 | `15.811927944421768` | `0.0` | `0.6496169790625572` | `1.0` | candidate |
| `kolmogorov-conditioning-cache-sigma0-batch32-32cases` | 32 | `15.017504721879959` | `0.0` | `0.6486558560281992` | `1.0` | candidate |

## Invalid Or Non-Authoritative Results

- `kolmogorov-bf16-batch32-smoke` is not authoritative because the scorer had not
  yet synced the local source changes to Kolmogorov.
- `kolmogorov-bf16-synced-invalid` is invalid. It produced very fast timing but
  failed sampling/guard checks because `Categorical` rejected bf16 probability
  tensors.
- The `extended_attention_mask` cache is not validated. Its 32-case run was
  interrupted at about 21/32 spectra. Do not claim it as an improvement until a
  complete comparable scorer run finishes.

## Code State To Remember

Accepted/useful changes in the current worktree:

- The scorer syncs local benchmark/model/sampler code to the remote checkout
  before running.
- `generate_with_formula_filter` has early canonical SMILES duplicate skipping
  and aggregate duplicate diagnostics.
- `scripts/benchmark_spec2mol.py` reports duplicate diagnostics.
- `DLM.forward` supports precomputed formula/fingerprint conditioning state.
- `Sampler.generate` and `Spec2MolSampler.generate` precompute conditioning
  embeddings/masks once per generation call.

Unvalidated change:

- `extended_attention_mask` can be precomputed and passed into the model, but
  the comparable scorer run was interrupted before completion.

New diagnostic support:

- `scripts/benchmark_spec2mol.py` accepts `--profile-generation`.
- `.autoresearch/scorers/frigid_speed_quality_scorer.sh` passes that flag when
  `FRIGID_SCORER_PROFILE_GENERATION=1`.
- The scorer now syncs `src/dlm/sampler.py` and `src/dlm/utils/spec2mol.py`
  in addition to the benchmark script, `model.py`, and `benchmark_utils.py`.
- Profile mode is diagnostic-only. It records per-case generation anatomy and
  sampler timing fields; it should not be treated as a comparable speed
  candidate because CUDA synchronization adds measurement overhead.
- First diagnostic artifact:
  `.autoresearch/iterations/kolmogorov-profile-generation-probe/`.
  On 2 spectra, model forward dominated the profiled generation time:
  `8.801494488492608s` model forward versus `0.24337605107575655s` sampling,
  `0.23280819039791822s` decode, and `0.01683588419109583s` conditioning setup.
- Follow-up diagnostic artifact:
  `.autoresearch/iterations/kolmogorov-profile-generation-breakdown-probe-v2/`.
  This synchronized forward breakdown showed
  `generation_profile_model_forward_time = 4.455873496830463` and
  `generation_profile_model_forward_backbone_time = 4.403329693712294` on 1
  spectrum, while conditioning prep stayed tiny. The next bottleneck to attack
  is the backbone encoder work inside each diffusion step.
- The next ablation knob is `num_tokens_unmask`.
  It is now wired through the scorer and benchmark path as an opt-in setting.
  The hypothesis is that unmasking more than one token per step can reduce the
  number of backbone passes per spectrum. A one-spectrum probe with
  `num_tokens_unmask = 2` is already stored in
  `.autoresearch/iterations/kolmogorov-num-tokens-unmask-2-probe/`, but it did
  not beat the earlier `num_tokens_unmask = 1` micro-profile: `5.055677652359009
  sec/case` versus `4.7610132694244385 sec/case`, with lower `tanimoto_top1`
  and fewer unique valid molecules. Treat `2` as a negative signal, not a
  promotion.
- The current profiled probes do not show meaningful length variance yet:
  predicted token length is flat at `120` and estimated padding is `0` in the
  stored generation profiles. Do not spend the next scorer slot on length
  grouping until a real mixed-length sample shows padding waste.
- The current 32-case aggregate also shows formula waste directly:
  `avg_total_valid = 87.09375` while `avg_formula_matches = 7.53125`, so only
  about `8.6%` of valid generations hit the target formula. That is the first
  concrete reason to keep formula-aware pruning as the next audit target.
- Local audit helper:

```bash
python3 scripts/audit_formula_waste.py \
  --input .autoresearch/iterations/kolmogorov-conditioning-cache-sigma0-batch32-32cases/detailed_results.csv \
  --output-json .autoresearch/iterations/kolmogorov-formula-waste-audit-local/audit.json \
  --output-md .autoresearch/iterations/kolmogorov-formula-waste-audit-local/audit.md
```

The current local snapshot from that helper reports:
`avg_total_valid = 87.09375`, `avg_formula_matches = 7.53125`,
`avg_formula_match_fraction_among_valid = 10.37%`, and
`avg_unique_valid_fraction_among_valid = 80.84%`.
- Remote access status on this workstation is currently blocked:
  `ssh kolmogorov`, `ssh spectrum`, `ssh faro`, and `ssh bern` all fail from the
  local environment, with `kolmogorov` failing DNS resolution and the jump
  hosts failing with `Operation not permitted` on port `22` and `46522`. Do not
  keep retrying the same route until the network/VPN path changes.

Do not revert unrelated dirty files, especially pre-existing changes in
`src/dlm/iceberg_sampler.py`, unless explicitly requested.

## Current Interpretation

- The dominant bottleneck is still generation: about 99 percent of wall time.
- Larger outer batch size alone is not the main lever.
- Early duplicate skipping is useful cleanup, but it does not move the main
  bottleneck enough.
- `sigma_lambda=0.0` is the best speed point, with a Tanimoto trade-off.
- `sigma_lambda=0.1` and `0.5` improve Tanimoto but are slower.
- `sigma_lambda=0.2` and `0.25` are intermediate but do not create a clear new
  Pareto point.
- Conditioning-cache is a real global speedup, but small.
- Further speed work should be based on generation profiling, not another blind
  sigma sweep.

## Resume Plan

1. Decide the unvalidated `extended_attention_mask` cache.
   - Either finish the interrupted 32-case `sigma_lambda=0.0` scorer run.
   - Or explicitly leave it as an unpromoted micro-optimization.
   - Compare against `15.017504721879959 sec/case`, not the older
     `15.097882315516472 sec/case`.

2. Add a tiny generation micro-profiler.
   - Profile 2-3 representative cases with CUDA synchronization.
   - Split time inside generation into conditioning, backbone forward calls,
     masking/sampling, formula filtering, RDKit validation, fingerprint scoring,
     and result assembly.
   - Keep this as a diagnostic run unless the scorer contract says otherwise.
   - Use `FRIGID_SCORER_PROFILE_GENERATION=1` on the scorer for this diagnostic.

3. Add per-case generation anatomy.
   - Record attempts, valid candidates, unique valid candidates, duplicate
     candidates, formula matches, stop reason, generated lengths, padding
     estimate, and wall time per case.
   - Use this to decide whether the next mechanism is pruning, length grouping,
     or sampler-side restructuring.

4. Run formula-aware pruning as a logging-only audit first.
   - Estimate how often partial generations could be rejected without changing
     final outputs.
   - Do not make pruning active until the audit proves it would preserve
     candidate sets.

5. Audit length and padding waste.
   - Use the `sigma_lambda` results as evidence that length variance matters.
   - Measure whether cases with similar predicted length can be grouped to
     reduce wasted decode work on one GPU.

6. Only after the profiling/audit results, choose the next scored patch.
   - Expected candidates: exact formula-aware pruning, length grouping, or a
     deeper sampler/backbone call reduction.
   - Avoid more batch-size-only, multi-GPU, or bf16-only attempts for now.

7. Keep chemistry-quality work separate.
   - For quality improvement, start a separate scorer mode with oracle@K,
     candidate-bank/reranker checks, fixed-fingerprint oracle, and ICEBERG trace
     diagnostics.
   - Do not select throughput candidates by test quality.

## Suggested Fresh Goal Prompt

Use this when resuming the campaign:

```text
Continue the FRIGID Autoresearch throughput campaign from
docs/FRIGID_AUTORESEARCH_CONTINUATION.md and .autoresearch/state.json.
Keep the supervisor local, run heavy work over ssh:kolmogorov, do not use
multi-GPU, and optimize global single-GPU seconds_per_case with exact_top1,
tanimoto_top1, and formula_success as quality guards. First resolve the
unvalidated extended_attention_mask cache or skip it explicitly, then add
generation micro-profiling and choose the next scored patch from measured hot
paths. Do not repeat sigma-only or batch-size-only sweeps.
```
