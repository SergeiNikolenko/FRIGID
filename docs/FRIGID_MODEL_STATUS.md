# FRIGID Model Status

This document is the architecture decision registry. Update it when a branch
changes state and link the corresponding run evidence and Linear issue.

## Confirmed quality path

| Component | State | Evidence | Decision |
| --- | --- | --- | --- |
| DLM control, 100 attempts | confirmed baseline | 1,024 molecule-diverse: Tanimoto@10 `0.4805`, Exact@10 `0.1650` | Quality reference. |
| DLM temperature 0.8, 200 attempts | confirmed candidate source | Compact Exact@10 `+0.0469` on micro128 and macro64; Tanimoto intervals include zero | Keep inside union, not as standalone ranker. |
| Train-only retrieval | confirmed candidate source | Positive union gain at 64, 200, and 1,024 | Keep inside union. |
| MolForge 0.172 | confirmed candidate source | At 1,024, adds Tanimoto@10 `+0.0178`, CI `[+0.0129, +0.0231]`; Exact@10 `+0.0186`, CI `[+0.0107, +0.0273]` | Keep inside union. |
| Four-source union | confirmed leader | 1,024 Tanimoto@10 `0.5277`, Exact@10 `0.2119`; compact micro256 Tanimoto@10 `+0.0796` CI `[+0.0566, +0.1051]`; macro64 `+0.0428` CI `[+0.0207, +0.0678]` | Promote to locked 1,024 gate, then full. |
| Full four-source confirmation | running | Spectrum preflight job `132` passed; jobs `133-168` cover both frozen DLM sources in 36 disjoint shards at commit `e1b18a9`. Exact matrix train-only retrieval job `169` is dependency-gated behind all DLM jobs at commit `733c588`. The preserved MolForge prefix covers `8,630/17,082` exactly; suffix job `170` covers the remaining `8,452` after job `169`, using clean FRIGID `ad8cf6a` and MolForge `2e5f37c` worktrees. Hash-gated full merge/fusion/bootstrap finalization is prepared at `01a98d6`. The pinned MCES runtime, sharded evaluator, and strict full merge with molecule-cluster bootstrap are prepared at `183fdd2`/`1def8b6`/`35ffd06`; no finalization or MCES job is queued before source validation | Validate every source at `17,082/17,082`, freeze exact shard-list hashes, run target-blind finalization, then smoke and scale the pinned thresholded-MCES evaluator. |

## Rejected or bounded branches

| Architecture | State | Result or blocker | Next valid action |
| --- | --- | --- | --- |
| DreaMS frozen fingerprint head | rejected | Tanimoto `~0.124` vs MIST `~0.542` | Do not repeat. |
| DreaMS calibrated/loss-tuned head | rejected | Best `~0.234` | Do not repeat. |
| DreaMS distillation | rejected | Best `~0.240` | Do not repeat. |
| DreaMS full fine-tune | rejected | Train `~0.84`, validation `~0.258`; severe overfit | Only revisit with a different task, such as retrieval side information. |
| MIST + DreaMS residual adapter | rejected | Gain `+0.00068`, below `+0.005` gate | Do not scale. |
| DLM adaptation to real MIST fingerprints | rejected | Reduced clean/noisy gap but lowered absolute quality | Redesign objective before retraining. |
| Mixed clean/noisy DLM adaptation | rejected | Worse on both clean and MIST inputs | Do not continue checkpoint. |
| NGBoost token-length model | bounded speed option | About twice as fast, but worse than no-NGBoost at 100 attempts | Use only when throughput matters. |
| MS-BART | rejected as current source | Weak standalone candidates and no useful union gain | Revisit only with a materially stronger checkpoint. |
| Consensus reranker | rejected | Positive first 32, negative held-out last 32 | Overfit; do not scale. |
| Oracle refinement model | rejected | Refined Tanimoto below baseline | Redesign training target and scorer. |
| Constrained STONED-SELFIES expansion | bounded diagnostic | Fixed16 target-absent panel: replacement `+0.0079` (`stoned_fixed16_replacement_v4_clean`, `6024fe6`), paired swap `+0.0091` (`stoned_fixed16_paired_swap_v1`, `94ce6cc`), no new target recoveries or MIST-ranking gain; insertion/deletion failed the formula-survival stop rule | Do not advance unchanged to micro128. Revisit only with a materially different formula-preserving operator or spectrum-aware selection. |
| RankLoop MIST + ChemBERTa dual encoder | rejected | Direct dev64 Tanimoto@1 `-0.1016`, CI `[-0.1417, -0.0648]`; residual fusion selected on dev64 failed locked micro128 with `-0.00393`, CI `[-0.00776, -0.00065]` | Do not repeat direct replacement or residual score tuning on this corpus. |
| RankLoop DreaMS + ChemBERTa dual encoder | rejected | Direct dev64 Tanimoto@1 `-0.0960`, CI `[-0.1318, -0.0641]`; dev-selected residual failed micro256 with `-0.00470`, CI `[-0.01010, -0.00053]`, and was also negative on molecule-disjoint macro64 | Do not advance to 1,024 or full. Move to an independently ablated forward-spectrum scorer. |
| RankLoop ICEBERG forward consistency | rejected | Dev64-selected blend/z-score `0.5` gave Tanimoto@1 `+0.01409` and Exact@1 `+0.0625`, but failed frozen micro128: Tanimoto@1 `-0.00761`, CI `[-0.01688, +0.00013]`; Exact@1 `-0.00781` | Do not retune on compact panels or advance to micro256, macro64, 1,024, or full. A future forward branch must change the scorer or missing-data policy using train/development evidence only. |
| RankLoop APS/RAPS conformal sets | bounded uncertainty utility | Frozen dev64 calibration. RAPS-confidence conditional exact coverage/set size: micro128 `0.8857/5.10`, micro256 `0.9104/5.23`, molecule-disjoint macro64 `1.0000/5.22`. Exact candidate recall is only `0.2734/0.2617/0.2344`; micro256 low-margin coverage is `0.8462`; abstention risk transfers at `0.30-0.40` instead of the calibrated `<=0.10` | Keep RAPS as an honest conditional set-size diagnostic only. Do not claim ranking gain, unconditional 90% coverage, subgroup stability, or reliable abstention. Revisit after full candidate recall improves and a larger disjoint calibration set exists. |

## Audited and prepared research branches

| Architecture | State | Current evidence | Required gate |
| --- | --- | --- | --- |
| ICEBERG spectral refinement | diagnostic only | 50 spectra: Exact@1 `0.16`, Tanimoto@1 `0.4505`; no identical-subset improvement proof | Frozen paired compact comparison. |
| GEMS-style formula-preserving search | implemented diagnostic | Target-blind bounded runner exists | Compare against the same frozen candidate list and budget. |
| DiffMS | audited, not promoted | Architecture and integration path reviewed; no compatible paired FRIGID result | Produce target-blind candidates or scorer outputs on a locked panel. |
| MBGen many-body graph diffusion | blocked by upstream assets | No public trained checkpoint, incomplete loader, evaluation padding issues, unclear license | Resume only after compatible weights and loader are available. |
| DualLGD graph diffusion | interface blocked | Published model uses its own spectrum encoder and Morgan-2048; FRIGID uses MIST Morgan-4096 | Prove train-only folding/interface equivalence before any test run. |
| Selective train-neighbor TTT | prepared, unevaluated | Audit-safe neighbor bundle builder and tests exist | Freeze policy, run micro gate, then macro64 with paired CI. |
| Spectral JEPA | research branch, no confirmed gate | Pretraining direction exists but no confirmed downstream quality gain | Require a frozen downstream candidate or reranking evaluation. |
| RankLoop corpus v1 | prepared, distribution-limited | `4,096` train-only queries and `64` negatives per query with zero train/development scaffold overlap, but only `3.805%` formula-matched negatives and no production-source candidate lists | Reuse infrastructure, but rebuild harder inference-shaped negatives before another learned reranker. |

## Reporting contract

For every architecture, report one of these outcomes:

- `confirmed`: passed the paired gate and may advance;
- `rejected`: failed a gate and must not be repeated unchanged;
- `bounded`: useful only for a stated secondary objective such as speed;
- `prepared`: implementation exists but no quality claim is allowed;
- `blocked`: external assets or a valid interface are missing;
- `running`: include host, session, run directory, progress, and expected next
  artifact.

The Russian experiment narrative remains in
`docs/FRIGID_EXPERIMENT_REPORT_RU.md`; detailed commands and historical runs
remain in `docs/FRIGID_TECHNICAL_RUN_LOG_RU.md`. The literature-backed
bottleneck diagnosis and architecture priorities are recorded in
`docs/FRIGID_2026_RESEARCH_PRIORITIES.md`.
