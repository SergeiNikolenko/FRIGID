# FRIGID 2026 Research Priorities

This note records the current diagnosis of the FRIGID quality bottleneck and
the architecture priorities derived from the 2025-2026 literature. It is a
research decision, not benchmark evidence. Model states and measured results
remain authoritative in `docs/FRIGID_MODEL_STATUS.md`.

## Current bottleneck diagnosis

MS/MS structure elucidation is intrinsically underdetermined. Closely related
isomers can produce similar fragments, while collision energy, adduct,
ionization mode, and instrument introduce substantial domain shift. Exact
formula conditioning narrows the search space but is also a strong assumption:
formula prediction is itself uncertain in realistic discovery workflows.

The current FRIGID pipeline has three sequential limits:

1. **Spectrum representation.** MIST fingerprint error creates an information
   ceiling before generation. Direct DreaMS fingerprint replacement,
   calibration, distillation, residual correction, and full fine-tuning did
   not improve the locked gates.
2. **Candidate recall.** The frozen four-source union materially improved
   recall and remains the confirmed leader. Additional blind SELFIES mutations
   produced many novel formula-valid structures but recovered no new targets.
3. **Candidate ranking.** STONED and two-switch diagnostics improved oracle
   best-candidate similarity more than frozen MIST-ranked metrics. The current
   actionable bottleneck is therefore spectrum-aware discrimination among
   formula-matched, structurally similar candidates, not unrestricted candidate
   count.

## Relevant 2025-2026 directions

### Cross-modal retrieval and reranking

- MSAlign aligns frozen DreaMS and ChemBERTa encoders with lightweight
  projection heads and a candidate-based contrastive objective.
- SECS combines spectrum-molecule contrastive alignment with evolutionary
  structure search.
- Cross-modal retrieval work increasingly emphasizes formula-, mass-,
  scaffold-, and fingerprint-matched hard negatives rather than random
  negatives.

This direction directly matches the observed FRIGID ranking bottleneck because
it optimizes spectrum-candidate compatibility without requiring prediction of
every Morgan fingerprint bit.

### Constrained and structure-informed generation

- FRIGID uses inference-time scaling and forward-fragment consistency for
  targeted remasking.
- DiffMS and FlowMS constrain graph generation with a known formula; FlowMS
  replaces diffusion with discrete flow matching.
- MARLIN replaces oracle formula conditioning with a precursor-mass shell and
  explicitly models fingerprint noise and candidate diversity.
- MADGEN, MSAnchor, and MetGenX reduce combinatorial search through retrieved
  scaffolds, anchor points, or structurally related templates.

These methods are candidates for selective recall expansion, especially on
queries where the frozen union has low estimated recall. They should not
replace the complete union without identical-subset paired evidence.

### Forward consistency and uncertainty

- Cycle-MS jointly learns inverse structure prediction and forward spectrum
  reconstruction through cycle consistency.
- Conformal and selective retrieval methods expose per-spectrum uncertainty
  and allow abstention or variable-size candidate sets under ambiguity.

Forward consistency is a plausible independent reranking feature. Uncertainty
estimation improves reliability but cannot repair an uninformative ranking
under severe distribution shift.

## Evaluation caveats

Reported numbers across papers are not directly comparable. Results depend on
the molecular split, candidate-set construction, formula availability,
instrument distribution, database overlap, and whether an oracle scaffold or
template is used. FRIGID decisions must therefore continue to use frozen
target-blind candidates, molecule-cluster paired confidence intervals, and the
existing micro256 plus molecule-disjoint macro64 promotion rule.

## Prioritized execution plan

1. Complete the full 17,082-spectrum frozen four-source baseline on Spectrum.
2. Implement the SPA-159 dual-encoder reranker with separate frozen MIST and
   frozen DreaMS spectral ablations.
3. Train with train-only formula-matched and Morgan/scaffold-similar hard
   negatives; rerank only the frozen top-50 and top-100 candidate sets.
4. If the dual encoder passes the development and compact gates, evaluate a
   top-32 or top-64 spectrum-candidate cross-encoder.
5. Add forward-spectrum consistency as an independently ablated ensemble
   feature.
6. Use structure-informed generation only for predeclared low-recall queries.
7. Add calibrated abstention or conformal candidate sets after ranking quality
   improves.

## Primary references

- FRIGID: https://arxiv.org/abs/2604.16648
- MSAlign: https://arxiv.org/abs/2605.19752
- DreaMS: https://www.nature.com/articles/s41587-025-02663-3
- SECS: https://www.nature.com/articles/s41467-026-73846-y
- FlowMS: https://arxiv.org/abs/2603.18397
- MARLIN: https://arxiv.org/abs/2607.04774
- MSAnchor: https://doi.org/10.1609/aaai.v40i2.37064
- MADGEN: https://arxiv.org/abs/2501.01950
- MetGenX: https://doi.org/10.1038/s41467-026-72149-6
- Cycle-MS: https://doi.org/10.1021/acs.jcim.5c02981
- Reliable molecular retrieval with conformal prediction:
  https://doi.org/10.1021/acs.jcim.6c00727
