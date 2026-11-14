# Autoresearch Guidance

This repository is bound for a FRIGID throughput and quality campaign.

Before proposing a scored iteration:

1. Read `.autoresearch/task.yaml`, `.autoresearch/scorer_contract.yaml`,
   `.autoresearch/research_backlog.md`, and `.autoresearch/state.json`.
2. Inspect the exact code path being changed.
3. Produce a hypothesis card with mechanism, evidence, files touched, expected
   metric effect, guard risk, scorer budget, rejection criteria, and novelty
   fingerprint.
4. Do not run heavy work locally. Use the scorer contract.
5. Do not run the scorer while `frigid_spectrum_base` is active on spectrum
   unless `ALLOW_CONCURRENT_FRIGID_SCORER=1` is explicitly set.

Scored iterations optimize `seconds_per_case` while preserving quality guards.
Repair-only changes are debug work and should not consume a comparable GPU
scorer slot unless paired with a material mechanism.
