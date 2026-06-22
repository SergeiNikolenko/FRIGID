# FRIGID Agent Instructions

This repository keeps project-specific agent guidance in `docs/`.

Start here:

- `docs/agent.md`: agent-facing handoff for the consolidated FRIGID project artifact package.
- `docs/FRIGID_OPERATIONAL_USAGE.md`: stable CLI entry points, required inputs, expected outputs, and run artifact layout.
- `docs/FRIGID_project/FRIGID_project_report_20260622/project_report.md`: narrative report covering recent reproduction, MIST diagnostics, oracle fingerprint ablations, ICEBERG refinement, ClearML training, and QA blockers.
- `docs/FRIGID_project/FRIGID_project_report_20260622/artifact_manifest.csv`: structured artifact index with local repository paths, remote run paths, key files, status, and conclusions.

Repository-specific rules:

- Treat `docs/FRIGID_project/` as the canonical local artifact package. Do not duplicate or move it unless explicitly requested.
- Remote paths under `/home/nikolenko/work/Projects/FRIGID` are provenance references to server-side runs and checkpoints. Do not assume those files were copied into this repository.
- The full MSG aggregate is diagnostic evidence, not a clean final paper reproduction, until the documented QA blockers are resolved.
- Keep local work limited to inspection, documentation, packaging, and lightweight smoke checks. Run heavyweight FRIGID benchmarks or training on the configured remote host.
