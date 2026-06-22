# Agent Note: FRIGID Project Artifacts

The consolidated FRIGID project artifact package is stored here:

- [FRIGID_project](./FRIGID_project/)

This folder was moved from the Desktop into the repository documentation tree so future agents can find the experiment outputs, QA evidence, reports, and artifact index from project context instead of relying on a user-specific Desktop location. The root [AGENTS.md](../AGENTS.md) is only a thin dispatcher; this file is the detailed project-artifact handoff.

Key entry points:

- [Project report](./FRIGID_project/FRIGID_project_report_20260622/project_report.md): narrative summary of the recent FRIGID work across benchmark reproduction, MIST diagnostics, oracle fingerprint ablations, ICEBERG refinement, ClearML training, and QA blockers.
- [Artifact manifest](./FRIGID_project/FRIGID_project_report_20260622/artifact_manifest.csv): structured index of local artifacts, remote run paths, key files, status, and conclusions.
- [Interactive EDA report](./FRIGID_project/FRIGID_spectrum_base_final_20260621/eda_chemical_full/frigid_chemical_eda_report.html): local Plotly report for the MSG benchmark chemistry/QA analysis.
- [Reproduction QA sources](./FRIGID_project/FRIGID_spectrum_base_final_20260621/reproduction_qa_sources/): CSV/JSON evidence for missing rows, broken symlinks, denominator changes, and output-schema issues.

Important context:

- The package includes large local artifacts and copied result folders. Do not duplicate it unless explicitly needed.
- Some reports preserve remote paths under `/home/nikolenko/work/Projects/FRIGID` for reproducibility. Those paths refer to server-side runs and checkpoints that were not copied into this repository.
- The full MSG aggregate is diagnostic evidence, not a clean final paper reproduction, because QA found missing manifest rows, broken package symlinks, and output-schema mismatch.
