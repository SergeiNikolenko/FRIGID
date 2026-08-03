# FRIGID Project Report Package

Created: 2026-06-22

This package consolidates the latest useful state from four Codex threads:

- `019edb13-c2bd-7752-ad25-430bc319049b`: MIST diagnostics and lightweight adapter experiments.
- `019eeeaa-1cec-7a12-bbd6-51e9d1912f8f`: oracle/ground-truth fingerprint ablations.
- `019eeedc-9d7d-7470-9987-c5eb9c8a22be`: oracle-refinement training and ClearML runs.
- `019ed1bb-12f7-7d20-969e-7d6dd2c73e3e`: full-run EDA, interactive report, and reproduction QA.

Files in this package:

- `project_report.md`: narrative status report with what was tried, what worked, what failed, and current conclusions.
- `artifact_manifest.csv`: structured artifact index with local paths, remote paths, key files, and status.
- `artifacts/`: symlinks to the local repository artifact folders and HTML report. Large folders are not duplicated.

Important scope notes:

- The symlinks point to existing local repository artifacts under `docs/FRIGID_project/`. Remote checkpoints and server-only outputs are listed in the manifest but are not copied here.
- The report focuses on the latest thread answers plus adjacent recent work that those answers summarized.
- The full MSG baseline is useful evidence, but the QA audit found blockers that prevent calling it a clean final paper reproduction.
