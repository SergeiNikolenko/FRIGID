# FRIGID historical evidence archive

This directory keeps the June 2026 report, baseline summaries and artifact index.
Large copied run outputs are stored separately on Gauss. The source Git history
and an unchanged original archive are preserved; no unique result files were
removed from those records.

## Stored packages

Host: `Gauss` (existing private research host).
Directory: `/home/nikolenko/work/Experiments/FRIGID/2026-11/pr3-artifact-archive/`.

- `frigid-original-artifacts-a3e47c4.tar.gz`: unchanged `git archive` of
  `docs/FRIGID_project` at commit `a3e47c4d33d12458c13c27668ceea4930b403161`.
- `frigid-evidence-a3e47c4.tar.gz`: portable result evidence package.
- [archive-summary.json](archive-summary.json): both archive hashes, sizes,
  verified file count and the seven external dataset paths.

The portable package retains all **760 regular files (481,451,150 bytes)**
byte-for-byte, verified individually with SHA-256. Its 12 internal links are
relative and resolve inside the archive. Its `archive_manifest.json` records
every file checksum, all original link targets and the handling of each link.

## Missing dataset inputs

This is a **result evidence archive**, not a complete runnable dataset.
The original package did not contain the seven targets behind 252 shard links:
`atom_types.txt`, `edge_types.txt`, `labels.tsv`, `n_counts.txt`, `neuraldecipher`,
`spec_files`, and `subformulae` under the historical
`/home/nikolenko/work/Projects/FRIGID/repro_cache/msg/`.
Those 252 paths are recorded as external references in the manifest and are
not emitted as broken links or fabricated placeholder data. On Gauss, a file
named `labels.tsv` still exists, but its identity with the historical source
has not been established. The current `spec_files`/`subformulae` links point to
an unavailable shared-storage directory. No substitute dataset was bundled.

Historical reports retain their original June observations, including the
252 broken-link QA finding. Repackaging evidence does not fix benchmark
coverage, scientific quality, or the missing source inputs.

## Restore and verify

On Gauss, read expected SHA-256 values from `archive-summary.json` and verify:

```bash
cd /home/nikolenko/work/Experiments/FRIGID/2026-11/pr3-artifact-archive
sha256sum frigid-original-artifacts-a3e47c4.tar.gz frigid-evidence-a3e47c4.tar.gz
mkdir -p /path/to/empty/evidence
# Extract only the portable package. The original preserves historical absolute links.
tar -xzf frigid-evidence-a3e47c4.tar.gz -C /path/to/empty/evidence
```

Extract into a separate evidence directory, not into the source checkout.
Paths in the historical artifact index refer to the extracted package root;
report-relative links work within that extracted tree.

Rebuild and verify from the preserved original:

```bash
python3 scripts/package_benchmark_artifacts.py \
  /path/to/frigid-original-artifacts-a3e47c4.tar.gz \
  /path/to/new-evidence.tar.gz \
  --source-commit a3e47c4d33d12458c13c27668ceea4930b403161 \
  --summary /path/to/new-summary.json
```

The packager refuses to overwrite an existing destination. It verifies all
regular-file hashes and every relative link after writing the archive. Gzip
container timestamps may change the rebuilt archive hash; file hashes and
manifest contents provide the content-level comparison.
