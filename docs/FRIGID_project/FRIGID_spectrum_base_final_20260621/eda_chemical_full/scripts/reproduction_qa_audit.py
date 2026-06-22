#!/usr/bin/env python3
"""Generate reproducibility QA source files for the FRIGID MSG base run."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PYDEPS = Path("/tmp/frigid_eda_pydeps")
if PYDEPS.exists():
    sys.path.insert(0, str(PYDEPS))

import pandas as pd


ROOT = Path("/Users/nikolenko/Desktop/FRIGID_spectrum_base_final_20260621")
RUN = ROOT / "msg_base_full_ngboost"
AGG = RUN / "aggregate"
OUT = ROOT / "reproduction_qa_sources"


def count_len_pk_warnings() -> int:
    count = 0
    for path in list((RUN / "logs").glob("*")) + list(RUN.glob("*.log")):
        if path.is_file():
            count += path.read_text(errors="ignore").count("data/len.pk not found")
    return count


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RUN / "manifest.json").read_text())
    aggregate = json.loads((AGG / "aggregate_statistics.json").read_text())
    detailed = pd.read_csv(AGG / "detailed_results.csv")
    predictions = pd.read_csv(AGG / "predictions.csv")

    shard_rows = []
    missing_rows = []
    for shard_dir in sorted((RUN / "shard_data").glob("shard_*")):
        shard = shard_dir.name
        split = pd.read_csv(shard_dir / "split.tsv", sep="\t")
        expected_ids = set(split.loc[split["split"] == "test", "name"].astype(str))
        out_path = RUN / "shard_outputs" / shard / "detailed_results.csv"
        if out_path.exists():
            observed = pd.read_csv(out_path)
            observed_ids = set(observed["spec_name"].astype(str))
        else:
            observed_ids = set()
        missing_ids = sorted(expected_ids - observed_ids)
        for spec_name in missing_ids:
            missing_rows.append({"shard": shard, "spec_name": spec_name})
        expected = len(expected_ids)
        observed_count = len(observed_ids)
        shard_rows.append(
            {
                "shard": shard,
                "expected": expected,
                "observed": observed_count,
                "missing": expected - observed_count,
                "coverage": observed_count / expected if expected else None,
            }
        )

    shard_coverage = pd.DataFrame(shard_rows)
    missing = pd.DataFrame(missing_rows)
    shard_coverage.to_csv(OUT / "shard_coverage.csv", index=False)
    missing.to_csv(OUT / "missing_test_spectra.csv", index=False)

    links = []
    for path in (RUN / "shard_data").rglob("*"):
        if path.is_symlink():
            target = os.readlink(path)
            links.append(
                {
                    "path": str(path.relative_to(RUN)),
                    "target": target,
                    "is_absolute_target": os.path.isabs(target),
                    "exists_from_package": path.exists(),
                }
            )
    symlinks = pd.DataFrame(links)
    symlinks.to_csv(OUT / "symlink_status.csv", index=False)

    pred_cols = [f"pred_smiles_{idx}" for idx in range(1, 11)]
    pred_index = predictions.set_index("name")
    position_counts = {idx: 0 for idx in range(1, 11)}
    not_found = 0
    mismatch_examples = []
    for row in detailed[["spec_name", "proposal_smiles"]].itertuples(index=False):
        pred_values = pred_index.loc[row.spec_name, pred_cols].tolist()
        try:
            position = pred_values.index(row.proposal_smiles) + 1
            position_counts[position] += 1
        except ValueError:
            not_found += 1
            if len(mismatch_examples) < 25:
                mismatch_examples.append(
                    {
                        "spec_name": row.spec_name,
                        "proposal_smiles": row.proposal_smiles,
                        "pred_smiles_1": pred_index.loc[row.spec_name, "pred_smiles_1"],
                    }
                )
    proposal_position = pd.DataFrame(
        [{"position": str(k), "rows": v} for k, v in position_counts.items()]
        + [{"position": "not_found", "rows": not_found}]
    )
    proposal_position.to_csv(OUT / "proposal_position_in_predictions.csv", index=False)
    pd.DataFrame(mismatch_examples).to_csv(OUT / "proposal_not_found_examples.csv", index=False)

    joined = detailed[["spec_name", "proposal_smiles"]].merge(
        predictions[["name", "pred_smiles_1"]],
        left_on="spec_name",
        right_on="name",
        how="inner",
    )
    top1_schema_match_rows = int((joined["proposal_smiles"] == joined["pred_smiles_1"]).sum())

    metric_rows = []
    observed_n = len(detailed)
    expected_n = int(manifest["test_samples"])
    for metric in ["exact_match_top1", "exact_match_top10"]:
        successes = float(detailed[metric].sum())
        metric_rows.append(
            {
                "metric": metric,
                "reported_observed_denominator": detailed[metric].mean(),
                "missing_as_failures_denominator": successes / expected_n,
                "successes": int(successes),
                "observed_rows": observed_n,
                "manifest_test_samples": expected_n,
            }
        )
    pd.DataFrame(metric_rows).to_csv(OUT / "metric_denominator_comparison.csv", index=False)

    macro_rows = []
    for metric in ["exact_match_top1", "exact_match_top10", "tanimoto_top1", "mist_tanimoto"]:
        macro_rows.append(
            {
                "metric": metric,
                "micro_per_spectrum": float(detailed[metric].mean()),
                "macro_per_target_smiles": float(detailed.groupby("target_smiles")[metric].mean().mean()),
            }
        )
    pd.DataFrame(macro_rows).to_csv(OUT / "micro_vs_macro_metrics.csv", index=False)

    qa_summary = {
        "manifest_test_samples": expected_n,
        "aggregate_detailed_rows": observed_n,
        "aggregate_prediction_rows": len(predictions),
        "missing_test_objects": expected_n - observed_n,
        "missing_test_object_share": (expected_n - observed_n) / expected_n,
        "shards_total": int(len(shard_coverage)),
        "shards_with_missing": int((shard_coverage["missing"] > 0).sum()),
        "completed_shards_reported": int(aggregate["completed_shards"]),
        "reported_exact_top1": float(aggregate["exact_match_top1"]),
        "reported_exact_top10": float(aggregate["exact_match_top10"]),
        "exact_top1_missing_as_failures": float(detailed["exact_match_top1"].sum() / expected_n),
        "exact_top10_missing_as_failures": float(detailed["exact_match_top10"].sum() / expected_n),
        "paper_target_top1": float(manifest["paper_target_msg_frigid_base_top1"]),
        "paper_target_top10": float(manifest["paper_target_msg_frigid_base_top10"]),
        "symlinks_total": int(len(symlinks)),
        "broken_symlinks": int((~symlinks["exists_from_package"]).sum()) if len(symlinks) else 0,
        "absolute_symlinks": int(symlinks["is_absolute_target"].sum()) if len(symlinks) else 0,
        "proposal_equals_pred_smiles_1_rows": top1_schema_match_rows,
        "proposal_equals_pred_smiles_1_share": top1_schema_match_rows / len(joined),
        "proposal_not_found_in_predictions_top10": not_found,
        "unique_target_smiles": int(detailed["target_smiles"].nunique()),
        "unique_target_inchi_keys": int(detailed["target_inchi_key"].nunique()),
        "len_pk_missing_warnings": count_len_pk_warnings(),
        "rdkit_formula_and_invalid_smiles_checks": "not_rerun_locally_rdkit_unavailable",
        "source_run_dir": str(RUN),
    }
    (OUT / "qa_summary.json").write_text(json.dumps(qa_summary, indent=2, sort_keys=True))

    report = f"""# FRIGID MSG Base Reproduction QA Sources

This folder contains locally recomputed QA evidence for the Desktop copy of the FRIGID MSG base run.

## Key Verified Blockers

- Manifest test samples: {expected_n}
- Aggregate detailed rows: {observed_n}
- Missing test objects: {expected_n - observed_n} ({(expected_n - observed_n) / expected_n:.2%})
- Shards with missing rows: {(shard_coverage["missing"] > 0).sum()} / {len(shard_coverage)}
- Reported completed shards: {aggregate["completed_shards"]}
- Reported Exact Top-1 / Top-10: {aggregate["exact_match_top1"]:.2%} / {aggregate["exact_match_top10"]:.2%}
- Exact Top-1 / Top-10 if missing objects are failures: {detailed["exact_match_top1"].sum() / expected_n:.2%} / {detailed["exact_match_top10"].sum() / expected_n:.2%}
- Broken symlinks: {(~symlinks["exists_from_package"]).sum() if len(symlinks) else 0} / {len(symlinks)}
- `proposal_smiles == pred_smiles_1`: {top1_schema_match_rows} / {len(joined)} ({top1_schema_match_rows / len(joined):.2%})
- Unique target SMILES: {detailed["target_smiles"].nunique()} over {observed_n} spectra
- `data/len.pk not found` warnings: {count_len_pk_warnings()}

## Files

- `qa_summary.json`: compact machine-readable summary.
- `shard_coverage.csv`: expected vs observed rows by shard.
- `missing_test_spectra.csv`: missing MassSpecGym IDs by shard.
- `symlink_status.csv`: symlink target and broken-link status.
- `proposal_position_in_predictions.csv`: where scored `proposal_smiles` appears in `pred_smiles_1..10`.
- `metric_denominator_comparison.csv`: reported observed denominator vs manifest denominator.
- `micro_vs_macro_metrics.csv`: per-spectrum vs per-target-SMILES metrics.

RDKit-dependent formula and invalid-SMILES checks from the pasted audit were not rerun locally because RDKit is not available in the current local Python environment.
"""
    (OUT / "README.md").write_text(report)
    print(json.dumps({"out": str(OUT), **qa_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
