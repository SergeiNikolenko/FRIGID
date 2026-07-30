#!/usr/bin/env python3
"""Score a completed, non-oracle MARLIN ClearML validation task."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


COMMON_REQUIRED_TAGS = {
    "end-to-end",
    "non-oracle",
    "spectrum-derived-fingerprint",
}
COMMON_FORBIDDEN_TAG_MARKERS = (
    "ground-truth-fingerprint",
    "oracle",
    "target-formula",
    "target-length",
)
SPLIT_CONTRACTS = {
    "selection": {
        "required": {"selection-validation"},
        "forbidden": {"heldout-nplib1", "locked-test"},
        "selection_split": "validation",
    },
    "final": {
        "required": {"heldout-nplib1"},
        "forbidden": {"selection-validation"},
        "selection_split": "locked NPLIB1 test",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-id",
        default=os.environ.get("MARLIN_AUTORESEARCH_CLEARML_TASK_ID"),
    )
    parser.add_argument(
        "--split",
        choices=tuple(SPLIT_CONTRACTS),
        default="selection",
    )
    return parser.parse_args()


def validate_tags(tags: set[str], split: str = "selection") -> None:
    contract = SPLIT_CONTRACTS[split]
    required = COMMON_REQUIRED_TAGS | contract["required"]
    missing = required - tags
    if missing:
        raise ValueError(
            f"ClearML task does not satisfy the {split} contract; missing tags: "
            + ", ".join(sorted(missing))
        )
    forbidden_markers = COMMON_FORBIDDEN_TAG_MARKERS + tuple(
        contract["forbidden"]
    )
    forbidden = sorted(
        tag
        for tag in tags
        if any(
            tag.lower() == marker or tag.lower().startswith(marker + "-")
            for marker in forbidden_markers
        )
    )
    if forbidden:
        raise ValueError(
            f"refusing task outside the {split} split contract: "
            + ", ".join(forbidden)
        )


def extract_metrics(last_scalars: dict[str, Any]) -> dict[str, float]:
    molecular = last_scalars.get("Molecular metrics")
    if not isinstance(molecular, dict):
        raise ValueError("ClearML task has no Molecular metrics")
    mapping = {
        "exact_top1": "Exact@1",
        "exact_top10": "Exact@10",
        "candidate_return_rate": "Candidate return",
        "validity": "Validity",
        "mass_validity": "Mass validity",
        "uniqueness": "Uniqueness",
        "tanimoto_top1": "Tanimoto@1 (returned)",
        "tanimoto_top10": "Tanimoto@10 (returned)",
    }
    metrics: dict[str, float] = {}
    for output_name, series in mapping.items():
        value = molecular.get(series)
        if value is None:
            if output_name.startswith("tanimoto_"):
                metrics[output_name] = 0.0
                continue
            raise ValueError(f"ClearML task is missing Molecular metrics/{series}")
        if isinstance(value, dict):
            value = value.get("last")
        metrics[output_name] = float(value)
    metrics["exact_score"] = (
        0.6 * metrics["exact_top1"] + 0.4 * metrics["exact_top10"]
    )
    return metrics


def main() -> None:
    args = parse_args()
    if not args.task_id:
        raise ValueError(
            "--task-id or MARLIN_AUTORESEARCH_CLEARML_TASK_ID is required"
        )

    from clearml import Task

    task = Task.get_task(task_id=args.task_id)
    if str(task.status) != "completed":
        raise ValueError(
            f"ClearML task {task.id} is {task.status!s}, expected completed"
        )
    tags = set(task.get_tags() or ())
    validate_tags(tags, args.split)
    metrics = extract_metrics(task.get_last_scalar_metrics())
    split_name = SPLIT_CONTRACTS[args.split]["selection_split"]
    payload = {
        **metrics,
        "clearml_task_id": task.id,
        "clearml_task_name": task.name,
        "metric_contract": f"{split_name}; spectrum-derived; non-oracle",
        "selection_split": split_name,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
