import csv
import json

from scripts.make_mist_cf_clean_split import make_clean_split


def test_make_clean_split_moves_all_connectivity_overlaps_to_test(tmp_path):
    split = tmp_path / "split.tsv"
    audit = tmp_path / "audit.json"
    output = tmp_path / "clean.tsv"
    with split.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["spec", "Fold_0"], delimiter="\t")
        writer.writeheader()
        writer.writerows(
            [
                {"spec": "a", "Fold_0": "train"},
                {"spec": "b", "Fold_0": "val"},
                {"spec": "c", "Fold_0": "test"},
                {"spec": "d", "Fold_0": "train"},
            ]
        )
    audit.write_text(
        json.dumps(
            {
                "connectivity_audit": {
                    "memberships": [
                        {"spec_name": "a"},
                        {"spec_name": "b"},
                        {"spec_name": "c"},
                    ]
                }
            }
        )
    )

    result = make_clean_split(split, audit, output)

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["Fold_0"] for row in rows] == ["test", "test", "test", "train"]
    assert result["moved_to_test_counts"] == {"train": 1, "val": 1}
    assert result["clean_split_counts"] == {"test": 3, "train": 1}
    assert result["remaining_train_or_validation_connectivity_overlap"] == 0
