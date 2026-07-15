import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "build_encoder_benchmark_partitions.py"
)
SPEC = importlib.util.spec_from_file_location("build_encoder_benchmark_partitions", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_partition_cli_writes_hashed_molecule_disjoint_manifest(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    output_dir = tmp_path / "partitions"
    pd.DataFrame(
        {
            "fingerprint_index": [0, 1, 2, 3, 4],
            "spec_name": ["a1", "a2", "b", "c", "d"],
            "inchi_key_first_block": ["AAAA", "AAAA", "BBBB", "CCCC", "DDDD"],
        }
    ).to_csv(metadata_path, index=False)

    exit_code = MODULE.main(
        [
            "--metadata",
            str(metadata_path),
            "--calibration-fraction",
            "0.5",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    manifest = pd.read_csv(output_dir / "encoder_benchmark_partitions.csv")
    summary = json.loads((output_dir / "partition_summary.json").read_text())
    assert set(manifest["benchmark_partition"]) == {"calibration", "evaluation"}
    assert (
        manifest.groupby("inchi_key_first_block")["benchmark_partition"].nunique().max()
        == 1
    )
    assert sum(summary["rows"].values()) == 5
    assert sum(summary["clusters"].values()) == 4
