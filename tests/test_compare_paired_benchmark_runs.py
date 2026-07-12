import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "compare_paired_benchmark_runs.py"
SPEC = importlib.util.spec_from_file_location(
    "compare_paired_benchmark_runs", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_frame(names, tanimoto, exact, formula_matches):
    return pd.DataFrame(
        {
            "spec_name": names,
            "fingerprint_source": "mist_binary",
            "target_smiles": [f"C{i}" for i in range(len(names))],
            "target_inchi_key": [f"KEY{i}" for i in range(len(names))],
            "mist_tanimoto": [0.5] * len(names),
            "tanimoto_top1": tanimoto,
            "tanimoto_top10": tanimoto,
            "exact_match_top1": exact,
            "exact_match_top10": exact,
            "total_formula_matched": formula_matches,
            "total_valid": [10] * len(names),
            "total_generated": [20] * len(names),
        }
    )


class ComparePairedBenchmarkRunsTest(unittest.TestCase):
    def test_compare_runs_aligns_order_and_derives_formula_success(self):
        reference = make_frame(
            ["s1", "s2", "s3", "s4"],
            [0.1, 0.2, 0.3, 0.4],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
        )
        candidate = make_frame(
            ["s1", "s2", "s3", "s4"],
            [0.2, 0.3, 0.4, 0.5],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
        ).iloc[::-1]

        summary, paired = MODULE.compare_runs(
            reference,
            candidate,
            ["tanimoto_top1", "exact_match_top1", "formula_success"],
            bootstrap_resamples=200,
            confidence=0.95,
            seed=7,
        )

        self.assertEqual(summary["n_pairs"], 4)
        self.assertFalse(summary["same_input_order"])
        self.assertAlmostEqual(summary["metrics"]["tanimoto_top1"]["mean_delta"], 0.1)
        self.assertAlmostEqual(summary["metrics"]["tanimoto_top1"]["ci_low"], 0.1)
        self.assertAlmostEqual(summary["metrics"]["tanimoto_top1"]["ci_high"], 0.1)
        self.assertAlmostEqual(
            summary["metrics"]["formula_success"]["mean_delta"], 0.25
        )
        self.assertEqual(paired["spec_name"].tolist(), ["s1", "s2", "s3", "s4"])

    def test_compare_runs_rejects_subset_mismatch(self):
        reference = make_frame(["s1", "s2"], [0.1, 0.2], [0, 0], [0, 0])
        candidate = make_frame(["s1", "s3"], [0.2, 0.3], [0, 0], [0, 0])

        with self.assertRaisesRegex(ValueError, "Paired subset mismatch"):
            MODULE.compare_runs(
                reference,
                candidate,
                ["tanimoto_top1"],
                bootstrap_resamples=10,
                confidence=0.95,
                seed=42,
            )

    def test_load_results_requires_source_for_multi_source_csv(self):
        frame = make_frame(["s1", "s2"], [0.1, 0.2], [0, 0], [0, 0])
        second = frame.copy()
        second["fingerprint_source"] = "ground_truth"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detailed_results.csv"
            pd.concat([frame, second], ignore_index=True).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "multiple fingerprint sources"):
                MODULE.load_results(path, fingerprint_source=None)

    def test_molecule_bootstrap_resamples_connectivity_clusters(self):
        reference = make_frame(
            ["s1", "s2", "s3", "s4"],
            [0.1, 0.2, 0.3, 0.4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        )
        candidate = reference.copy()
        candidate["tanimoto_top1"] = [0.2, 0.3, 0.3, 0.4]
        reference["target_inchi_key"] = ["A-X", "A-Y", "B-X", "C-X"]
        candidate["target_inchi_key"] = reference["target_inchi_key"]

        summary, paired = MODULE.compare_runs(
            reference,
            candidate,
            ["tanimoto_top1"],
            bootstrap_resamples=200,
            confidence=0.95,
            seed=9,
            bootstrap_unit="molecule",
            cluster_column="target_inchi_key",
        )

        self.assertEqual(summary["bootstrap"]["unit"], "molecule")
        self.assertEqual(summary["bootstrap"]["n_clusters"], 3)
        self.assertEqual(paired["bootstrap_cluster"].tolist(), ["A", "A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
