"""A repaired SAFE string must never pass as a candidate the model wrote.

``safe_to_smiles`` defaults to ``fix=True`` (``src/dlm/utils/utils_chem.py:26``): it
deletes every fragment that will not decode and returns what is left. The two classes
the decoder actually produces both come back as molecules that way -- an aromatic atom
outside a ring, and an aromatic ring RDKit cannot kekulize -- so a broken string can
enter the candidate pool as a stump carrying a fraction of the target mass.

These tests pin the three things that stop it: the classifier, the exclusion, and the
aggregate rate; plus the compatibility rule that a prediction file written before the
check is *unknown*, not clean.
"""

from dlm.utils.utils_chem import safe_to_smiles
from marlin.frigid_convention import frigid_convention_metrics, repair_provenance

import sys

from scripts.evaluate_marlin_nplib1 import (
    parse_args,
    repair_provenance_metrics,
    safe_string_needed_repair,
    split_repaired_candidates,
)

_MINIMAL_EVALUATION_ARGV = [
    "evaluate_marlin_nplib1.py",
    "--checkpoint", "checkpoint.ckpt",
    "--tokenizer", "tokenizer.json",
    "--metadata", "metadata.csv",
    "--fingerprints", "fingerprints.npz",
    "--fingerprint-key", "probs",
    "--lane", "dreams",
    "--output-dir", "out",
]


def test_excluding_repaired_candidates_is_the_default_on_the_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", _MINIMAL_EVALUATION_ARGV)
    assert parse_args().keep_repaired_candidates is False

    monkeypatch.setattr(
        sys, "argv", _MINIMAL_EVALUATION_ARGV + ["--keep-repaired-candidates"]
    )
    assert parse_args().keep_repaired_candidates is True


# Both classes the project lead localised in generation. Neither parses as written;
# both come back from the repair as a stump.
_SEVEN_MEMBERED_AROMATIC_RING = "c1cccccc1-1.[H+]2.O2-1"
_AROMATIC_ATOM_OUTSIDE_A_RING = "c12ccccc1.c13.[H]1.[o+]23"
# Written by the sampler and returned as a real candidate on the locked test split.
_A_STRING_THE_MODEL_SPELLED = "CC12CCC3=C4CCC(=O)C=C4CCC3C1CCC25O.C5C#%19.C=%19"


def test_a_safe_string_that_parses_as_written_is_clean():
    assert safe_to_smiles(_A_STRING_THE_MODEL_SPELLED, fix=False) is not None
    assert safe_string_needed_repair(_A_STRING_THE_MODEL_SPELLED) is False
    assert safe_string_needed_repair("c1ccccc1") is False


def test_a_safe_string_that_only_survives_repair_is_marked_repaired():
    # The repair answers, which is exactly why the flag has to exist: without it
    # "O" enters the pool as a candidate for a 300-600 Da target.
    assert safe_to_smiles(_SEVEN_MEMBERED_AROMATIC_RING, fix=True) == "O"
    assert safe_to_smiles(_SEVEN_MEMBERED_AROMATIC_RING, fix=False) is None
    assert safe_string_needed_repair(_SEVEN_MEMBERED_AROMATIC_RING) is True

    assert safe_to_smiles(_AROMATIC_ATOM_OUTSIDE_A_RING, fix=True) == "[H][H].c1ccccc1"
    assert safe_string_needed_repair(_AROMATIC_ATOM_OUTSIDE_A_RING) is True


def test_a_repaired_candidate_is_not_scored_by_default():
    clean = {"smiles": "c1ccccc1", "repaired": False}
    stump = {"smiles": "O", "repaired": True}

    scored, repaired = split_repaired_candidates([stump, clean], keep_repaired=False)

    assert scored == [clean]
    assert repaired == [stump]


def test_the_escape_hatch_keeps_a_repaired_candidate_flagged_not_hidden():
    clean = {"smiles": "c1ccccc1", "repaired": False}
    stump = {"smiles": "O", "repaired": True}

    scored, repaired = split_repaired_candidates([stump, clean], keep_repaired=True)

    assert scored == [stump, clean]
    assert repaired == []
    # Kept means kept *and* flagged; the row still shows which one it is.
    assert [candidate["repaired"] for candidate in scored] == [True, False]


def _row(*, generated, repaired):
    return {
        "generated_candidate_count": generated,
        "repaired_candidate_count": repaired,
    }


def test_the_aggregation_reports_the_repaired_candidate_rate():
    rows = [_row(generated=4, repaired=1), _row(generated=6, repaired=0)]

    for metrics in (repair_provenance_metrics(rows), repair_provenance(rows)):
        assert metrics["generated_candidates"] == 10
        assert metrics["repaired_candidates"] == 1
        assert metrics["repaired_candidate_rate"] == 0.1
        assert metrics["spectra_with_repaired_candidate"] == 1
        assert metrics["rows_with_repair_provenance"] == 2
        assert metrics["rows_without_repair_provenance"] == 0


def test_a_stored_row_without_the_field_is_unknown_and_does_not_crash_the_scorer():
    # Exactly the shape of every row in the stored prediction files: no provenance
    # keys at all.
    legacy = {
        "spec_name": "CCMSLIB00000001645",
        "candidate_returned": True,
        "exact_top1": False,
        "exact_top10": False,
        "tanimoto_top1": 0.4,
        "tanimoto_top10": 0.4,
        "formula_top1": False,
        "attempts": 8,
        "candidates": [{"target_fingerprint_tanimoto": 0.4}],
    }

    metrics = frigid_convention_metrics([legacy])

    # The file is scored as before -- nothing about it changes.
    assert metrics["total_spectra"] == 1
    assert metrics["candidate_return_rate"] == 1.0
    assert metrics["tanimoto_top1_mean"] == 0.4
    # ...and the rate says "not measured", not 0.0 and not 1.0.
    assert metrics["repaired_candidate_rate"] is None
    assert metrics["rows_with_repair_provenance"] == 0
    assert metrics["rows_without_repair_provenance"] == 1
    assert repair_provenance_metrics([legacy])["repaired_candidate_rate"] is None


def test_a_mixed_file_rates_only_the_rows_that_were_examined():
    legacy = {"spec_name": "old", "candidate_returned": False}
    rows = [legacy, _row(generated=2, repaired=1)]

    metrics = repair_provenance(rows)

    assert metrics["rows_with_repair_provenance"] == 1
    assert metrics["rows_without_repair_provenance"] == 1
    # 1/2, not 1/2-of-three-rows: the unexamined row is outside the denominator.
    assert metrics["repaired_candidate_rate"] == 0.5
