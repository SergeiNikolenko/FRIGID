"""A corpus arm has to say, per row, whether it could have trained on the answer.

The first 10,000,000 fp2mol rows carry 23 of the 320 clean-panel connectivity
blocks against a 3-spectrum baseline (``scripts/build_nplib1_holdout_inchikeys.py``),
so a corpus stage that loses its exclusion list does not crash --- it quietly
scores itself on structures it trained on. The guard is therefore not the
exclusion alone but the *evidence* that the exclusion was in the path: every
prediction row carries ``seen_in_corpus``, and the metrics carry Exact@1 on the
held-out complement beside the overall number.
"""

from __future__ import annotations

import sys

import pytest

from marlin.corpus_stream import PACKAGED_HOLDOUT_INCHIKEYS
from scripts.evaluate_marlin_nplib1 import (
    corpus_contamination_metrics,
    parse_args,
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


def _row(name: str, *, seen, exact: bool) -> dict:
    return {
        "spec_name": name,
        "seen_in_corpus": seen,
        "exact_top1": exact,
        "exact_top10": exact,
    }


def test_the_flag_is_off_by_default_and_unknown_is_not_clean(monkeypatch):
    monkeypatch.setattr(sys, "argv", _MINIMAL_EVALUATION_ARGV)
    assert parse_args().corpus_exclude_inchikeys is None

    metrics = corpus_contamination_metrics(
        [{"spec_name": "a", "exact_top1": True, "exact_top10": True}]
    )
    assert metrics["seen_in_corpus_spectra"] is None
    assert metrics["exact_top1_corpus_unseen"] is None


def test_a_wired_exclusion_reports_zero_seen_rows_and_the_same_exact_at_1():
    rows = [
        _row("a", seen=False, exact=True),
        _row("b", seen=False, exact=False),
        _row("c", seen=False, exact=False),
        _row("d", seen=False, exact=True),
    ]

    metrics = corpus_contamination_metrics(rows)

    assert metrics["seen_in_corpus_spectra"] == 0
    assert metrics["corpus_unseen_spectra"] == 4
    # With nothing seen, the complement is the panel and the two numbers agree.
    assert metrics["exact_top1_corpus_unseen"] == pytest.approx(0.5)


def test_a_contaminated_panel_is_visible_in_the_split():
    rows = [
        _row("a", seen=True, exact=True),
        _row("b", seen=True, exact=True),
        _row("c", seen=False, exact=False),
        _row("d", seen=False, exact=False),
    ]

    metrics = corpus_contamination_metrics(rows)

    assert metrics["seen_in_corpus_spectra"] == 2
    # The headline would read 50%; on the rows the corpus could not have taught
    # it, the decoder scores zero. That is the failure this flag exists to make
    # impossible to miss.
    assert metrics["exact_top1_corpus_unseen"] == pytest.approx(0.0)
    assert metrics["exact_top1_corpus_seen"] == pytest.approx(1.0)


def test_rows_from_an_older_file_are_unknown_rather_than_unseen():
    rows = [
        _row("a", seen=False, exact=True),
        {"spec_name": "b", "exact_top1": True, "exact_top10": True},
    ]

    metrics = corpus_contamination_metrics(rows)

    assert metrics["corpus_provenance_spectra"] == 1
    assert metrics["corpus_unseen_spectra"] == 1


def test_the_cli_accepts_the_packaged_list(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _MINIMAL_EVALUATION_ARGV
        + ["--corpus-exclude-inchikeys", str(PACKAGED_HOLDOUT_INCHIKEYS)],
    )
    assert parse_args().corpus_exclude_inchikeys == PACKAGED_HOLDOUT_INCHIKEYS


def test_the_panel_is_entirely_inside_the_packaged_exclusion():
    """The guard's own precondition, measured rather than assumed.

    Every connectivity block of the clean 321 panel is in the packaged list, so
    a correctly wired corpus run must report ``seen_in_corpus_spectra == 0`` on
    that panel. If this test fails, the exclusion list and the panel have drifted
    apart and no corpus arm may be read on it.
    """
    import pandas as pd

    from marlin.training import load_excluded_connectivity_keys

    excluded = load_excluded_connectivity_keys(PACKAGED_HOLDOUT_INCHIKEYS)
    panel = pd.read_csv(
        "configs/benchmarks/nplib1_v1/nplib1_val_clean322_v1.tsv", sep="\t"
    )
    blocks = {
        str(value).split("-")[0] for value in panel["inchikey_first_block"].dropna()
    }
    assert blocks
    assert not (blocks - excluded)
