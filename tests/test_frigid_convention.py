from marlin.frigid_convention import frigid_convention_metrics


def _row(*, returned, exact=False, tanimoto=0.0, formula=False, attempts=8,
         candidate_similarities=()):
    return {
        "candidate_returned": returned,
        "exact_top1": exact,
        "exact_top10": exact,
        "tanimoto_top1": tanimoto,
        "tanimoto_top10": tanimoto,
        "formula_top1": formula,
        "attempts": attempts,
        "candidates": [
            {"target_fingerprint_tanimoto": s} for s in candidate_similarities
        ],
    }


def test_tanimoto_counts_a_missing_candidate_as_zero():
    # Two spectra, one returns a 0.8 match and one returns nothing. Our own metric
    # would report 0.8 over the single returned row; FRIGID's reports 0.4.
    rows = [
        _row(returned=True, tanimoto=0.8, candidate_similarities=(0.8,)),
        _row(returned=False, tanimoto=0.0),
    ]

    metrics = frigid_convention_metrics(rows)

    assert metrics["tanimoto_top1_mean"] == 0.4
    assert metrics["candidate_return_rate"] == 0.5
    assert metrics["never_matched_rate"] == 0.5
    assert metrics["total_spectra"] == 2


def test_exact_match_already_agrees_between_the_conventions():
    rows = [_row(returned=True, exact=True, tanimoto=1.0), _row(returned=False)]

    metrics = frigid_convention_metrics(rows)

    assert metrics["exact_match_top1"] == 0.5
    assert metrics["exact_match_top10"] == 0.5


def test_tanimoto_mean_spans_every_returned_candidate():
    rows = [_row(returned=True, tanimoto=0.9, candidate_similarities=(0.9, 0.1))]

    assert frigid_convention_metrics(rows)["tanimoto_mean"] == 0.5


def test_attempts_to_match_excludes_never_matched_spectra():
    rows = [
        _row(returned=True, attempts=4),
        _row(returned=False, attempts=64),
    ]

    metrics = frigid_convention_metrics(rows)

    assert metrics["avg_attempts_to_match"] == 4.0
    assert metrics["attempts_mean"] == 34.0


def test_empty_file_reports_no_spectra_rather_than_dividing_by_zero():
    assert frigid_convention_metrics([]) == {"total_spectra": 0}
