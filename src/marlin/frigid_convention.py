"""Score stored predictions under FRIGID's benchmark conventions.

Our own metric set and FRIGID's agree on `Exact@k`, which both average over every
spectrum and count a spectrum that returned nothing as a miss. They disagree on
`Tanimoto@k`: we average over the spectra that returned a candidate, FRIGID averages
over all of them with a missing candidate contributing 0.0. FRIGID pads its candidate
list with non-formula-matched molecules until it reaches the requested count, so it
almost never returns nothing and the two denominators nearly coincide for it; for a
mass-shell constrained decoder they differ by the candidate return rate.

Comparing the two projects therefore requires one scorer over both prediction files
rather than two metric implementations that happen to share field names. Both
`evaluate_marlin_nplib1.py` and `evaluate_frigid_parity.py` write the same row schema,
so this module reads either.

Every value carries its denominator in `denominators`, because the whole point of this
module is that a metric name without a denominator is not a number you can compare.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _rows(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _mean(values: Iterable[float]) -> float | None:
    collected = [float(v) for v in values if v is not None]
    if not collected:
        return None
    return sum(collected) / len(collected)


def repair_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report how many returned candidates only survived the SAFE repair.

    ``safe_to_smiles`` defaults to ``fix=True`` (``src/dlm/utils/utils_chem.py:26``),
    which deletes every fragment that will not decode and returns what is left. A
    candidate built that way is a stump of the string the model wrote, so scoring it
    as a molecule the model produced makes every validity and fragmentation number
    downstream wrong.

    Prediction files written before ``evaluate_marlin_nplib1.py`` recorded this carry
    no ``generated_candidate_count``. Those rows are **unknown**: they contribute to
    neither the numerator nor the denominator, and never default to clean. The rate
    is ``None`` when no row in the file was examined, so a legacy file reports the
    absence of the measurement instead of a fabricated 0.0.
    """
    known = [row for row in rows if row.get("generated_candidate_count") is not None]
    generated = sum(int(row["generated_candidate_count"]) for row in known)
    repaired = sum(int(row.get("repaired_candidate_count") or 0) for row in known)
    return {
        "rows_with_repair_provenance": len(known),
        "rows_without_repair_provenance": len(rows) - len(known),
        "generated_candidates": generated,
        "repaired_candidates": repaired,
        "repaired_candidate_rate": (repaired / generated) if generated else None,
        "spectra_with_repaired_candidate": sum(
            1 for row in known if int(row.get("repaired_candidate_count") or 0) > 0
        ),
    }


def frigid_convention_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return FRIGID-convention metrics for one prediction file."""
    if not rows:
        return {"total_spectra": 0}

    n = len(rows)
    returned = [row for row in rows if row.get("candidate_returned")]

    def over_all(key: str) -> float:
        # A spectrum with no candidate contributes 0.0, which is FRIGID's rule.
        return sum(float(row.get(key) or 0.0) for row in rows) / n

    metrics: dict[str, Any] = {
        "total_spectra": n,
        "exact_match_top1": over_all("exact_top1"),
        "exact_match_top10": over_all("exact_top10"),
        "tanimoto_top1_mean": over_all("tanimoto_top1"),
        "tanimoto_top10_mean": over_all("tanimoto_top10"),
        "formula_match_success_rate": over_all("formula_top1"),
        "candidate_return_rate": len(returned) / n,
        "never_matched_rate": 1.0 - (len(returned) / n),
        "attempts_mean": _mean(row.get("attempts", 0) for row in rows),
        "truncated_spectra": sum(1 for row in rows if row.get("truncated")),
        **repair_provenance(rows),
    }

    # Tanimoto over every candidate, not only the ranked first: FRIGID's
    # `tanimoto_mean` measures the whole returned set rather than its head.
    per_spectrum_means = []
    for row in rows:
        candidates = row.get("candidates") or []
        similarities = [
            c.get("target_fingerprint_tanimoto")
            for c in candidates
            if c.get("target_fingerprint_tanimoto") is not None
        ]
        per_spectrum_means.append(_mean(similarities) or 0.0)
    metrics["tanimoto_mean"] = sum(per_spectrum_means) / n

    # Attempts spent before the first returned candidate, over the spectra that
    # produced one. Reported separately from attempts_mean because averaging a
    # never-matched spectrum's full budget into it would understate the cost.
    matched_attempts = [
        float(row.get("attempts") or 0.0) for row in returned if row.get("attempts")
    ]
    metrics["avg_attempts_to_match"] = _mean(matched_attempts)

    metrics["denominators"] = {
        "exact_match_top1": "all spectra",
        "exact_match_top10": "all spectra",
        "tanimoto_top1_mean": "all spectra, no candidate counts as 0.0",
        "tanimoto_top10_mean": "all spectra, no candidate counts as 0.0",
        "tanimoto_mean": "all spectra, mean over every returned candidate",
        "formula_match_success_rate": "all spectra",
        "candidate_return_rate": "all spectra",
        "never_matched_rate": "all spectra",
        "attempts_mean": "all spectra",
        "avg_attempts_to_match": "spectra with a returned candidate",
        "repaired_candidate_rate": (
            "candidates generated by spectra whose row carries repair provenance; "
            "null when the file predates the check"
        ),
        "spectra_with_repaired_candidate": "spectra whose row carries repair provenance",
    }
    return metrics


def score_prediction_file(path: str | Path) -> dict[str, Any]:
    return frigid_convention_metrics(_rows(path))
