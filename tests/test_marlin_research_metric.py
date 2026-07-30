import pytest

from marlin.research_metric import staged_research_metric


def test_stage_bands_are_lexicographic() -> None:
    valid = staged_research_metric({"validity": 1.0})
    mass_valid = staged_research_metric(
        {"validity": 0.1, "mass_validity": 0.01}
    )
    returned = staged_research_metric(
        {
            "validity": 0.1,
            "mass_validity": 0.01,
            "candidate_return_rate": 0.01,
        }
    )
    exact = staged_research_metric(
        {
            "validity": 0.1,
            "mass_validity": 0.01,
            "candidate_return_rate": 0.01,
            "exact_top10": 0.01,
        }
    )

    assert valid["research_score"] == 1.0
    assert mass_valid["research_score"] > valid["research_score"]
    assert returned["research_score"] > mass_valid["research_score"]
    assert exact["research_score"] > returned["research_score"]


def test_invalid_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="validity"):
        staged_research_metric({"validity": float("nan")})
