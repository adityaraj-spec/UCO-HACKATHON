import pytest

from evaluation.metrics import compute_eer, wilson_interval


def test_compute_eer_separates_perfect_scores():
    result = compute_eer(genuine_scores=[0.8, 0.9], impostor_scores=[0.1, 0.2])

    assert result.eer_percent == pytest.approx(0.0)
    assert result.far == pytest.approx(0.0)
    assert result.frr == pytest.approx(0.0)


def test_compute_eer_requires_both_classes():
    with pytest.raises(ValueError):
        compute_eer(genuine_scores=[], impostor_scores=[0.1])

    with pytest.raises(ValueError):
        compute_eer(genuine_scores=[0.9], impostor_scores=[])


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(successes=5, total=20)

    assert 0.0 <= low <= 0.25 <= high <= 1.0


def test_wilson_interval_zero_total_is_zero_width():
    assert wilson_interval(successes=0, total=0) == (0.0, 0.0)
