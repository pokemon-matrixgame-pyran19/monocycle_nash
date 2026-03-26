from __future__ import annotations

import pytest

from monocycle_nash.approximation.random_experiment_statistics import ApproximationQualityStatistics


def test_approximation_quality_statistics_summarize() -> None:
    stats = ApproximationQualityStatistics("group")
    stats.add(0.2, "small")
    stats.add(0.5, "large")
    stats.add(0.3, "small")

    summary = stats.summarize()

    assert summary.count == 3
    assert summary.mean == pytest.approx((0.2 + 0.5 + 0.3) / 3)
    assert summary.stddev == pytest.approx(0.1247219129)


def test_approximation_quality_statistics_summarize_grouped() -> None:
    stats = ApproximationQualityStatistics("group", "stage")
    stats.add(0.2, "small", "A")
    stats.add(0.5, "large", "A")
    stats.add(0.3, "small", "B")

    grouped = stats.summarize_grouped("group")

    assert grouped[("small",)].count == 2
    assert grouped[("small",)].mean == pytest.approx(0.25)
    assert grouped[("large",)].count == 1


def test_approximation_quality_statistics_summarize_grouped_by_composite_labels() -> None:
    stats = ApproximationQualityStatistics("group", "stage")
    stats.add(0.2, "small", "A")
    stats.add(0.5, "small", "A")
    stats.add(0.3, "small", "B")

    grouped = stats.summarize_grouped("group", "stage")

    assert grouped[("small", "A")].count == 2
    assert grouped[("small", "A")].mean == pytest.approx(0.35)
    assert grouped[("small", "B")].count == 1


def test_approximation_quality_statistics_requires_scores() -> None:
    stats = ApproximationQualityStatistics("group")

    with pytest.raises(ValueError, match="score"):
        stats.summarize()


def test_approximation_quality_statistics_requires_group_keys() -> None:
    stats = ApproximationQualityStatistics("group")
    stats.add(0.1, "small")

    with pytest.raises(ValueError, match="group_keys"):
        stats.summarize_grouped()


def test_approximation_quality_statistics_requires_fixed_label_value_count() -> None:
    stats = ApproximationQualityStatistics("group", "stage")

    with pytest.raises(ValueError, match="label_values"):
        stats.add(0.1, "small")
