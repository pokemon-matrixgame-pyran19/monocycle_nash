from __future__ import annotations

import pytest

from monocycle_nash.approximation.random_experiment_domain import ApproximationQualityStatistics


def test_approximation_quality_statistics_summarize() -> None:
    stats = ApproximationQualityStatistics()
    stats.add(0.2, parameters={"group": "small"})
    stats.add(0.5, parameters={"group": "large"})
    stats.add(0.3, parameters={"group": "small"})

    summary = stats.summarize()

    assert summary.count == 3
    assert summary.mean == pytest.approx((0.2 + 0.5 + 0.3) / 3)
    assert summary.stddev == pytest.approx(0.1247219129)


def test_approximation_quality_statistics_summarize_grouped() -> None:
    stats = ApproximationQualityStatistics()
    stats.add(0.2, parameters={"group": "small", "stage": "A"})
    stats.add(0.5, parameters={"group": "large", "stage": "A"})
    stats.add(0.3, parameters={"group": "small", "stage": "B"})

    grouped = stats.summarize_grouped("group")

    assert grouped["small"].count == 2
    assert grouped["small"].mean == pytest.approx(0.25)
    assert grouped["large"].count == 1


def test_approximation_quality_statistics_requires_scores() -> None:
    stats = ApproximationQualityStatistics()

    with pytest.raises(ValueError, match="score"):
        stats.summarize()


def test_approximation_quality_statistics_requires_group_keys() -> None:
    stats = ApproximationQualityStatistics()
    stats.add(0.1)

    with pytest.raises(ValueError, match="group_keys"):
        stats.summarize_grouped()
