from __future__ import annotations

import pytest

from monocycle_nash.approximation.random_experiment_domain import ApproximationQualityStatistics


def test_approximation_quality_statistics_summarize() -> None:
    stats = ApproximationQualityStatistics()
    stats.add(0.2)
    stats.add(0.5)
    stats.add(0.3)

    summary = stats.summarize()

    assert summary.count == 3
    assert summary.minimum == pytest.approx(0.2)
    assert summary.maximum == pytest.approx(0.5)
    assert summary.mean == pytest.approx((0.2 + 0.5 + 0.3) / 3)
    assert summary.median == pytest.approx(0.3)
    assert summary.stddev == pytest.approx(0.1247219129)


def test_approximation_quality_statistics_requires_scores() -> None:
    stats = ApproximationQualityStatistics()

    with pytest.raises(ValueError, match="score"):
        stats.summarize()
