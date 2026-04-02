from __future__ import annotations

from monocycle_nash.analysis.app.experiment_team_strict_spectrum import (
    _build_hypothesis_analysis,
    _summarize_trials,
    _write_trials_csv,
)


def test_summarize_trials_adds_gap_hypothesis_metrics() -> None:
    trials = [
        {
            "trial_index": 0,
            "lambda1": 2.4,
            "lambda2": 1.2,
            "lambda3": 0.9,
            "ratio2_to_1": 0.5,
            "ratio3_to_1": 0.375,
            "dominant_gap": 2.0,
            "support_size": 3,
            "is_support_3": True,
        },
        {
            "trial_index": 1,
            "lambda1": 1.4,
            "lambda2": 1.0,
            "lambda3": 0.6,
            "ratio2_to_1": 0.714285,
            "ratio3_to_1": 0.428571,
            "dominant_gap": 1.4,
            "support_size": 2,
            "is_support_3": False,
        },
        {
            "trial_index": 2,
            "lambda1": 1.0,
            "lambda2": None,
            "lambda3": None,
            "ratio2_to_1": None,
            "ratio3_to_1": None,
            "dominant_gap": None,
            "support_size": 3,
            "is_support_3": True,
        },
    ]

    summary, hypothesis_analysis = _summarize_trials(trials)

    assert summary["support3_rate_gap_ge_2"] == 1.0
    assert summary["support3_rate_gap_lt_2"] == 0.0
    assert summary["delta_support3_rate"] == 1.0
    assert hypothesis_analysis["overall_support3_rate"] == 0.5


def test_build_hypothesis_analysis_gap_bins() -> None:
    trials = [
        {"dominant_gap": 1.1, "support_size": 2, "is_support_3": False},
        {"dominant_gap": 1.25, "support_size": 3, "is_support_3": True},
        {"dominant_gap": 1.7, "support_size": 4, "is_support_3": False},
        {"dominant_gap": 2.5, "support_size": 3, "is_support_3": True},
        {"dominant_gap": 3.2, "support_size": 3, "is_support_3": True},
        {"dominant_gap": None, "support_size": 3, "is_support_3": True},
    ]

    analysis = _build_hypothesis_analysis(trials)

    bins = {entry["bin"]: entry for entry in analysis["gap_bins"]}
    assert bins["[1.0,1.2)"]["count"] == 1
    assert bins["[1.2,1.5)"]["count"] == 1
    assert bins["[1.5,2.0)"]["count"] == 1
    assert bins["[2.0,3.0)"]["count"] == 1
    assert bins["[3.0,inf)"]["count"] == 1
    assert analysis["overall_support3_rate"] == 0.6


def test_write_trials_csv_columns(tmp_path) -> None:
    output_csv = tmp_path / "team_strict_spectrum_trials.csv"
    _write_trials_csv(
        output_csv,
        [
            {
                "trial_index": 7,
                "lambda1": 3.0,
                "lambda2": 2.0,
                "lambda3": 1.0,
                "ratio2_to_1": 2 / 3,
                "ratio3_to_1": 1 / 3,
                "dominant_gap": 1.5,
                "support_size": 3,
                "is_support_3": True,
            }
        ],
    )

    lines = output_csv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "trial,lambda1,lambda2,lambda3,ratio2_to_1,ratio3_to_1,dominant_gap,support_size,is_support_3"
    assert lines[1].startswith("7,3.0,2.0,1.0,")
