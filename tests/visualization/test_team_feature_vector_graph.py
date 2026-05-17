from monocycle_nash.domain.team import TeamFeatureVector
from monocycle_nash.domain.visualization.team_feature_vector_graph import TeamFeatureVectorScatterPlotter


def test_draw_creates_team_feature_vector_scatter_svg(tmp_path) -> None:
    plotter = TeamFeatureVectorScatterPlotter(
        labels=["team_i", "team_j"],
        vectors=[TeamFeatureVector(1.5, -0.5), TeamFeatureVector(-2.0, 3.0)],
    )

    output = tmp_path / "team_feature_vectors.svg"
    saved = plotter.draw(output)

    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "team_i" in content
    assert "team_j" in content
    assert "原点 (0, 0)" in content
    assert "marker-end" not in content


def test_scatter_plotter_requires_non_empty_labels_and_vectors() -> None:
    try:
        TeamFeatureVectorScatterPlotter(labels=[], vectors=[])
        assert False, "ValueError が発生するべき"
    except ValueError as exc:
        assert "at least one" in str(exc)
