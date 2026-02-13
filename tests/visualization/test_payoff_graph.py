import numpy as np

from monocycle_nash.visualization import PayoffDirectedGraphPlotter


def test_extract_edges_respects_positive_and_threshold() -> None:
    matrix = np.array(
        [
            [0.0, 0.4, -0.2],
            [-0.4, 0.0, 0.9],
            [0.2, -0.9, 0.0],
        ]
    )
    labels = ["A", "B", "C"]
    plotter = PayoffDirectedGraphPlotter(matrix, labels, threshold=0.3)

    edges = plotter.extract_edges()

    assert [(e.source, e.target, e.value) for e in edges] == [
        (0, 1, 0.4),
        (1, 2, 0.9),
    ]


def test_draw_creates_svg_file(tmp_path) -> None:
    matrix = np.array(
        [
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 0.5],
            [0.3, 0.0, 0.0],
        ]
    )
    labels = ["キャラA", "キャラB", "キャラC"]
    plotter = PayoffDirectedGraphPlotter(matrix, labels, threshold=0.2)

    output = tmp_path / "payoff_graph.svg"
    saved_path = plotter.draw(output)

    assert saved_path.exists()
    assert saved_path.suffix == ".svg"
    content = saved_path.read_text(encoding="utf-8")
    assert "キャラA" in content
    assert "0.80" in content
