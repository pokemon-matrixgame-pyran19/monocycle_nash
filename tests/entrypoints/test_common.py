from __future__ import annotations

from pathlib import Path

import pytest

from monocycle_nash.entrypoints.common import build_matrix, load_inputs


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_load_inputs_from_run_config(tmp_path: Path) -> None:
    _write(
        tmp_path / "data" / "run_config" / "baseline" / "rps3_graph.toml",
        '''
        matrix = "rps3"
        graph = "payoff/default"
        setting = "local"
        ''',
    )
    _write(
        tmp_path / "data" / "matrix" / "rps3" / "data.toml",
        '''
        matrix = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
        ''',
    )
    _write(
        tmp_path / "data" / "graph" / "payoff" / "default" / "data.toml",
        '''
        threshold = 0.0
        canvas_size = 840
        ''',
    )
    _write(
        tmp_path / "data" / "setting" / "local.toml",
        '''
        [runmeta]
        sqlite_path = ".runmeta/run_history.db"

        [output]
        base_dir = "result"
        ''',
    )

    matrix, graph, setting, run_cfg = load_inputs("baseline/rps3_graph", tmp_path / "data", require_graph=True)

    assert matrix["matrix"][0] == [0, 1, -1]
    assert graph is not None
    assert graph["canvas_size"] == 840
    assert setting["output"]["base_dir"] == "result"
    assert run_cfg.name == "rps3_graph.toml"


def test_build_matrix_requires_exclusive_matrix_or_characters() -> None:
    with pytest.raises(ValueError, match="どちらか片方"):
        build_matrix({"matrix": [[0.0]], "characters": [{"label": "a", "p": 1.0, "v": [0.0, 1.0]}]})


def test_build_matrix_from_characters_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="重複"):
        build_matrix(
            {
                "characters": [
                    {"label": "a", "p": 1.0, "v": [0.0, 1.0]},
                    {"label": "a", "p": 2.0, "v": [1.0, 0.0]},
                ]
            }
        )
