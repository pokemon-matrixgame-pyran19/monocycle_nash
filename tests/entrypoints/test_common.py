from __future__ import annotations

from pathlib import Path

import pytest

from monocycle_nash.entrypoints.common import build_characters, build_matrix, load_inputs, prepare_run_session


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


def test_build_characters_rejects_non_numeric_vector() -> None:
    with pytest.raises(ValueError, match="数値配列"):
        build_characters({"characters": [{"label": "a", "p": 1.0, "v": ["x", 0.0]}]})


def test_load_inputs_requires_graph_when_entrypoint_needs_it(tmp_path: Path) -> None:
    _write(
        tmp_path / "data" / "run_config" / "baseline" / "rps3_solve.toml",
        '''
        matrix = "rps3"
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
        tmp_path / "data" / "setting" / "local.toml",
        '''
        [output]
        base_dir = "result"
        ''',
    )

    with pytest.raises(ValueError, match="run_config.graph"):
        load_inputs("baseline/rps3_solve", tmp_path / "data", require_graph=True)


def test_load_inputs_rejects_empty_run_config_reference(tmp_path: Path) -> None:
    _write(
        tmp_path / "data" / "run_config" / "baseline" / "invalid.toml",
        '''
        matrix = ""
        setting = "local"
        ''',
    )

    with pytest.raises(ValueError, match="空でない文字列"):
        load_inputs("baseline/invalid", tmp_path / "data", require_graph=False)


def test_load_inputs_rejects_invalid_graph_type(tmp_path: Path) -> None:
    _write(
        tmp_path / "data" / "run_config" / "baseline" / "invalid_graph.toml",
        '''
        matrix = "rps3"
        graph = "payoff/default"
        setting = "local"
        ''',
    )
    _write(tmp_path / "data" / "matrix" / "rps3" / "data.toml", 'matrix = [[0, 1], [-1, 0]]')
    _write(tmp_path / "data" / "graph" / "payoff" / "default" / "data.toml", 'canvas_size = "large"')
    _write(tmp_path / "data" / "setting" / "local.toml", '[output]\nbase_dir = "result"')

    with pytest.raises(ValueError, match="graph.canvas_size"):
        load_inputs("baseline/invalid_graph", tmp_path / "data", require_graph=True)


def test_load_inputs_rejects_invalid_setting_type(tmp_path: Path) -> None:
    _write(
        tmp_path / "data" / "run_config" / "baseline" / "invalid_setting.toml",
        '''
        matrix = "rps3"
        setting = "local"
        ''',
    )
    _write(tmp_path / "data" / "matrix" / "rps3" / "data.toml", 'matrix = [[0, 1], [-1, 0]]')
    _write(tmp_path / "data" / "setting" / "local.toml", '[output]\nbase_dir = 123')

    with pytest.raises(ValueError, match="setting.output.base_dir"):
        load_inputs("baseline/invalid_setting", tmp_path / "data", require_graph=False)


def test_prepare_run_session_creates_output_base_dir_run_folder(tmp_path: Path) -> None:
    output_base = tmp_path / "custom_result"
    setting = {
        "runmeta": {"sqlite_path": str(tmp_path / ".runmeta" / "run_history.db")},
        "output": {"base_dir": str(output_base)},
    }

    service, ctx, conn = prepare_run_session(setting, "python -m monocycle_nash.entrypoints.solve_payoff --run-config x")
    conn.close()

    run_dir = output_base / str(ctx.run_id)
    assert run_dir.exists()
    assert (run_dir / "input").is_dir()
    assert (run_dir / "output").is_dir()
    assert (run_dir / "logs").is_dir()
