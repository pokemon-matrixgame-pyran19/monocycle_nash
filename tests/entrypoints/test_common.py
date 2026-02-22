from __future__ import annotations

from pathlib import Path

import pytest

from monocycle_nash.application_support import build_characters, build_matrix, prepare_run_session, write_input_snapshots
from monocycle_nash.loader import RunConfigLoader


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_run_config_loader_loads_feature_inputs_from_main_toml(tmp_path: Path) -> None:
    _write(
        tmp_path / "data" / "run_config" / "main.toml",
        '''
        features = ["graph_payoff"]

        [shared]
        matrix = "rps3"
        setting = "local"

        [graph_payoff]
        graph = "payoff/default"
        ''',
    )
    _write(tmp_path / "data" / "matrix" / "rps3" / "data.toml", 'matrix = [[0, 1], [-1, 0]]')
    _write(tmp_path / "data" / "graph" / "payoff" / "default" / "data.toml", 'canvas_size = 840')
    _write(tmp_path / "data" / "setting" / "local.toml", '[output]\nbase_dir = "result"')

    loader = RunConfigLoader(data_dir=tmp_path / "data")
    matrix, graph, setting = loader.load_feature_inputs(
        "graph_payoff",
        requires_graph=True,
        config_path=tmp_path / "data" / "run_config" / "main.toml",
    )

    assert matrix["matrix"][0] == [0, 1]
    assert graph is not None
    assert graph["canvas_size"] == 840
    assert setting["output"]["base_dir"] == "result"


def test_run_config_loader_allows_empty_graph_for_solve(tmp_path: Path) -> None:
    _write(
        tmp_path / "data" / "run_config" / "main.toml",
        '''
        features = ["solve_payoff"]

        [shared]
        matrix = "rps3"
        setting = "local"

        [solve_payoff]
        graph = ""
        ''',
    )
    _write(tmp_path / "data" / "matrix" / "rps3" / "data.toml", 'matrix = [[0, 1], [-1, 0]]')
    _write(tmp_path / "data" / "setting" / "local.toml", '[output]\nbase_dir = "result"')

    loader = RunConfigLoader(data_dir=tmp_path / "data")
    refs = loader.load_feature_refs("solve_payoff", config_path=tmp_path / "data" / "run_config" / "main.toml")
    assert refs.graph is None


def test_build_matrix_requires_exclusive_matrix_or_characters() -> None:
    with pytest.raises(ValueError, match="どちらか片方"):
        build_matrix({"matrix": [[0.0]], "characters": [{"label": "a", "p": 1.0, "v": [0.0, 1.0]}]})


def test_build_characters_rejects_non_numeric_vector() -> None:
    with pytest.raises(ValueError):
        build_characters({"characters": [{"label": "a", "p": 1.0, "v": ["x", 0.0]}]})


def test_prepare_run_session_creates_output_base_dir_run_folder(tmp_path: Path) -> None:
    output_base = tmp_path / "custom_result"
    setting = {
        "runmeta": {"sqlite_path": str(tmp_path / ".runmeta" / "run_history.db")},
        "output": {"base_dir": str(output_base)},
    }

    service, ctx, conn = prepare_run_session(setting, "uv run main")
    conn.close()

    run_dir = output_base / str(ctx.run_id)
    assert run_dir.exists()
    assert (run_dir / "input").is_dir()
    assert (run_dir / "output").is_dir()
    assert (run_dir / "logs").is_dir()


def test_write_input_snapshots_skips_graph_file_when_not_provided(tmp_path: Path) -> None:
    output_base = tmp_path / "result"
    setting = {
        "runmeta": {"sqlite_path": str(tmp_path / ".runmeta" / "run_history.db")},
        "output": {"base_dir": str(output_base)},
    }

    service, ctx, conn = prepare_run_session(setting, "uv run main")
    try:
        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data={"matrix": [[0.0]]},
            graph_data=None,
            setting_data={"output": {"base_dir": "result"}},
        )
    finally:
        conn.close()

    input_dir = output_base / str(ctx.run_id) / "input"
    assert (input_dir / "matrix.toml").exists()
    assert not (input_dir / "graph.toml").exists()
    assert (input_dir / "setting.toml").exists()
