from __future__ import annotations

from pathlib import Path

import pytest

from monocycle_nash.loader.main_config import MainConfigLoader


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_main_config_loader_resolves_shared_and_feature_override(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(
        data_dir / "run_config" / "main.toml",
        '''
        features = ["graph_payoff"]

        [shared]
        matrix = "rps3"
        setting = "local"

        [graph_payoff]
        graph = "default"
        ''',
    )
    _write(data_dir / "matrix" / "rps3" / "data.toml", 'matrix = [[0, 1], [-1, 0]]')
    _write(data_dir / "graph" / "default" / "data.toml", '[payoff]\ncanvas_size = 840')
    _write(data_dir / "setting" / "local.toml", '[output]\nbase_dir = "result"')

    loader = MainConfigLoader(data_dir / "run_config" / "main.toml")
    loaded = loader.load_inputs_for_feature("graph_payoff", graph_section="payoff")

    assert loader.load_features() == ["graph_payoff"]
    assert loaded.matrix_data["matrix"] == [[0, 1], [-1, 0]]
    assert loaded.graph_data is not None
    assert loaded.graph_data["canvas_size"] == 840


def test_main_config_loader_requires_declared_feature_section(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(data_dir / "run_config" / "main.toml", 'features = ["solve_payoff"]\n[shared]\nmatrix = "rps3"\nsetting = "local"')

    loader = MainConfigLoader(data_dir / "run_config" / "main.toml")
    with pytest.raises(ValueError, match="main_config.solve_payoff"):
        loader.load_inputs_for_feature("solve_payoff")
