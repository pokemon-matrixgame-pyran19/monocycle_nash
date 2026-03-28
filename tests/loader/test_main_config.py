from __future__ import annotations

from pathlib import Path

import pytest

from monocycle_nash.runtime.infra.loader.main_config import MainConfigLoader


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_main_config_loader_load_features(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(data_dir / "run_config" / "main.toml", 'features = ["graph_payoff", "solve_payoff"]')

    loader = MainConfigLoader(data_dir / "run_config" / "main.toml")
    assert loader.load_features() == ["graph_payoff", "solve_payoff"]


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
        matrix = "janken"
        ''',
    )

    loader = MainConfigLoader(data_dir / "run_config" / "main.toml")
    merged = loader.load_feature_config("graph_payoff")

    assert merged["matrix"] == "janken"
    assert merged["setting"] == "local"
    assert merged["graph"] == "default"


def test_main_config_loader_requires_declared_feature_section(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(data_dir / "run_config" / "main.toml", 'features = ["solve_payoff"]\n[shared]\nmatrix = "rps3"\nsetting = "local"')

    loader = MainConfigLoader(data_dir / "run_config" / "main.toml")
    with pytest.raises(ValueError, match="main_config.solve_payoff"):
        loader.load_feature_config("solve_payoff")


def test_main_config_loader_exposes_data_root(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(data_dir / "run_config" / "main.toml", 'features = ["solve_payoff"]')
    loader = MainConfigLoader(data_dir / "run_config" / "main.toml")

    assert loader.main_config_path == data_dir / "run_config" / "main.toml"
    assert loader.data_root == data_dir
