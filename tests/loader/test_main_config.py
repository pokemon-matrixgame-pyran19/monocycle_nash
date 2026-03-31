from __future__ import annotations

from pathlib import Path

import pytest

from monocycle_nash.loader.main_config import MainConfigLoader


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

        [configs]
        graph_payoff = "graph_payoff.toml"
        ''',
    )
    _write(
        data_dir / "run_config" / "graph_payoff.toml",
        '''
        matrix = "janken"
        setting = "local"
        graph = "default"
        ''',
    )

    loader = MainConfigLoader(data_dir / "run_config" / "main.toml")
    config_path = loader.resolve_feature_config_path("graph_payoff")

    assert config_path == data_dir / "run_config" / "graph_payoff.toml"


def test_main_config_loader_requires_declared_feature_section(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(data_dir / "run_config" / "main.toml", 'features = ["solve_payoff"]\n[configs]\n')

    loader = MainConfigLoader(data_dir / "run_config" / "main.toml")
    with pytest.raises(ValueError, match="main_config.configs.solve_payoff"):
        loader.resolve_feature_config_path("solve_payoff")


def test_main_config_loader_exposes_data_root(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(data_dir / "run_config" / "main.toml", 'features = ["solve_payoff"]')
    loader = MainConfigLoader(data_dir / "run_config" / "main.toml")

    assert loader.main_config_path == data_dir / "run_config" / "main.toml"
    assert loader.data_root == data_dir


def test_main_config_loader_builds_feature_run_plans(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(
        data_dir / "run_config" / "main.toml",
        '''
        features = ["solve_payoff", "graph_payoff"]

        [configs]
        solve_payoff = "solve_payoff.toml"
        graph_payoff = "graph_payoff.toml"
        ''',
    )
    _write(data_dir / "run_config" / "solve_payoff.toml", 'matrix = "m"\nsetting = "local"')
    _write(data_dir / "run_config" / "graph_payoff.toml", 'matrix = "m"\nsetting = "local"\ngraph = "default"')

    loader = MainConfigLoader(data_dir / "run_config" / "main.toml")
    plans = loader.load_feature_run_plans()

    assert [x.feature for x in plans] == ["solve_payoff", "graph_payoff"]
    assert [x.config_path for x in plans] == [
        data_dir / "run_config" / "solve_payoff.toml",
        data_dir / "run_config" / "graph_payoff.toml",
    ]
