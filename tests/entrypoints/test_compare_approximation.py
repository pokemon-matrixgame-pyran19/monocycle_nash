from __future__ import annotations

import json
from pathlib import Path

from monocycle_nash.approximation.compare_approximation_app import run
from monocycle_nash.loader.main_config import MainConfigLoader


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _write_setting(data_dir: Path, tmp_path: Path) -> None:
    _write(
        data_dir / "setting" / "local.toml",
        f'''
        [runmeta]
        sqlite_path = "{(tmp_path / '.runmeta' / 'run_history.db').as_posix()}"

        [output]
        base_dir = "{(tmp_path / 'result').as_posix()}"
        ''',
    )


def test_compare_approximation_writes_quality_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(
        data_dir / "run_config" / "main.toml",
        '''
        features = ["compare_approximation"]

        [shared]
        matrix = "source_chars"
        setting = "local"

        [compare_approximation]
        approximation = "default"
        ''',
    )
    _write(
        data_dir / "matrix" / "source_chars" / "data.toml",
        '''
        characters = [
          { label = "rock", p = 0.5, v = [1.0, 0.0] },
          { label = "paper", p = 0.5, v = [-0.5, 0.8660254] },
          { label = "scissors", p = 0.5, v = [-0.5, -0.8660254] }
        ]
        ''',
    )
    _write(
        data_dir / "approximation" / "default" / "data.toml",
        '''
        source_matrix = "source_chars"
        reference_matrix = "source_chars"

        [approxmation]
        approximation = "MonocycleToGeneralApproximation"
        distance = "MaxElementDifferenceDistance"
        ''',
    )
    _write_setting(data_dir, tmp_path)

    code = run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    assert code == 0

    run_dirs = [x for x in (tmp_path / "result").iterdir() if x.is_dir()]
    assert len(run_dirs) == 1

    result_path = run_dirs[0] / "output" / "approximation_quality.json"
    assert result_path.exists()

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["approximation"] == "MonocycleToGeneralApproximation"
    assert payload["distance"] == "MaxElementDifferenceDistance"
    assert payload["quality"] == 0.0


def test_compare_approximation_returns_failure_code_for_invalid_config(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(
        data_dir / "run_config" / "main.toml",
        '''
        features = ["compare_approximation"]

        [shared]
        matrix = "source_chars"
        setting = "local"

        [compare_approximation]
        approximation = "default"
        ''',
    )
    _write(
        data_dir / "matrix" / "source_chars" / "data.toml",
        '''
        characters = [
          { label = "a", p = 1.0, v = [1.0, 0.0] },
          { label = "b", p = 1.0, v = [0.0, 1.0] },
          { label = "c", p = 1.0, v = [-1.0, 0.0] }
        ]
        ''',
    )
    _write(
        data_dir / "approximation" / "default" / "data.toml",
        '''
        source_matrix = "source_chars"
        reference_matrix = "source_chars"
        distance = "UnsupportedDistance"
        ''',
    )
    _write_setting(data_dir, tmp_path)

    code = run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    assert code == 1


def test_compare_approximation_supports_equilibrium_u_strategy_distance(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(
        data_dir / "run_config" / "main.toml",
        '''
        features = ["compare_approximation"]

        [shared]
        matrix = "source_matrix"
        setting = "local"

        [compare_approximation]
        approximation = "default"
        ''',
    )
    _write(
        data_dir / "matrix" / "source_matrix" / "data.toml",
        '''
        matrix = [
          [0.0, 2.0],
          [-2.0, 0.0],
        ]
        ''',
    )
    _write(
        data_dir / "matrix" / "reference_matrix" / "data.toml",
        '''
        matrix = [
          [0.0, 1.0],
          [-1.0, 0.0],
        ]
        ''',
    )
    _write(
        data_dir / "approximation" / "default" / "data.toml",
        '''
        source_matrix = "source_matrix"
        reference_matrix = "reference_matrix"

        [approxmation]
        approximation = "MonocycleToGeneralApproximation"
        distance = "EquilibriumUStrategyDifferenceDistance"
        ''',
    )
    _write_setting(data_dir, tmp_path)

    code = run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    assert code == 0

    run_dirs = [x for x in (tmp_path / "result").iterdir() if x.is_dir()]
    assert len(run_dirs) == 1

    result_path = run_dirs[0] / "output" / "approximation_quality.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload["distance"] == "EquilibriumUStrategyDifferenceDistance"
    assert payload["quality"] == 1.0
