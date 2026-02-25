from __future__ import annotations

import json
from pathlib import Path

from monocycle_nash.approximation.compare_random_approximation_app import run
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


def test_compare_random_approximation_writes_summary_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(
        data_dir / "run_config" / "main.toml",
        '''
        features = ["compare_random_approximation"]

        [shared]
        matrix = "base"
        setting = "local"

        [compare_random_approximation]
        approximation = "default"
        random_matrix = "r4"
        ''',
    )
    _write(data_dir / "matrix" / "base" / "data.toml", 'matrix = [[0, 1], [-1, 0]]')
    _write(
        data_dir / "approximation" / "default" / "data.toml",
        '''
        [approxmation]
        approximation = "MonocycleToGeneralApproximation"
        distance = "MaxElementDifferenceDistance"
        ''',
    )
    _write(
        data_dir / "random_matrix" / "r4" / "data.toml",
        '''
        size = 4
        generation_count = 8
        acceptance_condition = "even_size"
        random_seed = 123
        ''',
    )
    _write_setting(data_dir, tmp_path)

    code = run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    assert code == 0

    run_dirs = [x for x in (tmp_path / "result").iterdir() if x.is_dir()]
    assert len(run_dirs) == 1

    result_path = run_dirs[0] / "output" / "random_approximation_quality.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload["generation_count"] == 8
    assert payload["random_matrix"]["acceptance_condition"] == "even_size"
    assert payload["quality"]["min"] == 0.0
    assert payload["quality"]["max"] == 0.0
    assert payload["quality"]["mean"] == 0.0


def test_compare_random_approximation_returns_failure_for_invalid_condition(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(
        data_dir / "run_config" / "main.toml",
        '''
        features = ["compare_random_approximation"]

        [shared]
        matrix = "base"
        setting = "local"

        [compare_random_approximation]
        approximation = "default"
        random_matrix = "invalid"
        ''',
    )
    _write(data_dir / "matrix" / "base" / "data.toml", 'matrix = [[0, 1], [-1, 0]]')
    _write(
        data_dir / "approximation" / "default" / "data.toml",
        '''
        approximation = "MonocycleToGeneralApproximation"
        distance = "MaxElementDifferenceDistance"
        ''',
    )
    _write(
        data_dir / "random_matrix" / "invalid" / "data.toml",
        '''
        size = 3
        acceptance_condition = "unknown_keyword"
        ''',
    )
    _write_setting(data_dir, tmp_path)

    code = run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    assert code == 1
