from __future__ import annotations

import json
from pathlib import Path

from monocycle_nash.analysis.app.experiment_team_strict_spectrum import run
from monocycle_nash.runtime.infra.loader.main_config import MainConfigLoader


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


def _write_main_config(data_dir: Path, *, experiment_name: str) -> None:
    _write(
        data_dir / "run_config" / "main.toml",
        f'''
        features = ["experiment_team_strict_spectrum"]

        [shared]
        setting = "local"
        matrix = "base"

        [experiment_team_strict_spectrum]
        experiment = "{experiment_name}"
        ''',
    )


def _write_experiment(
    data_dir: Path,
    *,
    name: str,
    generation_count: int = 5,
    power_low: float = -1.0,
    power_high: float = 1.0,
    vector_low: float = -1.0,
    vector_high: float = 1.0,
) -> None:
    _write(
        data_dir / "experiment" / "team_strict_spectrum" / name / "data.toml",
        f'''
        generation_count = {generation_count}
        power_low = {power_low}
        power_high = {power_high}
        vector_low = {vector_low}
        vector_high = {vector_high}
        random_seed = 123
        ''',
    )


def test_experiment_team_strict_spectrum_writes_result_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_main_config(data_dir, experiment_name="mini")
    _write_experiment(data_dir, name="mini", generation_count=5)
    _write_setting(data_dir, tmp_path)

    code = run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    assert code == 0

    run_dirs = [x for x in (tmp_path / "result").iterdir() if x.is_dir()]
    assert len(run_dirs) == 1

    result_path = run_dirs[0] / "output" / "team_strict_spectrum_experiment.json"
    assert result_path.exists()

    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert len(payload["trials"]) == 5
    for trial in payload["trials"]:
        assert "ratio2_to_1" in trial
        assert "ratio3_to_1" in trial
        assert "support_size" in trial

    assert "support_size_histogram" in payload["summary"]
    assert "support_size_le_3_rate" in payload["summary"]


def test_experiment_team_strict_spectrum_fails_when_generation_count_is_non_positive(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_main_config(data_dir, experiment_name="invalid_count")
    _write_experiment(data_dir, name="invalid_count", generation_count=0)
    _write_setting(data_dir, tmp_path)

    code = run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    assert code == 1


def test_experiment_team_strict_spectrum_fails_for_invalid_power_range(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_main_config(data_dir, experiment_name="invalid_power")
    _write_experiment(
        data_dir,
        name="invalid_power",
        generation_count=5,
        power_low=1.0,
        power_high=1.0,
    )
    _write_setting(data_dir, tmp_path)

    code = run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    assert code == 1


def test_experiment_team_strict_spectrum_fails_for_invalid_vector_range(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_main_config(data_dir, experiment_name="invalid_vector")
    _write_experiment(
        data_dir,
        name="invalid_vector",
        generation_count=5,
        vector_low=0.5,
        vector_high=0.5,
    )
    _write_setting(data_dir, tmp_path)

    code = run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    assert code == 1
