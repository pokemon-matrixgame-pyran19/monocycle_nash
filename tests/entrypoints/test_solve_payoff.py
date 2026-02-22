from __future__ import annotations

import json
from pathlib import Path

from monocycle_nash.equilibrium.solve_payoff_app import run
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


def test_solve_payoff_outputs_eigenvalues_for_alternating_matrix(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(
        data_dir / "run_config" / "baseline" / "rps3_solve.toml",
        '''
        matrix = "rps3"
        setting = "local"
        ''',
    )
    _write(
        data_dir / "matrix" / "rps3" / "data.toml",
        '''
        matrix = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
        labels = ["rock", "paper", "scissors"]
        ''',
    )
    _write_setting(data_dir, tmp_path)

    (data_dir / "run_config" / "main.toml").write_text("features=[\"solve_payoff\"]\n[shared]\nmatrix=\"rps3\"\nsetting=\"local\"\n[solve_payoff]\n", encoding="utf-8")
    code = run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    assert code == 0

    run_dirs = [x for x in (tmp_path / "result").iterdir() if x.is_dir()]
    assert len(run_dirs) == 1

    eigenvalues_path = run_dirs[0] / "output" / "eigenvalues.json"
    assert eigenvalues_path.exists()

    payload = json.loads(eigenvalues_path.read_text(encoding="utf-8"))
    assert "eigenvalues" in payload
    values = payload["eigenvalues"]
    assert len(values) == 3
    assert all(isinstance(v, float) for v in values)
    assert all(v >= 0.0 for v in values)


def test_solve_payoff_skips_eigenvalues_for_non_alternating_matrix(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write(
        data_dir / "run_config" / "baseline" / "non_alt_solve.toml",
        '''
        matrix = "non_alt"
        setting = "local"
        ''',
    )
    _write(
        data_dir / "matrix" / "non_alt" / "data.toml",
        '''
        matrix = [[0, 1], [0, 0]]
        labels = ["a", "b"]
        ''',
    )
    _write_setting(data_dir, tmp_path)

    (data_dir / "run_config" / "main.toml").write_text("features=[\"solve_payoff\"]\n[shared]\nmatrix=\"non_alt\"\nsetting=\"local\"\n[solve_payoff]\n", encoding="utf-8")
    code = run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    assert code == 0

    run_dirs = [x for x in (tmp_path / "result").iterdir() if x.is_dir()]
    assert len(run_dirs) == 1

    eigenvalues_path = run_dirs[0] / "output" / "eigenvalues.json"
    assert not eigenvalues_path.exists()
