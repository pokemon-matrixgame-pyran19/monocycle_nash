from __future__ import annotations

from pathlib import Path

import pytest
import monocycle_nash.entrypoints.common as common_mod

from monocycle_nash.entrypoints.common import (
    build_characters,
    build_matrix,
    load_inputs,
    prepare_run_session,
    write_input_snapshots,
)


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


def test_write_input_snapshots_saves_resolved_split_toml_files(tmp_path: Path) -> None:
    output_base = tmp_path / "result"
    setting = {
        "runmeta": {"sqlite_path": str(tmp_path / ".runmeta" / "run_history.db")},
        "output": {"base_dir": str(output_base)},
    }
    matrix_data = {
        "matrix": [[0.0, 1.0], [-1.0, 0.0]],
        "labels": ["rock", "paper"],
    }
    graph_data = {"threshold": 0.1, "canvas_size": 900}
    setting_data = {
        "runmeta": {"sqlite_path": "db.sqlite"},
        "output": {"base_dir": "result"},
    }

    service, ctx, conn = prepare_run_session(setting, "python -m monocycle_nash.entrypoints.graph_payoff --run-config x")
    try:
        write_input_snapshots(
            service,
            ctx.run_id,
            matrix_data=matrix_data,
            graph_data=graph_data,
            setting_data=setting_data,
        )
    finally:
        conn.close()

    input_dir = output_base / str(ctx.run_id) / "input"
    assert (input_dir / "matrix.toml").read_text(encoding="utf-8") == (
        'matrix = [[0.0, 1.0], [-1.0, 0.0]]\nlabels = ["rock", "paper"]\n'
    )
    assert (input_dir / "graph.toml").read_text(encoding="utf-8") == (
        "threshold = 0.1\ncanvas_size = 900\n"
    )
    assert (input_dir / "setting.toml").read_text(encoding="utf-8") == (
        '[runmeta]\nsqlite_path = "db.sqlite"\n\n[output]\nbase_dir = "result"\n'
    )


def test_write_input_snapshots_skips_graph_file_when_not_provided(tmp_path: Path) -> None:
    output_base = tmp_path / "result"
    setting = {
        "runmeta": {"sqlite_path": str(tmp_path / ".runmeta" / "run_history.db")},
        "output": {"base_dir": str(output_base)},
    }

    service, ctx, conn = prepare_run_session(setting, "python -m monocycle_nash.entrypoints.solve_payoff --run-config x")
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


def test_prepare_run_session_uses_analysis_project_for_runmeta_linkage(tmp_path: Path) -> None:
    output_base = tmp_path / "result"
    project_root = tmp_path / "analysis-main"
    setting = {
        "runmeta": {"sqlite_path": str(tmp_path / ".runmeta" / "run_history.db")},
        "output": {"base_dir": str(output_base)},
        "analysis_project": {
            "project_id": "analysis-main",
            "project_path": str(project_root),
        },
    }

    service, ctx, conn = prepare_run_session(setting, "python -m monocycle_nash.entrypoints.solve_payoff --run-config x")
    try:
        run = service.runs_repository.find_by_id(ctx.run_id)
    finally:
        conn.close()

    assert run is not None
    assert run.project_id == "analysis-main"
    assert run.project_path == str(project_root)

    refs_dir = project_root / "experiment_refs"
    symlink_path = refs_dir / str(ctx.run_id)
    txt_path = refs_dir / f"{ctx.run_id}.txt"
    assert symlink_path.exists() or txt_path.exists()


def test_prepare_run_session_updates_existing_project_path(tmp_path: Path) -> None:
    output_base = tmp_path / "result"
    old_project_root = tmp_path / "old-analysis"
    new_project_root = tmp_path / "new-analysis"
    db_path = tmp_path / ".runmeta" / "run_history.db"

    first_setting = {
        "runmeta": {"sqlite_path": str(db_path)},
        "output": {"base_dir": str(output_base)},
        "analysis_project": {
            "project_id": "analysis-main",
            "project_path": str(old_project_root),
        },
    }
    second_setting = {
        "runmeta": {"sqlite_path": str(db_path)},
        "output": {"base_dir": str(output_base)},
        "analysis_project": {
            "project_id": "analysis-main",
            "project_path": str(new_project_root),
        },
    }

    _, first_ctx, first_conn = prepare_run_session(
        first_setting,
        "python -m monocycle_nash.entrypoints.solve_payoff --run-config first",
    )
    first_conn.close()

    service, second_ctx, second_conn = prepare_run_session(
        second_setting,
        "python -m monocycle_nash.entrypoints.solve_payoff --run-config second",
    )
    try:
        project = service.runs_repository.conn.execute(
            "SELECT project_path FROM projects WHERE project_id = ?",
            ("analysis-main",),
        ).fetchone()
        run = service.runs_repository.find_by_id(second_ctx.run_id)
    finally:
        second_conn.close()

    assert project is not None
    assert project["project_path"] == str(new_project_root)
    assert run is not None
    assert run.project_path == str(new_project_root)

    assert (new_project_root / "experiment_refs" / str(second_ctx.run_id)).exists() or (
        new_project_root / "experiment_refs" / f"{second_ctx.run_id}.txt"
    ).exists()


def test_prepare_run_session_writes_txt_when_symlink_and_junction_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_base = tmp_path / "result"
    project_root = tmp_path / "analysis-main"
    setting = {
        "runmeta": {"sqlite_path": str(tmp_path / ".runmeta" / "run_history.db")},
        "output": {"base_dir": str(output_base)},
        "analysis_project": {
            "project_id": "analysis-main",
            "project_path": str(project_root),
        },
    }

    def _raise_symlink(self: Path, target: Path, target_is_directory: bool = False) -> None:
        raise OSError("symlink disabled")

    monkeypatch.setattr(Path, "symlink_to", _raise_symlink)
    monkeypatch.setattr(common_mod, "_try_create_windows_junction", lambda **_: False)

    _, ctx, conn = prepare_run_session(setting, "python -m monocycle_nash.entrypoints.solve_payoff --run-config txt")
    conn.close()

    txt_path = project_root / "experiment_refs" / f"{ctx.run_id}.txt"
    assert txt_path.exists()
    body = txt_path.read_text(encoding="utf-8")
    assert "result_path=" in body
    assert "status=running" in body


def test_build_matrix_team_mode_with_empty_string_uses_input_matrix_as_is() -> None:
    matrix = build_matrix(
        {
            "matrix": [[0.0, 1.0], [-1.0, 0.0]],
            "team": "",
            "labels": ["A", "B"],
        }
    )

    assert matrix.labels == ["A", "B"]
    assert matrix.matrix[0, 1] == 1.0


def test_build_matrix_team_mode_generates_default_team_payoff_matrix() -> None:
    matrix = build_matrix(
        {
            "matrix": [
                [0.0, 2.0, -1.0],
                [-2.0, 0.0, 3.0],
                [1.0, -3.0, 0.0],
            ],
            "labels": ["A", "B", "C"],
            "team": "2by2",
        }
    )

    assert matrix.matrix.shape == (3, 3)
    assert matrix.labels == ["A+B", "A+C", "B+C"]


def test_build_matrix_team_mode_accepts_explicit_teams() -> None:
    matrix = build_matrix(
        {
            "matrix": [
                [0.0, 1.0, -1.0],
                [-1.0, 0.0, 2.0],
                [1.0, -2.0, 0.0],
            ],
            "labels": ["A", "B", "C"],
            "team": "strict",
            "teams": [
                {"label": "AB", "members": ["A", "B"]},
                {"label": "AC", "members": ["A", "C"]},
            ],
        }
    )

    assert matrix.labels == ["AB", "AC"]
    assert matrix.matrix.shape == (2, 2)


def test_build_matrix_rejects_unknown_team_mode() -> None:
    with pytest.raises(ValueError, match='team は "", "strict", "2by2", "monocycle"'):
        build_matrix({"matrix": [[0.0, 1.0], [-1.0, 0.0]], "team": "fast"})


def test_build_matrix_rejects_teams_without_team_mode() -> None:
    with pytest.raises(ValueError, match="team モード"):
        build_matrix(
            {
                "matrix": [[0.0, 1.0], [-1.0, 0.0]],
                "teams": [{"label": "AB", "members": [0, 1]}],
            }
        )
