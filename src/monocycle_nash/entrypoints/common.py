from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from monocycle_nash.character import Character, MatchupVector
from monocycle_nash.loader import ExperimentDataLoader, SettingDataLoader, TomlTreeLoader
from monocycle_nash.matrix import PayoffMatrixBuilder
from monocycle_nash.matrix.base import PayoffMatrix
from monocycle_nash.runmeta.artifact_store import RunArtifactStore
from monocycle_nash.runmeta.clock import now_jst_iso
from monocycle_nash.runmeta.db import SQLiteConnectionFactory, migrate
from monocycle_nash.runmeta.repositories import ProjectsRepository, RunsRepository
from monocycle_nash.runmeta.service import RunSessionService


def build_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument("--run-config", required=True, help="run_config path (e.g. baseline/rps3_graph)")
    p.add_argument("--data-dir", default="data", help="input data root")
    return p


def load_inputs(run_config: str, data_dir: str | Path = "data", *, require_graph: bool) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any], Path]:
    data_root = Path(data_dir)
    cfg = _load_run_config(run_config, data_root)
    refs = _resolve_run_config_refs(cfg, require_graph=require_graph)

    exp_loader = ExperimentDataLoader(base_dir=data_root)
    matrix_data = exp_loader.load("matrix", refs.matrix)
    graph_data = exp_loader.load("graph", refs.graph) if refs.graph is not None else None
    setting = SettingDataLoader(base_dir=data_root / "setting").load(refs.setting)
    validate_matrix_input(matrix_data)
    validate_graph_input(graph_data)
    validate_setting_input(setting)
    return matrix_data, graph_data, setting, _run_config_path(run_config, data_root)


class _RunConfigRefs:
    def __init__(self, *, matrix: str, setting: str, graph: str | None):
        self.matrix = matrix
        self.setting = setting
        self.graph = graph


def _resolve_run_config_refs(cfg: dict[str, Any], *, require_graph: bool) -> _RunConfigRefs:
    matrix_name = _require_run_config_str(cfg, key="matrix")
    setting_name = _require_run_config_str(cfg, key="setting")
    graph_name = _optional_run_config_str(cfg, key="graph")

    if require_graph and graph_name is None:
        raise ValueError("このエントリーポイントでは run_config.graph が必須です")

    return _RunConfigRefs(matrix=matrix_name, setting=setting_name, graph=graph_name)


def _require_run_config_str(cfg: dict[str, Any], *, key: str) -> str:
    value = _optional_run_config_str(cfg, key=key)
    if value is None:
        raise ValueError(f"run_config.{key} は必須の文字列です")
    return value


def _optional_run_config_str(cfg: dict[str, Any], *, key: str) -> str | None:
    value = cfg.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"run_config.{key} は空でない文字列で指定してください")
    return value


def build_matrix(matrix_data: dict[str, Any]) -> PayoffMatrix:
    validate_matrix_input(matrix_data)

    labels = matrix_data.get("labels")
    if has_matrix_input(matrix_data):
        matrix = np.asarray(matrix_data["matrix"], dtype=float)
        return PayoffMatrixBuilder.from_general_matrix(matrix, labels=labels)

    return PayoffMatrixBuilder.from_characters(build_characters(matrix_data), labels=labels)


def has_matrix_input(matrix_data: dict[str, Any]) -> bool:
    return "matrix" in matrix_data


def build_characters(matrix_data: dict[str, Any]) -> list[Character]:
    validate_matrix_input(matrix_data)
    if "characters" not in matrix_data:
        raise ValueError("characters 入力が必要です")

    chars_raw = matrix_data["characters"]
    characters: list[Character] = []
    for item in chars_raw:
        characters.append(
            Character(
                float(item["p"]),
                MatchupVector(float(item["v"][0]), float(item["v"][1])),
                label=item["label"],
            )
        )
    return characters


def validate_matrix_input(matrix_data: dict[str, Any]) -> None:
    has_matrix = "matrix" in matrix_data
    has_characters = "characters" in matrix_data
    if has_matrix == has_characters:
        raise ValueError("matrix または characters のどちらか片方のみ指定してください")

    labels = matrix_data.get("labels")
    if labels is not None and (not isinstance(labels, list) or not all(isinstance(x, str) for x in labels)):
        raise ValueError("labels は文字列配列で指定してください")

    if has_matrix:
        matrix = np.asarray(matrix_data["matrix"], dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix は正方2次元配列である必要があります")
        if labels is not None and len(labels) != matrix.shape[0]:
            raise ValueError("labels 数と matrix サイズが一致しません")
        return

    chars_raw = matrix_data["characters"]
    if not isinstance(chars_raw, list) or not chars_raw:
        raise ValueError("characters は1件以上の配列が必要です")

    characters: list[Character] = []
    seen_labels: set[str] = set()
    for idx, item in enumerate(chars_raw):
        if not isinstance(item, dict):
            raise ValueError(f"characters[{idx}] はテーブルで指定してください")
        label = item.get("label")
        power = item.get("p")
        vector = item.get("v")
        if not isinstance(label, str) or not label:
            raise ValueError(f"characters[{idx}].label は必須文字列です")
        if label in seen_labels:
            raise ValueError(f"characters.label が重複しています: {label}")
        if not isinstance(power, (int, float)):
            raise ValueError(f"characters[{idx}].p は数値で指定してください")
        if not isinstance(vector, list) or len(vector) != 2:
            raise ValueError(f"characters[{idx}].v は長さ2の配列で指定してください")
        if any(not isinstance(value, (int, float)) for value in vector):
            raise ValueError(f"characters[{idx}].v は数値配列で指定してください")
        characters.append(Character(float(power), MatchupVector(float(vector[0]), float(vector[1])), label=label))
        seen_labels.add(label)


def validate_graph_input(graph_data: dict[str, Any] | None) -> None:
    if graph_data is None:
        return
    _optional_number(graph_data, key="threshold")
    _optional_int(graph_data, key="canvas_size")
    _optional_int(graph_data, key="margin")


def validate_setting_input(setting_data: dict[str, Any]) -> None:
    runmeta = setting_data.get("runmeta")
    output = setting_data.get("output")
    analysis_project = setting_data.get("analysis_project")

    if runmeta is not None and not isinstance(runmeta, dict):
        raise ValueError("setting.runmeta はテーブルで指定してください")
    if output is not None and not isinstance(output, dict):
        raise ValueError("setting.output はテーブルで指定してください")
    if analysis_project is not None and not isinstance(analysis_project, dict):
        raise ValueError("setting.analysis_project はテーブルで指定してください")

    if isinstance(runmeta, dict):
        _optional_non_empty_string(runmeta, key="sqlite_path", name="setting.runmeta.sqlite_path")
    if isinstance(output, dict):
        _optional_non_empty_string(output, key="base_dir", name="setting.output.base_dir")
    if isinstance(analysis_project, dict):
        _optional_non_empty_string(analysis_project, key="project_id", name="setting.analysis_project.project_id")
        _optional_non_empty_string(analysis_project, key="project_path", name="setting.analysis_project.project_path")


def _optional_non_empty_string(container: dict[str, Any], *, key: str, name: str) -> str | None:
    value = container.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} は空でない文字列で指定してください")
    return value


def _optional_number(container: dict[str, Any], *, key: str) -> float | None:
    value = container.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"graph.{key} は数値で指定してください")
    return float(value)


def _optional_int(container: dict[str, Any], *, key: str) -> int | None:
    value = container.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"graph.{key} は整数で指定してください")
    return value


def prepare_run_session(setting: dict[str, Any], command: str) -> tuple[RunSessionService, Any, sqlite3.Connection]:
    runmeta = setting.get("runmeta", {}) if isinstance(setting.get("runmeta"), dict) else {}
    output = setting.get("output", {}) if isinstance(setting.get("output"), dict) else {}
    project = setting.get("analysis_project", {}) if isinstance(setting.get("analysis_project"), dict) else {}

    db_path = str(runmeta.get("sqlite_path", ".runmeta/run_history.db"))
    output_root = Path(str(output.get("base_dir", "results")))

    conn = SQLiteConnectionFactory(db_path).connect()
    migrate(conn)

    projects = ProjectsRepository(conn)
    runs = RunsRepository(conn)
    service = RunSessionService(runs_repository=runs, artifact_store=RunArtifactStore(output_root))

    project_id = project.get("project_id") if isinstance(project.get("project_id"), str) and project.get("project_id") else None
    project_path = project.get("project_path") if isinstance(project.get("project_path"), str) and project.get("project_path") else None
    effective_project_path = _ensure_project_linkage(projects, project_id=project_id, project_path=project_path)

    ctx = service.start(command=command, project_id=project_id)
    # エントリーポイント実装の責務として、
    # output.base_dir/<run_id>/ の実行ディレクトリを明示的に確保する。
    result_dir = service.artifact_store.create_run_dir(ctx.run_id)
    _create_analysis_project_reference(run_id=ctx.run_id, result_dir=result_dir, project_path=effective_project_path)
    return service, ctx, conn


def _ensure_project_linkage(
    projects: ProjectsRepository,
    *,
    project_id: str | None,
    project_path: str | None,
) -> str | None:
    if project_id is None:
        return None

    project_row = projects.find(project_id)
    if project_row is None:
        stored_path = project_path or ""
        projects.add(project_id=project_id, project_path=stored_path, created_at=now_jst_iso())
        return stored_path

    if project_path is not None and project_path != project_row.project_path:
        projects.update(project_id, project_path=project_path)
        return project_path

    return project_row.project_path


def _create_analysis_project_reference(*, run_id: int, result_dir: Path, project_path: str | None) -> None:
    if not project_path:
        return

    refs_dir = Path(project_path) / "experiment_refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = refs_dir / str(run_id)
    _remove_existing_reference(ref_dir)

    target_dir = result_dir.resolve()
    try:
        ref_dir.symlink_to(target_dir, target_is_directory=True)
        _remove_existing_reference(refs_dir / f"{run_id}.txt")
        return
    except OSError:
        pass

    if _try_create_windows_junction(link_dir=ref_dir, target_dir=target_dir):
        _remove_existing_reference(refs_dir / f"{run_id}.txt")
        return

    (refs_dir / f"{run_id}.txt").write_text(
        "\n".join(
            [
                f"result_path={target_dir}",
                f"created_at={now_jst_iso()}",
                "status=running",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _remove_existing_reference(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _try_create_windows_junction(*, link_dir: Path, target_dir: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_dir), str(target_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def write_input_snapshots(
    service: RunSessionService,
    run_id: int,
    *,
    matrix_data: dict[str, Any],
    graph_data: dict[str, Any] | None,
    setting_data: dict[str, Any],
) -> None:
    input_dir = service.artifact_store.run_dir(run_id) / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "matrix.toml").write_text(_to_toml(matrix_data), encoding="utf-8")
    if graph_data is not None:
        (input_dir / "graph.toml").write_text(_to_toml(graph_data), encoding="utf-8")
    (input_dir / "setting.toml").write_text(_to_toml(setting_data), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_run_config(run_config: str, data_root: Path) -> dict[str, Any]:
    run_config_file = _run_config_path(run_config, data_root)
    return TomlTreeLoader().load(run_config_file)


def _run_config_path(run_config: str, data_root: Path) -> Path:
    path = Path(run_config)
    if not path.suffix:
        path = path.with_suffix(".toml")
    if not path.is_absolute():
        path = data_root / "run_config" / path
    return path


def _to_toml(value: Any) -> str:
    lines = _dump_toml(value)
    return "\n".join(lines).rstrip() + "\n"


def _dump_toml(value: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, dict):
        raise ValueError("TOML 出力対象は dict のみ対応")

    scalar_lines: list[str] = []
    table_lines: list[str] = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("TOML キーは文字列のみ対応")
        if isinstance(item, dict):
            table_name = f"{prefix}.{key}" if prefix else key
            nested = _dump_toml(item, table_name)
            if nested:
                if table_lines:
                    table_lines.append("")
                table_lines.append(f"[{table_name}]")
                table_lines.extend(nested)
        else:
            scalar_lines.append(f"{key} = {_toml_literal(item)}")

    return scalar_lines + ([""] + table_lines if scalar_lines and table_lines else table_lines)


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        if value and all(isinstance(x, dict) for x in value):
            chunks = []
            for item in value:
                body = ", ".join(f"{k} = {_toml_literal(v)}" for k, v in item.items())
                chunks.append("{" + body + "}")
            return "[" + ", ".join(chunks) + "]"
        return "[" + ", ".join(_toml_literal(x) for x in value) + "]"
    raise ValueError(f"未対応のTOML値型です: {type(value)}")
