from __future__ import annotations

import argparse
import json
import sqlite3
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
        return PayoffMatrixBuilder.from_general_matrix(matrix, labels=labels)

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
        characters.append(Character(float(power), MatchupVector(float(vector[0]), float(vector[1])), label=label))
        seen_labels.add(label)

    return PayoffMatrixBuilder.from_characters(characters, labels=labels)


def prepare_run_session(setting: dict[str, Any], command: str) -> tuple[RunSessionService, Any, sqlite3.Connection]:
    runmeta = setting.get("runmeta", {}) if isinstance(setting.get("runmeta"), dict) else {}
    output = setting.get("output", {}) if isinstance(setting.get("output"), dict) else {}
    project = setting.get("analysis_project", {}) if isinstance(setting.get("analysis_project"), dict) else {}

    db_path = str(runmeta.get("sqlite_path", ".runmeta/run_history.db"))
    output_root = Path(str(output.get("base_dir", "result")))

    conn = SQLiteConnectionFactory(db_path).connect()
    migrate(conn)

    projects = ProjectsRepository(conn)
    runs = RunsRepository(conn)
    service = RunSessionService(runs_repository=runs, artifact_store=RunArtifactStore(output_root))

    project_id = project.get("project_id") if isinstance(project.get("project_id"), str) and project.get("project_id") else None
    project_path = project.get("project_path") if isinstance(project.get("project_path"), str) else None
    if project_id and projects.find(project_id) is None:
        projects.add(project_id=project_id, project_path=project_path or "", created_at=now_jst_iso())

    ctx = service.start(command=command, project_id=project_id)
    return service, ctx, conn


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
