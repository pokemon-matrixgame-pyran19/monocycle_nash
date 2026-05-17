"""CLI 実行フロー。"""

from __future__ import annotations

import argparse
import shutil
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from monocycle_nash.application.matrix_config_tree import MatrixConfigTree, MatrixConfigTreeResolver
from monocycle_nash.application.matrix_node_factory import MatrixNodeFactory
from monocycle_nash.application.snapshot import ConfigTreeSnapshot
from monocycle_nash.infrastructure.input import (
    TomlCharacterListFilePort,
    TomlMatrixConfigPort,
    TomlTeamListFilePort,
)
from monocycle_nash.infrastructure.output import (
    FileSystemOutputPathPort,
    TomlConfigTreeSnapshotStore,
)

_RESULT_DIR = "result"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="monocycle-nash",
        description="TOML 設定から利得行列を解決し、必要な出力を生成します。",
    )
    parser.add_argument(
        "config_id",
        help="設定IDまたは設定ファイルパス（拡張子省略可）",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="相対 config_id の解決基準となるデータディレクトリ",
    )
    return parser


def run(
    *,
    config_id: str,
    data_dir: Path | str,
) -> int:
    config_port = TomlMatrixConfigPort(data_dir=data_dir)
    spec = config_port.load_node_spec(config_id)
    config_path = config_port.resolve_config_path(config_id)
    run_id = _resolve_run_id(result_base_dir=Path(_RESULT_DIR), config_path=config_path)

    root = MatrixNodeFactory().build(spec)
    resolver = MatrixConfigTreeResolver(
        output_path_port=FileSystemOutputPathPort(
            result_base_dir=_RESULT_DIR,
            run_id_override=run_id,
        ),
        character_list_file_port=TomlCharacterListFilePort(data_dir=data_dir),
        team_list_file_port=TomlTeamListFilePort(data_dir=data_dir),
    )
    result = resolver.resolve(MatrixConfigTree(root=root))

    snapshot_store = TomlConfigTreeSnapshotStore(result_base_dir=_RESULT_DIR)
    snapshot_path = snapshot_store.store(
        run_id=run_id,
        snapshot=ConfigTreeSnapshot(root=spec),
    )

    print(f"run_id: {run_id}")
    print(f"matrix shape: {result.root.value.matrix.shape}")
    print(f"snapshot: {snapshot_path}")
    if result.outputs:
        print("outputs:")
        for resolved in result.outputs:
            print(f"- {resolved.path}")
    else:
        print("outputs: []")
    return 0


def _resolve_run_id(*, result_base_dir: Path, config_path: Path) -> str:
    if _is_temp_run_enabled(config_path):
        temp_dir = result_base_dir / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return "temp"
    return _next_serial_run_id(result_base_dir)


def _is_temp_run_enabled(config_path: Path) -> bool:
    """`[run].temp` を読む。読み込み/構文エラーは呼び出し側へ伝播させる。"""
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    run_section = data.get("run")
    if not isinstance(run_section, dict):
        return False
    return bool(run_section.get("temp", False))


def _next_serial_run_id(result_base_dir: Path) -> str:
    if not result_base_dir.exists():
        return "1"
    serial_ids: list[int] = []
    for child in result_base_dir.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        serial_id = int(child.name)
        # run_id は 1 始まりなので 0 ディレクトリは採番対象から除外する。
        if serial_id >= 1:
            serial_ids.append(serial_id)
    if not serial_ids:
        return "1"
    return str(max(serial_ids) + 1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return run(
            config_id=args.config_id,
            data_dir=args.data_dir,
        )
    except Exception as exc:
        print(f"実行に失敗しました ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
