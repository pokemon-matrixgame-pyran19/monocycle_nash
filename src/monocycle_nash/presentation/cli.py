"""CLI 実行フロー。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from monocycle_nash.application.matrix_config_tree import MatrixConfigTree, MatrixConfigTreeResolver
from monocycle_nash.application.matrix_node_factory import MatrixNodeFactory
from monocycle_nash.application.snapshot import ConfigTreeSnapshot
from monocycle_nash.infrastructure.input.toml_matrix_config_port import TomlMatrixConfigPort
from monocycle_nash.infrastructure.output import (
    FileSystemOutputPathPort,
    TomlConfigTreeSnapshotStore,
)


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
    parser.add_argument(
        "--result-dir",
        default="result",
        help="出力先ディレクトリ",
    )
    return parser


def run(
    *,
    config_id: str,
    data_dir: Path | str,
    result_dir: Path | str,
) -> int:
    config_port = TomlMatrixConfigPort(data_dir=data_dir)
    spec = config_port.load_node_spec(config_id)

    root = MatrixNodeFactory().build(spec)
    resolver = MatrixConfigTreeResolver(
        output_path_port=FileSystemOutputPathPort(result_base_dir=result_dir),
    )
    result = resolver.resolve(MatrixConfigTree(root=root))

    snapshot_store = TomlConfigTreeSnapshotStore(result_base_dir=result_dir)
    snapshot_path = snapshot_store.store(
        run_id=str(result.run_id),
        snapshot=ConfigTreeSnapshot(root=spec),
    )

    print(f"run_id: {result.run_id}")
    print(f"matrix shape: {result.root.matrix.shape}")
    print(f"snapshot: {snapshot_path}")
    if result.outputs:
        print("outputs:")
        for resolved in result.outputs:
            print(f"- {resolved.path}")
    else:
        print("outputs: []")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return run(
            config_id=args.config_id,
            data_dir=args.data_dir,
            result_dir=args.result_dir,
        )
    except Exception as exc:
        print(f"実行に失敗しました ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
