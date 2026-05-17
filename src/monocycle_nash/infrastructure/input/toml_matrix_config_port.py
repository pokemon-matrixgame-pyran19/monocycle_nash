"""TOML ファイルを読み込んで NodeSpec を返す MatrixTreeConfigPort 実装。

TOML ファイルのスキーマ（トップレベルキー）:
  method   : str  — ノード種別（必須）
  name     : str  — ノード名（省略時 "root"）
  [params] : テーブル — method 固有のパラメータ
  [refs]   : テーブル — 補助データへのファイルパス参照（値は文字列）
  [children.<key>] : ネストしたノード仕様（再帰的に同スキーマを適用）
 [[outputs]] : 出力ノード仕様の配列
    method   : str
    runner   : str | null  — 同一 runner 名で最終集約実行（省略可）
    [params] : テーブル — 出力固有のパラメータ

使用例（data/rps.toml）::

    method = "monocycle_from_characters"
    name = "rps"

    [[params.characters]]
    power = 0.5
    vector = [1.0, 0.0]
    label = "Rock"

使用例（refs でファイル参照）::

    method = "monocycle_from_characters"
    name = "janken"

    [refs]
    characters = "characters/janken"

    [[outputs]]
    method = "payoff_directed_graph"
    [outputs.params]
    filename = "janken.svg"
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from monocycle_nash.application.node_spec import NodeSpec, OutputSpec
from monocycle_nash.application.ports import MatrixTreeConfigPort


class TomlMatrixConfigPort(MatrixTreeConfigPort):
    """TOML ファイルから NodeSpec を読み込むポート実装。

    config_id には以下のいずれかを指定できる:
    - 絶対パス（拡張子あり・なし）
    - data_dir からの相対パス（拡張子省略時 .toml を補完）
    """

    def __init__(self, data_dir: Path | str = "data") -> None:
        self._data_dir = Path(data_dir)

    def load_node_spec(self, config_id: str) -> NodeSpec:
        """設定 ID から TOML を読み込み NodeSpec を返す。"""
        path = self._resolve_path(config_id)
        data = self._load_toml(path)
        return self._parse_node_spec(data)

    def resolve_config_path(self, config_id: str) -> Path:
        """設定 ID から実ファイルパスを解決して返す。"""
        return self._resolve_path(config_id)

    # ------------------------------------------------------------------
    # 内部実装
    # ------------------------------------------------------------------

    def _resolve_path(self, config_id: str) -> Path:
        candidate = Path(config_id)
        if candidate.is_absolute():
            return candidate if candidate.suffix else candidate.with_suffix(".toml")
        candidate = self._data_dir / config_id
        if not candidate.suffix:
            candidate = candidate.with_suffix(".toml")
        return candidate

    def _load_toml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
        with path.open("rb") as f:
            return tomllib.load(f)

    def _parse_node_spec(self, data: dict[str, Any]) -> NodeSpec:
        if "method" not in data:
            raise KeyError("設定ファイルに 'method' キーが必要です")
        method: str = data["method"]
        name: str = data.get("name", "root")
        params: dict[str, Any] = dict(data.get("params", {}))
        refs: dict[str, str] = {k: str(v) for k, v in data.get("refs", {}).items()}
        children: dict[str, NodeSpec] = {
            k: self._parse_node_spec(v)
            for k, v in data.get("children", {}).items()
        }
        outputs: tuple[OutputSpec, ...] = tuple(
            self._parse_output_spec(o) for o in data.get("outputs", [])
        )
        return NodeSpec(
            method=method,
            name=name,
            params=params,
            refs=refs,
            children=children,
            outputs=outputs,
        )

    def _parse_output_spec(self, data: dict[str, Any]) -> OutputSpec:
        if "method" not in data:
            raise KeyError("outputs エントリに 'method' キーが必要です")
        return OutputSpec(
            method=data["method"],
            runner=data.get("runner"),
            params=dict(data.get("params", {})),
        )
