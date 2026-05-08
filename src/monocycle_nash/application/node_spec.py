"""ノード設定 DTO — インフラ層がアプリ層へノード設定を渡すための中間表現。

設定ファイルから読み込んだノード定義を typed ApplicationNode ツリー内の
行列構築ノードへ変換する前のアンマーシャリング済みデータ構造として使用する。

スキーマは全ノード共通で method + name + params + children + refs + outputs の6要素。
- method: ノード種別を識別する文字列
- name: ノード名（省略時は "root"）
- params: method 固有のスカラー・リスト等パラメータ
- children: 子ノード仕様（役割名 → NodeSpec）
- refs: 補助データへのファイル参照（役割名 → パス文字列）
- outputs: 出力ノード仕様のタプル
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OutputSpec:
    """OutputNode の設定 DTO。"""

    method: str
    runner: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeSpec:
    """行列構築ノードの設定 DTO。

    インフラ層（設定ファイル読み込み）とアプリ層（ノード生成）の間を橋渡しする。
    すべての行列構築ノード種別をこの1構造で表現できるよう設計する。
    """

    method: str
    name: str = "root"
    params: dict[str, Any] = field(default_factory=dict)
    children: dict[str, NodeSpec] = field(default_factory=dict)
    refs: dict[str, str] = field(default_factory=dict)
    outputs: tuple[OutputSpec, ...] = field(default_factory=tuple)
