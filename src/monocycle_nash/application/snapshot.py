"""設定ツリースナップショット DTO。"""

from __future__ import annotations

from dataclasses import dataclass

from monocycle_nash.application.node_spec import NodeSpec


@dataclass(frozen=True)
class ConfigTreeSnapshot:
    """実行時に読み込んだ設定ツリーのスナップショット。

    ルート NodeSpec のみを保持する。メタ情報は run 側で管理されるため、
    このオブジェクトは設定ツリーの内容だけを保持することで関心を分離する。
    """

    root: NodeSpec
