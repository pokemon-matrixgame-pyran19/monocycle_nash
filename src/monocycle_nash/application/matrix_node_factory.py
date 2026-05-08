"""NodeSpec から typed ApplicationNode ツリーの行列構築ノードを組み立てるファクトリ。

インフラ層がロードした NodeSpec を受け取り、
対応する具象 MatrixNode インスタンスに変換して返す。

各 MatrixNode サブクラスが `node_method="..."` 宣言と `_from_spec` classmethod を
自身に持つことで、ディスパッチと構築ロジックをノード側にカプセル化している。
新規ノード種別を追加する場合は MatrixNodeFactory を変更せず、
matrix_nodes.py に具象クラスを追加するだけでよい。
"""

from __future__ import annotations

from monocycle_nash.application.matrix_nodes import MatrixNode
from monocycle_nash.application.node_spec import NodeSpec


class MatrixNodeFactory:
    """NodeSpec から typed ApplicationNode ツリーの行列構築ノードを組み立てるファクトリ。

    ディスパッチと構築ロジックは各 MatrixNode サブクラスの _from_spec に委譲する。
    """

    def build(self, spec: NodeSpec) -> MatrixNode:
        """NodeSpec を対応する MatrixNode に変換して返す。"""
        return MatrixNode.create_from_spec(spec, self.build)
