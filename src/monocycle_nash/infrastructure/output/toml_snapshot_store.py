"""設定ツリースナップショットを TOML ファイルに保存するストア実装。

保存先: result/<run_id>/input/config_tree.toml

NodeSpec の構造を再帰的に TOML 互換の辞書へ変換し、
tomli_w で書き出す。空のセクション（params / refs / children / outputs）は
ファイルをシンプルに保つため省略する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli_w

from monocycle_nash.application.node_spec import NodeSpec, OutputSpec
from monocycle_nash.application.ports import ConfigTreeSnapshotStorePort
from monocycle_nash.application.snapshot import ConfigTreeSnapshot


class TomlConfigTreeSnapshotStore(ConfigTreeSnapshotStorePort):
    """TOML 形式で設定ツリースナップショットを保存するストア。

    `result/<run_id>/input/config_tree.toml` にルート NodeSpec を書き出す。
    """

    FILENAME = "config_tree.toml"

    def __init__(self, result_base_dir: Path | str = "result") -> None:
        self._result_base_dir = Path(result_base_dir)

    def store(self, run_id: str, snapshot: ConfigTreeSnapshot) -> Path:
        """スナップショットを TOML ファイルに保存してパスを返す。"""
        path = self._result_base_dir / run_id / "input" / self.FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._node_spec_to_dict(snapshot.root)
        with path.open("wb") as f:
            tomli_w.dump(data, f)
        return path

    # ------------------------------------------------------------------
    # 内部実装
    # ------------------------------------------------------------------

    def _node_spec_to_dict(self, spec: NodeSpec) -> dict[str, Any]:
        d: dict[str, Any] = {"method": spec.method, "name": spec.name}
        if spec.params:
            d["params"] = spec.params
        if spec.refs:
            d["refs"] = dict(spec.refs)
        if spec.children:
            d["children"] = {
                k: self._node_spec_to_dict(v) for k, v in spec.children.items()
            }
        if spec.outputs:
            d["outputs"] = [self._output_spec_to_dict(o) for o in spec.outputs]
        return d

    def _output_spec_to_dict(self, spec: OutputSpec) -> dict[str, Any]:
        d: dict[str, Any] = {"method": spec.method}
        if spec.params:
            d["params"] = spec.params
        return d
