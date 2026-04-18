"""ファイルシステム上で出力先パスを決定する OutputPathPort 実装。"""

from __future__ import annotations

import re
from pathlib import Path

from monocycle_nash.application.ports import OutputPathPort


class FileSystemOutputPathPort(OutputPathPort):
    """`result/<実行単位ID>/<ノード階層>/<output_method>/` を構築する。"""

    _SAFE_CHARS = re.compile(r"[^0-9A-Za-z_-]+")

    def __init__(self, result_base_dir: Path | str = "result") -> None:
        self._result_base_dir = Path(result_base_dir)

    def resolve_output_path(
        self,
        *,
        execution_unit_id: str,
        node_path: tuple[str, ...],
        output_method: str,
        filename: str,
    ) -> Path:
        safe_execution_unit_id = self._sanitize_component(execution_unit_id)
        safe_output_method = self._sanitize_component(output_method)
        safe_node_path = (
            tuple(self._sanitize_component(name) for name in node_path)
            if node_path
            # 空階層は root フォルダにフォールバックする。
            else ("root",)
        )
        safe_filename = self._sanitize_filename(filename)

        path = (
            self._result_base_dir
            / safe_execution_unit_id
            / Path(*safe_node_path)
            / safe_output_method
            / safe_filename
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _sanitize_component(self, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            return "_empty"
        sanitized = self._SAFE_CHARS.sub("_", stripped)
        normalized = sanitized.strip("_")
        if normalized:
            return normalized
        return "_dot"

    def _sanitize_filename(self, filename: str) -> str:
        name = Path(filename).name
        stem = self._sanitize_component(Path(name).stem)
        suffixes = [
            self._sanitize_component(suffix[1:])
            for suffix in Path(name).suffixes
            if len(suffix) > 1
        ]
        if suffixes:
            return f"{stem}." + ".".join(suffixes)
        return stem
