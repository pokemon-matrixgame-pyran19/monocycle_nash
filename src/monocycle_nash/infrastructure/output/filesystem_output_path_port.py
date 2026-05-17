"""ファイルシステム上で出力先パスを決定する OutputPathPort 実装。"""

from __future__ import annotations

import re
from pathlib import Path

from monocycle_nash.application.ports import OutputPathPort


class FileSystemOutputPathPort(OutputPathPort):
    """`result/<実行単位ID>/<ノード階層>/<output_method>/` を構築する。"""

    _SAFE_CHARS = re.compile(r"[^0-9A-Za-z_-]+")

    def __init__(
        self,
        result_base_dir: Path | str = "results",
        run_id_override: str | None = None,
    ) -> None:
        self._result_base_dir = Path(result_base_dir)
        self._run_id_override = run_id_override

    def resolve_output_path(
        self,
        *,
        run_id: str,
        node_path: tuple[str, ...],
        output_method: str,
        filename: str,
    ) -> Path:
        effective_run_id = self._run_id_override if self._run_id_override else run_id
        safe_run_id = self._sanitize_component(effective_run_id)
        safe_output_method = self._sanitize_component(output_method)
        if len(node_path) == 0:
            # 防御的に、空階層入力時は root フォルダへフォールバックする。
            safe_node_path = ("root",)
        else:
            safe_node_path = tuple(self._sanitize_component(name) for name in node_path)
        safe_filename = self._sanitize_filename(filename)

        path = (
            self._result_base_dir
            / safe_run_id
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
        name_path = Path(filename).name
        basename = Path(name_path)
        stem = self._sanitize_component(basename.stem)
        suffixes = [
            self._sanitize_component(suffix[1:])
            for suffix in basename.suffixes
            if len(suffix) > 1
        ]
        if suffixes:
            return f"{stem}.{'.'.join(suffixes)}"
        return stem
