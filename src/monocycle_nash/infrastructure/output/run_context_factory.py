"""ファイルシステムベースの結果出力 OutputPort 実装。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from monocycle_nash.application.ports import OutputPort, RunContext


class FileOutputAdapter(OutputPort):
    """ファイルシステムベースの結果出力。

    result/<run_id>/input/    : 入力ファイルのスナップショット
    result/<run_id>/artifact/ : 計算結果の出力
    result/<run_id>/metadata.json : メタデータ
    """

    def __init__(self, result_base_dir: Path | str = "result") -> None:
        self._base_dir = Path(result_base_dir)

    def create_run_context(self, use_case_name: str) -> RunContext:
        """タイムスタンプベースの run_id でディレクトリを作成。"""
        run_id = time.strftime("%Y%m%d_%H%M%S")
        ctx = RunContext(run_id=run_id, base_dir=self._base_dir)
        ctx.input_dir.mkdir(parents=True, exist_ok=True)
        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        return ctx

    def write_json(self, ctx: RunContext, filename: str, data: dict[str, Any]) -> None:
        """artifact_dir にJSONファイルを書き出す。"""
        path = ctx.artifact_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_svg(self, ctx: RunContext, filename: str, content: str) -> None:
        """artifact_dir にSVGファイルを書き出す。"""
        path = ctx.artifact_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def save_input_snapshot(self, ctx: RunContext, filename: str, data: dict[str, Any]) -> None:
        """input_dir に入力データのスナップショットを保存。"""
        path = ctx.input_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_metadata(self, ctx: RunContext, metadata: dict[str, Any]) -> None:
        """metadata.json に実行メタデータを書き出す。"""
        ctx.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
