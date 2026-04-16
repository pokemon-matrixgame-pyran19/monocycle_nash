"""SVG可視化アダプター - VisualizationPort 実装。"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from monocycle_nash.application.ports import VisualizationPort
from monocycle_nash.domain.character import Character
from monocycle_nash.infrastructure.visualization.character_vector_graph import (
    CharacterVectorGraphPlotter,
)
from monocycle_nash.infrastructure.visualization.payoff_graph import (
    PayoffDirectedGraphPlotter,
)


class SvgVisualizationAdapter(VisualizationPort):
    """プロッターを使ってSVG文字列を生成するアダプター。"""

    def draw_payoff_graph(
        self,
        matrix: np.ndarray,
        labels: list[str],
        threshold: float = 0.0,
        canvas_size: int = 840,
    ) -> str:
        """利得行列のグラフをSVG文字列で返す。"""
        plotter = PayoffDirectedGraphPlotter(matrix, labels, threshold=threshold)
        return self._draw_to_string(
            lambda path: plotter.draw(path, canvas_size=canvas_size)
        )

    def draw_character_plot(
        self,
        characters: list[Character],
        canvas_size: int = 840,
        margin: int = 90,
    ) -> str:
        """キャラクターベクトルプロットをSVG文字列で返す。"""
        plotter = CharacterVectorGraphPlotter(characters)
        return self._draw_to_string(
            lambda path: plotter.draw(path, canvas_size=canvas_size, margin=margin)
        )

    @staticmethod
    def _draw_to_string(draw_fn: Callable[[Path], Path]) -> str:
        """プロッターの draw メソッドを一時ファイル経由で呼び出し、SVG文字列を返す。"""
        # Use a workspace-local scratch file to avoid /tmp
        scratch_dir = Path(".scratch_svg")
        scratch_dir.mkdir(exist_ok=True)
        scratch_path = scratch_dir / f"_viz_{os.getpid()}.svg"
        try:
            draw_fn(scratch_path)
            return scratch_path.read_text(encoding="utf-8")
        finally:
            scratch_path.unlink(missing_ok=True)
            # Clean up scratch dir if empty
            try:
                scratch_dir.rmdir()
            except OSError:
                pass
