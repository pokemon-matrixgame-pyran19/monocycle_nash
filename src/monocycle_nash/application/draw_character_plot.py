"""Mini UC: キャラクターパラメータのプロットを SVG で生成する。"""

from __future__ import annotations

from monocycle_nash.domain.matrix.monocycle import MonocyclePayoffMatrix

from .ports import GraphConfigPort, VisualizationPort


class DrawCharacterPlotUseCase:
    """ユースケース: グラフ設定ファイルを読み込み、キャラクターパラメータをプロットした SVG を生成する。"""

    def __init__(
        self,
        visualization: VisualizationPort,
        config_port: GraphConfigPort,
    ) -> None:
        self._visualization = visualization
        self._config_port = config_port

    def execute(self, matrix: MonocyclePayoffMatrix, graph_id: str) -> str:
        """
        キャラクターパラメータのプロットを生成する。

        Args:
            matrix: 単相性モデル利得行列（キャラクター情報を持つ）
            graph_id: グラフ設定ファイルの識別子

        Returns:
            SVG 文字列
        """
        config = self._config_port.load_graph_config(graph_id)
        char_cfg = config.get("character", {})
        return self._visualization.draw_character_plot(
            characters=matrix.characters,
            canvas_size=int(char_cfg.get("canvas_size", 840)),
            margin=int(char_cfg.get("margin", 90)),
        )
