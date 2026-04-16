"""Mini UC: 利得行列のグラフを SVG で生成する。"""

from __future__ import annotations

from monocycle_nash.domain.matrix.base import PayoffMatrix

from .ports import GraphConfigPort, VisualizationPort


class DrawPayoffGraphUseCase:
    """ユースケース: グラフ設定ファイルを読み込み、利得行列をグラフ表現した SVG を生成する。"""

    def __init__(
        self,
        visualization: VisualizationPort,
        config_port: GraphConfigPort,
    ) -> None:
        self._visualization = visualization
        self._config_port = config_port

    def execute(self, matrix: PayoffMatrix, graph_id: str) -> str:
        """
        利得行列のグラフを生成する。

        Args:
            matrix: 対象の利得行列
            graph_id: グラフ設定ファイルの識別子

        Returns:
            SVG 文字列
        """
        config = self._config_port.load_graph_config(graph_id)
        payoff_cfg = config.get("payoff", {})
        return self._visualization.draw_payoff_graph(
            matrix=matrix.matrix,
            labels=matrix.labels,
            threshold=float(payoff_cfg.get("threshold", 0.0)),
            canvas_size=int(payoff_cfg.get("canvas_size", 840)),
        )
