from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np


@dataclass(frozen=True)
class GraphEdge:
    """有向グラフの辺情報。"""

    source: int
    target: int
    value: float


class PayoffDirectedGraphPlotter:
    """利得行列から純粋戦略間の有向グラフ画像(SVG)を生成する。"""

    def __init__(self, payoff_matrix: np.ndarray, labels: list[str], threshold: float = 0.0):
        matrix = np.asarray(payoff_matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("payoff_matrix は2次元配列である必要があります")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("交代行列を想定するため、正方行列が必要です")
        if len(labels) != matrix.shape[0]:
            raise ValueError("labels の数は行列サイズと一致する必要があります")

        self._matrix = matrix
        self._labels = labels
        self._threshold = float(threshold)

    def extract_edges(self) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        size = self._matrix.shape[0]
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                value = float(self._matrix[i, j])
                if value > self._threshold:
                    edges.append(GraphEdge(i, j, value))
        return edges

    def draw(self, output_path: str | Path, canvas_size: int = 840) -> Path:
        """有向グラフをSVG画像として保存。"""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        center = canvas_size / 2
        radius = canvas_size * 0.34
        node_r = 42
        positions = self._circle_layout(self._matrix.shape[0], center, radius)
        edges = self.extract_edges()

        max_value = max((edge.value for edge in edges), default=1.0)

        defs: list[str] = []
        edge_draw_parts: list[str] = []

        for edge_index, edge in enumerate(edges):
            start = positions[edge.source]
            end = positions[edge.target]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dist = math.hypot(dx, dy)
            ux, uy = dx / dist, dy / dist
            sx, sy = start[0] + ux * node_r, start[1] + uy * node_r
            tx, ty = end[0] - ux * node_r, end[1] - uy * node_r

            norm = edge.value / max_value if max_value > 0 else 0.0
            stroke_width = 1.5 + 5.0 * norm
            opacity = 0.4 + 0.6 * norm
            edge_color = self._interpolate_color(norm)
            marker_id = f"arrow-{edge_index}"

            defs.append(
                f'<marker id="{marker_id}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">'
                f'<polygon points="0 0, 10 3.5, 0 7" fill="{edge_color}" /></marker>'
            )

            edge_draw_parts.append(
                f'<line x1="{sx:.2f}" y1="{sy:.2f}" x2="{tx:.2f}" y2="{ty:.2f}" '
                f'stroke="{edge_color}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}" marker-end="url(#{marker_id})" />'
            )

            mx, my = (sx + tx) / 2, (sy + ty) / 2
            edge_draw_parts.append(
                f'<rect x="{mx - 22:.2f}" y="{my - 10:.2f}" width="44" height="20" rx="4" '
                'fill="#ffffff" stroke="#94a3b8" stroke-width="1" opacity="0.95" />'
            )
            edge_draw_parts.append(
                f'<text x="{mx:.2f}" y="{my:.2f}" text-anchor="middle" dominant-baseline="middle" '
                f'font-size="14" fill="#1e40af">{edge.value:.2f}</text>'
            )

        svg_parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}">',
            '<defs>',
            *defs,
            '</defs>',
            '<rect width="100%" height="100%" fill="white" />',
            *edge_draw_parts,
        ]

        for i, (x, y) in enumerate(positions):
            label = self._escape(self._labels[i])
            svg_parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{node_r}" fill="#e5e7eb" stroke="#111827" stroke-width="2" />')
            svg_parts.append(
                f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="middle" dominant-baseline="middle" '
                f'font-size="15" fill="#111827">{label}</text>'
            )

        svg_parts.append('</svg>')
        output.write_text("\n".join(svg_parts), encoding="utf-8")
        return output

    @staticmethod
    def _interpolate_color(norm: float) -> str:
        """エッジ強度に応じて青→赤の色グラデーションを返す。"""
        clamped = max(0.0, min(1.0, norm))
        low = (29, 78, 216)   # #1d4ed8
        high = (220, 38, 38)  # #dc2626
        rgb = tuple(int(low[i] + (high[i] - low[i]) * clamped) for i in range(3))
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    @staticmethod
    def _circle_layout(size: int, center: float, radius: float) -> list[tuple[float, float]]:
        angles = np.linspace(0, 2 * np.pi, size, endpoint=False)
        return [
            (float(center + radius * np.cos(theta)), float(center + radius * np.sin(theta)))
            for theta in angles
        ]

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
