"""構築特徴ベクトル群から有向グラフ画像(SVG)を生成する。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from monocycle_nash.domain.team import TeamFeatureVector


@dataclass(frozen=True)
class FeatureVectorGraphEdge:
    source: int
    target: int
    value: float


class TeamFeatureVectorDirectedGraphPlotter:
    """特徴ベクトル間の外積に基づく有向グラフを描画する。"""

    def __init__(
        self,
        labels: list[str],
        vectors: list[TeamFeatureVector],
        threshold: float = 0.0,
    ):
        if len(labels) != len(vectors):
            raise ValueError("labels and vectors must have the same length")
        if not labels:
            raise ValueError("labels must contain at least one element")
        self._labels = labels
        self._vectors = vectors
        self._threshold = float(threshold)

    def extract_edges(self) -> list[FeatureVectorGraphEdge]:
        edges: list[FeatureVectorGraphEdge] = []
        for i, vi in enumerate(self._vectors):
            for j, vj in enumerate(self._vectors):
                if i == j:
                    continue
                cross = vi.x * vj.y - vi.y * vj.x
                if cross > self._threshold:
                    edges.append(FeatureVectorGraphEdge(source=i, target=j, value=float(cross)))
        return edges

    def draw(self, output_path: str | Path, canvas_size: int = 840) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        center = canvas_size / 2
        radius = canvas_size * 0.34
        node_r = 42
        positions = self._circle_layout(len(self._labels), center, radius)
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

        svg_parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}">',
            "<defs>",
            *defs,
            "</defs>",
            '<rect width="100%" height="100%" fill="white" />',
            *edge_draw_parts,
        ]

        for i, (x, y) in enumerate(positions):
            label = self._escape(self._labels[i])
            vector = self._vectors[i]
            svg_parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{node_r}" fill="#e5e7eb" stroke="#111827" stroke-width="2" />'
            )
            svg_parts.append(
                f'<text x="{x:.2f}" y="{y - 8:.2f}" text-anchor="middle" dominant-baseline="middle" '
                f'font-size="13" fill="#111827">{label}</text>'
            )
            svg_parts.append(
                f'<text x="{x:.2f}" y="{y + 12:.2f}" text-anchor="middle" dominant-baseline="middle" '
                f'font-size="11" fill="#1f2937">({vector.x:.2f}, {vector.y:.2f})</text>'
            )

        svg_parts.append("</svg>")
        output.write_text("\n".join(svg_parts), encoding="utf-8")
        return output

    @staticmethod
    def _interpolate_color(norm: float) -> str:
        clamped = max(0.0, min(1.0, norm))
        low = (29, 78, 216)
        high = (220, 38, 38)
        rgb = tuple(int(low[i] + (high[i] - low[i]) * clamped) for i in range(3))
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    @staticmethod
    def _circle_layout(node_count: int, center: float, radius: float) -> list[tuple[float, float]]:
        angles = [2 * math.pi * i / node_count for i in range(node_count)]
        return [
            (float(center + radius * math.cos(theta)), float(center + radius * math.sin(theta)))
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
