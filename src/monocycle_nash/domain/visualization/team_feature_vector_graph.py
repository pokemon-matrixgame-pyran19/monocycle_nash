"""構築特徴ベクトル群を2次元散布図(SVG)として生成する。"""

from __future__ import annotations

from pathlib import Path

from monocycle_nash.domain.team import TeamFeatureVector


class TeamFeatureVectorScatterPlotter:
    """構築特徴ベクトルを2次元平面に散布図として描画する。"""

    def __init__(
        self,
        labels: list[str],
        vectors: list[TeamFeatureVector],
    ):
        if len(labels) != len(vectors):
            raise ValueError("labels and vectors must have the same length")
        if not labels:
            raise ValueError("labels must contain at least one element")
        self._labels = labels
        self._vectors = vectors

    def draw(self, output_path: str | Path, canvas_size: int = 840, margin: int = 90) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        xs = [float(v.x) for v in self._vectors]
        ys = [float(v.y) for v in self._vectors]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        pad_x = span_x * 0.2
        pad_y = span_y * 0.2
        world_min_x, world_max_x = min_x - pad_x, max_x + pad_x
        world_min_y, world_max_y = min_y - pad_y, max_y + pad_y

        world_min_x = min(world_min_x, 0.0)
        world_max_x = max(world_max_x, 0.0)
        world_min_y = min(world_min_y, 0.0)
        world_max_y = max(world_max_y, 0.0)

        width = canvas_size
        height = canvas_size
        inner_w = width - margin * 2
        inner_h = height - margin * 2

        def sx(x: float) -> float:
            return margin + (x - world_min_x) / (world_max_x - world_min_x) * inner_w

        def sy(y: float) -> float:
            return height - margin - (y - world_min_y) / (world_max_y - world_min_y) * inner_h

        axis_x0 = sx(world_min_x)
        axis_x1 = sx(world_max_x)
        axis_y0 = sy(world_min_y)
        axis_y1 = sy(world_max_y)
        zero_x = sx(0.0)
        zero_y = sy(0.0)
        origin_point_radius = 5.5
        origin_label_x_offset = 10.0
        point_radius = 14.0
        label_y_offset = 16.0
        vector_text_y_offset = 18.0

        svg_parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white" />',
            f'<line x1="{axis_x0:.2f}" y1="{axis_y0:.2f}" x2="{axis_x1:.2f}" y2="{axis_y0:.2f}" stroke="#d1d5db" stroke-width="1" />',
            f'<line x1="{axis_x0:.2f}" y1="{axis_y1:.2f}" x2="{axis_x0:.2f}" y2="{axis_y0:.2f}" stroke="#d1d5db" stroke-width="1" />',
            f'<line x1="{axis_x0:.2f}" y1="{zero_y:.2f}" x2="{axis_x1:.2f}" y2="{zero_y:.2f}" stroke="#6b7280" stroke-width="1.5" />',
            f'<line x1="{zero_x:.2f}" y1="{axis_y1:.2f}" x2="{zero_x:.2f}" y2="{axis_y0:.2f}" stroke="#6b7280" stroke-width="1.5" />',
            f'<circle cx="{zero_x:.2f}" cy="{zero_y:.2f}" r="{origin_point_radius:.2f}" fill="#ef4444" stroke="white" stroke-width="1.5" />',
            f'<text x="{zero_x + origin_label_x_offset:.2f}" y="{zero_y - origin_label_x_offset:.2f}" text-anchor="start" dominant-baseline="baseline" '
            f'font-size="14" fill="#991b1b">原点 (0, 0)</text>',
        ]

        for i, vector in enumerate(self._vectors):
            x = sx(float(vector.x))
            y = sy(float(vector.y))
            label = self._escape(self._labels[i])
            svg_parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{point_radius:.2f}" fill="#dbeafe" stroke="#2563eb" stroke-width="2" opacity="0.90" />'
            )
            svg_parts.append(
                f'<text x="{x:.2f}" y="{y - label_y_offset:.2f}" text-anchor="middle" dominant-baseline="middle" '
                f'font-size="13" fill="#1e3a8a">{label}</text>'
            )
            svg_parts.append(
                f'<text x="{x:.2f}" y="{y + vector_text_y_offset:.2f}" text-anchor="middle" dominant-baseline="middle" '
                f'font-size="11" fill="#1f2937">({vector.x:.2f}, {vector.y:.2f})</text>'
            )

        svg_parts.append("</svg>")
        output.write_text("\n".join(svg_parts), encoding="utf-8")
        return output

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )


# Backward compatible alias for existing imports.
TeamFeatureVectorDirectedGraphPlotter = TeamFeatureVectorScatterPlotter
