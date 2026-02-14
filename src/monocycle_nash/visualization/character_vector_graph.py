from __future__ import annotations

from pathlib import Path

from monocycle_nash.character import Character


class CharacterVectorGraphPlotter:
    """単相性モデルのキャラクター配列から相性ベクトル散布図(SVG)を生成する。"""

    def __init__(self, characters: list[Character]):
        if not characters:
            raise ValueError("characters は1件以上必要です")
        self._characters = characters

    def draw(self, output_path: str | Path, canvas_size: int = 840, margin: int = 90) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        xs = [float(c.v.x) for c in self._characters]
        ys = [float(c.v.y) for c in self._characters]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # 表示領域が狭い場合にも一定の余白を確保する
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        pad_x = span_x * 0.2
        pad_y = span_y * 0.2
        world_min_x, world_max_x = min_x - pad_x, max_x + pad_x
        world_min_y, world_max_y = min_y - pad_y, max_y + pad_y

        # 原点(0, 0)は相性ベクトル解釈で重要なので、必ず表示範囲に含める
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
            # SVGはy軸が下向きのため反転
            return height - margin - (y - world_min_y) / (world_max_y - world_min_y) * inner_h

        powers = [float(c.p) for c in self._characters]
        min_p, max_p = min(powers), max(powers)

        def radius(p: float) -> float:
            if max_p == min_p:
                return 28.0
            norm = (p - min_p) / (max_p - min_p)
            return 18.0 + norm * 28.0

        axis_x0 = sx(world_min_x)
        axis_x1 = sx(world_max_x)
        axis_y0 = sy(world_min_y)
        axis_y1 = sy(world_max_y)
        zero_x = sx(0.0)
        zero_y = sy(0.0)

        svg_parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white" />',
            f'<line x1="{axis_x0:.2f}" y1="{axis_y0:.2f}" x2="{axis_x1:.2f}" y2="{axis_y0:.2f}" stroke="#d1d5db" stroke-width="1" />',
            f'<line x1="{axis_x0:.2f}" y1="{axis_y1:.2f}" x2="{axis_x0:.2f}" y2="{axis_y0:.2f}" stroke="#d1d5db" stroke-width="1" />',
        ]

        svg_parts.append(
            f'<line x1="{axis_x0:.2f}" y1="{zero_y:.2f}" x2="{axis_x1:.2f}" y2="{zero_y:.2f}" stroke="#6b7280" stroke-width="1.5" />'
        )
        svg_parts.append(
            f'<line x1="{zero_x:.2f}" y1="{axis_y1:.2f}" x2="{zero_x:.2f}" y2="{axis_y0:.2f}" stroke="#6b7280" stroke-width="1.5" />'
        )
        svg_parts.append(
            f'<circle cx="{zero_x:.2f}" cy="{zero_y:.2f}" r="5.50" fill="#ef4444" stroke="white" stroke-width="1.5" />'
        )
        svg_parts.append(
            f'<text x="{zero_x + 10:.2f}" y="{zero_y - 10:.2f}" text-anchor="start" dominant-baseline="baseline" '
            f'font-size="14" fill="#991b1b">原点 (0, 0)</text>'
        )

        for c in self._characters:
            x = sx(float(c.v.x))
            y = sy(float(c.v.y))
            r = radius(float(c.p))
            label = self._escape(c.label)

            svg_parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="#dbeafe" stroke="#2563eb" stroke-width="2" opacity="0.88" />'
            )
            svg_parts.append(
                f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="middle" dominant-baseline="middle" '
                f'font-size="14" fill="#1e3a8a">{label}</text>'
            )

        svg_parts.append('</svg>')
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
