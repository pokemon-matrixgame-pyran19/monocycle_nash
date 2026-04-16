"""
TOML木構造ローダー。

`$ref` を再帰的に解決して、1つの辞書へ展開する。
既存の TomlTreeLoader を簡素化した実装。
"""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any


REF_KEY = "$ref"


class TomlTreeLoader:
    """`$ref` を解決しながら TOML を読み込む。"""

    def load(self, root_file: Path | str) -> dict[str, Any]:
        """入口ファイルから木構造を読み込む。"""
        root_path = Path(root_file)
        data = self._load_single(root_path)
        return self._resolve(data, base_dir=root_path.parent, visited=[root_path.resolve()])

    def _load_single(self, file_path: Path) -> dict[str, Any]:
        """単一 TOML ファイルを辞書として読み込む。"""
        if not file_path.exists():
            raise FileNotFoundError(f"TOMLファイルが見つかりません: {file_path}")
        with file_path.open("rb") as f:
            data = tomllib.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"TOMLのトップレベルはテーブルである必要があります: {file_path}")
        return data

    def _resolve(
        self,
        node: Any,
        base_dir: Path,
        visited: list[Path],
    ) -> Any:
        if isinstance(node, dict):
            if REF_KEY in node:
                return self._resolve_ref(node, base_dir, visited)
            return {
                key: self._resolve(value, base_dir, visited)
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [self._resolve(item, base_dir, visited) for item in node]
        return node

    def _resolve_ref(
        self,
        node: dict[str, Any],
        base_dir: Path,
        visited: list[Path],
    ) -> Any:
        if len(node) != 1:
            raise ValueError(
                f"$refノードは {REF_KEY} だけを含めてください。 keys={list(node.keys())}"
            )
        ref_value = node[REF_KEY]
        if not isinstance(ref_value, str) or not ref_value.strip():
            raise ValueError(f"$ref は空でない文字列で指定してください: {ref_value!r}")

        ref_path = self._find_ref_path(ref_value.strip(), base_dir)
        ref_resolved = ref_path.resolve()
        if ref_resolved in visited:
            chain = " -> ".join(str(p) for p in [*visited, ref_resolved])
            raise ValueError(f"$ref の循環参照を検出しました: {chain}")

        ref_data = self._load_single(ref_path)
        return self._resolve(ref_data, ref_path.parent, [*visited, ref_resolved])

    @staticmethod
    def _find_ref_path(ref_name: str, base_dir: Path) -> Path:
        """$ref の参照先パスを解決する。"""
        ref = Path(ref_name)
        # 明示的な相対パスまたは絶対パス
        if ref.is_absolute() or "/" in ref_name or "\\" in ref_name or ref_name.endswith(".toml"):
            candidate = ref if ref.is_absolute() else (base_dir / ref)
            if not candidate.suffix:
                candidate = candidate.with_suffix(".toml")
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"$ref参照先が見つかりません: {candidate}")

        # 短縮名: base_dir/<name>.toml
        candidate = base_dir / f"{ref_name}.toml"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"$ref参照先が見つかりません: {ref_name} (候補: {candidate})")
