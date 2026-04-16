"""
TOML木構造ローダー。

`$ref` を再帰的に解決して、1つの辞書へ展開する。
"""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any


class TomlTreeLoader:
    """`$ref` を解決しながら TOML を読み込む。"""

    REF_KEY = "$ref"

    def load(self, root_file: Path | str) -> dict[str, Any]:
        """入口ファイルから木構造を読み込む。"""
        root_path = Path(root_file)
        data = self.load_toml(root_path)
        return self._resolve_node(
            data,
            base_dir=root_path.parent,
            current_key=None,
            visited=[root_path.resolve()],
        )

    def load_toml(self, file_path: Path | str) -> dict[str, Any]:
        """単一 TOML ファイルを辞書として読み込む。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"TOMLファイルが見つかりません: {path}")
        with path.open("rb") as f:
            data = tomllib.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"TOMLのトップレベルはテーブル(dict)である必要があります: {path}")
        return data

    def _resolve_node(
        self,
        node: Any,
        base_dir: Path,
        current_key: str | None,
        visited: list[Path],
    ) -> Any:
        if isinstance(node, dict):
            if self.REF_KEY in node:
                return self._resolve_ref_node(
                    node=node,
                    base_dir=base_dir,
                    current_key=current_key,
                    visited=visited,
                )
            resolved: dict[Any, Any] = {}
            for key, value in node.items():
                key_name = key if isinstance(key, str) else None
                resolved[key] = self._resolve_node(
                    node=value,
                    base_dir=base_dir,
                    current_key=key_name,
                    visited=visited,
                )
            return resolved
        if isinstance(node, list):
            return [
                self._resolve_node(
                    node=item,
                    base_dir=base_dir,
                    current_key=None,
                    visited=visited,
                )
                for item in node
            ]
        return node

    def _resolve_ref_node(
        self,
        node: dict[Any, Any],
        base_dir: Path,
        current_key: str | None,
        visited: list[Path],
    ) -> Any:
        if len(node) != 1:
            raise ValueError(
                f"$refノードは {self.REF_KEY} だけを含めてください。"
                f" keys={list(node.keys())}"
            )
        ref_value = node.get(self.REF_KEY)
        if not isinstance(ref_value, str) or not ref_value.strip():
            raise ValueError(f"$ref は空でない文字列で指定してください: {ref_value!r}")
        ref_path = self._resolve_ref_path(ref_value.strip(), base_dir, current_key)
        ref_resolved = ref_path.resolve()
        if ref_resolved in visited:
            chain = " -> ".join(str(p) for p in visited + [ref_resolved])
            raise ValueError(f"$ref の循環参照を検出しました: {chain}")

        ref_data = self.load_toml(ref_path)
        return self._resolve_node(
            node=ref_data,
            base_dir=ref_path.parent,
            current_key=None,
            visited=[*visited, ref_resolved],
        )

    def _resolve_ref_path(self, ref_name: str, base_dir: Path, current_key: str | None) -> Path:
        ref = Path(ref_name)
        if ref.is_absolute() or self._is_explicit_relative(ref_name):
            candidate = ref if ref.is_absolute() else (base_dir / ref)
            candidate = self._append_toml_suffix(candidate)
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"$ref参照先が見つかりません: {candidate}")

        candidates: list[Path] = []
        if current_key:
            candidates.append(base_dir / current_key / f"{ref_name}.toml")
        candidates.append(base_dir / f"{ref_name}.toml")

        for candidate in candidates:
            if candidate.exists():
                return candidate

        hint = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"$ref参照先が見つかりません: {ref_name} (候補: {hint})")

    @staticmethod
    def _is_explicit_relative(ref_name: str) -> bool:
        return "/" in ref_name or "\\" in ref_name or ref_name.endswith(".toml")

    @staticmethod
    def _append_toml_suffix(path: Path) -> Path:
        if path.suffix:
            return path
        return path.with_suffix(".toml")
