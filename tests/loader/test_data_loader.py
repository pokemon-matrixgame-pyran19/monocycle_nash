from __future__ import annotations

from pathlib import Path

import pytest

from monocycle_nash.loader import ExperimentDataLoader, SettingDataLoader, TomlTreeLoader


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


class TestTomlTreeLoader:
    def test_load_single_file(self, tmp_path: Path):
        root = tmp_path / "data.toml"
        _write(
            root,
            """
            x = 1
            y = 3
            """,
        )

        loader = TomlTreeLoader()
        result = loader.load(root)

        assert result == {"x": 1, "y": 3}

    def test_resolve_ref_by_key_directory(self, tmp_path: Path):
        root = tmp_path / "data.toml"
        _write(
            root,
            """
            x = 1
            [sub_state]
            "$ref" = "nash"
            """,
        )
        _write(
            tmp_path / "sub_state" / "nash.toml",
            """
            z = 2
            w = 6
            """,
        )

        loader = TomlTreeLoader()
        result = loader.load(root)

        assert result == {"x": 1, "sub_state": {"z": 2, "w": 6}}

    def test_resolve_nested_ref(self, tmp_path: Path):
        root = tmp_path / "data.toml"
        _write(
            root,
            """
            [sub_state]
            "$ref" = "nash"
            """,
        )
        _write(
            tmp_path / "sub_state" / "nash.toml",
            """
            [detail]
            "$ref" = "inner"
            """,
        )
        _write(
            tmp_path / "sub_state" / "detail" / "inner.toml",
            """
            value = 42
            """,
        )

        loader = TomlTreeLoader()
        result = loader.load(root)

        assert result == {"sub_state": {"detail": {"value": 42}}}

    def test_ref_node_with_extra_keys_raises(self, tmp_path: Path):
        root = tmp_path / "data.toml"
        _write(
            root,
            """
            [sub_state]
            "$ref" = "nash"
            value = 1
            """,
        )

        loader = TomlTreeLoader()
        with pytest.raises(ValueError, match="\\$refノード"):
            loader.load(root)

    def test_cycle_reference_raises(self, tmp_path: Path):
        root = tmp_path / "data.toml"
        _write(
            root,
            """
            [sub]
            "$ref" = "loop"
            """,
        )
        _write(
            tmp_path / "sub" / "loop.toml",
            """
            "$ref" = "../data"
            """,
        )

        loader = TomlTreeLoader()
        with pytest.raises(ValueError, match="循環参照"):
            loader.load(root)


class TestExperimentDataLoader:
    def test_load(self, tmp_path: Path):
        _write(
            tmp_path / "data" / "spam" / "janken" / "data.toml",
            """
            x = 1
            [sub_state]
            "$ref" = "nash"
            """,
        )
        _write(
            tmp_path / "data" / "spam" / "janken" / "sub_state" / "nash.toml",
            """
            z = 2
            """,
        )

        loader = ExperimentDataLoader(base_dir=tmp_path / "data")
        result = loader.load("spam", "janken")

        assert result == {"x": 1, "sub_state": {"z": 2}}

    def test_load_from_path_directory(self, tmp_path: Path):
        _write(
            tmp_path / "data" / "spam" / "janken" / "data.toml",
            """
            x = 10
            """,
        )

        loader = ExperimentDataLoader(base_dir=tmp_path / "data")
        result = loader.load_from_path("spam/janken")

        assert result == {"x": 10}


class TestSettingDataLoader:
    def test_load_name_without_suffix(self, tmp_path: Path):
        _write(
            tmp_path / "data" / "setting" / "runtime.toml",
            """
            seed = 123
            """,
        )

        loader = SettingDataLoader(base_dir=tmp_path / "data" / "setting")
        result = loader.load("runtime")

        assert result == {"seed": 123}

    def test_load_file_relative_path(self, tmp_path: Path):
        _write(
            tmp_path / "data" / "setting" / "gpu" / "local.toml",
            """
            enabled = true
            """,
        )

        loader = SettingDataLoader(base_dir=tmp_path / "data" / "setting")
        result = loader.load_file("gpu/local.toml")

        assert result == {"enabled": True}
