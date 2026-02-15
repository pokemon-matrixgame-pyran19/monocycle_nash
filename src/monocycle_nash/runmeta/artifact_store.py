"""Artifact storage for run metadata."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass
class RunArtifactStore:
    """Filesystem-backed artifact store for each run."""

    result_root: Path | str = Path("result")

    def __post_init__(self) -> None:
        self.result_root = Path(self.result_root)

    def run_dir(self, run_id: int | str) -> Path:
        return self.result_root / str(run_id)

    def create_run_dir(self, run_id: int | str) -> Path:
        """Create `<result>/<run_id>/input|output|logs` if missing."""

        run_dir = self.run_dir(run_id)
        (run_dir / "input").mkdir(parents=True, exist_ok=True)
        (run_dir / "output").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        return run_dir

    def write_meta(self, run_id: int | str, meta: Mapping[str, object]) -> Path:
        """Write `<result>/<run_id>/meta.json`."""

        run_dir = self.create_run_dir(run_id)
        meta_path = run_dir / "meta.json"
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return meta_path

    def write_input_file(self, run_id: int | str, name: str, content: str | bytes) -> Path:
        return self._write_subfile(run_id, "input", name, content)

    def write_output_file(self, run_id: int | str, name: str, content: str | bytes) -> Path:
        return self._write_subfile(run_id, "output", name, content)

    def write_log_file(self, run_id: int | str, name: str, content: str | bytes) -> Path:
        return self._write_subfile(run_id, "logs", name, content)

    def save_input_snapshot(
        self,
        run_id: int | str,
        source_paths: list[Path | str],
        *,
        merged_input_content: str | bytes | None = None,
        merged_input_name: str = "merged_input.toml",
    ) -> list[Path]:
        """Copy immutable input snapshots and optionally persist merged input."""

        input_dir = self.create_run_dir(run_id) / "input"
        copied_paths: list[Path] = []

        for source in source_paths:
            src = Path(source)
            if not src.exists():
                raise FileNotFoundError(f"Input source does not exist: {src}")

            target_name = self._dedup_target_name(input_dir, src.name)
            dst = input_dir / target_name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            copied_paths.append(dst)

        if merged_input_content is not None:
            copied_paths.append(
                self._write_subfile(run_id, "input", merged_input_name, merged_input_content)
            )

        return copied_paths

    def delete_run_artifacts(self, run_id: int | str) -> None:
        """Delete `<result>/<run_id>` recursively."""

        run_dir = self.run_dir(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)

    def _write_subfile(
        self,
        run_id: int | str,
        kind: str,
        name: str,
        content: str | bytes,
    ) -> Path:
        path = self.create_run_dir(run_id) / kind / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    @staticmethod
    def _dedup_target_name(parent: Path, file_name: str) -> str:
        candidate = parent / file_name
        if not candidate.exists():
            return file_name

        stem = candidate.stem
        suffix = candidate.suffix
        index = 2
        while True:
            renamed = f"{stem}_{index}{suffix}"
            if not (parent / renamed).exists():
                return renamed
            index += 1

