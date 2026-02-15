"""Filesystem artifact storage for run result folders."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .models import InputSnapshotInfo


@dataclass
class RunArtifactStore:
    result_root: Path = Path("results")

    def __post_init__(self) -> None:
        self.result_root = Path(self.result_root)

    def run_dir(self, run_id: int) -> Path:
        return self.result_root / str(run_id)

    def create_run_dir(self, run_id: int) -> Path:
        run_dir = self.run_dir(run_id)
        (run_dir / "input").mkdir(parents=True, exist_ok=True)
        (run_dir / "output").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_input_snapshot(
        self,
        run_id: int,
        src_input_path: Path | str,
        merged_input_bytes: bytes | None = None,
    ) -> InputSnapshotInfo:
        src = Path(src_input_path)
        input_dir = self.create_run_dir(run_id) / "input"

        if src.is_dir():
            dst = input_dir / src.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            checksum = None
            source_path = str(src)
        elif src.is_file():
            dst = input_dir / src.name
            shutil.copy2(src, dst)
            checksum = hashlib.sha256(dst.read_bytes()).hexdigest()
            source_path = str(src)
        else:
            raise FileNotFoundError(f"Input source not found: {src}")

        stored_file = None
        split_config_applied = False
        if merged_input_bytes is not None:
            stored = input_dir / "merged_input.toml"
            stored.write_bytes(merged_input_bytes)
            stored_file = str(Path("input") / "merged_input.toml")
            split_config_applied = True
            checksum = hashlib.sha256(merged_input_bytes).hexdigest()

        return InputSnapshotInfo(
            source_path=source_path,
            split_config_applied=split_config_applied,
            stored_file=stored_file,
            checksum_sha256=checksum,
        )

    def write_initial_meta(self, run_id: int, payload: Mapping[str, object]) -> None:
        self._write_meta(run_id, payload)

    def write_final_meta(self, run_id: int, payload: Mapping[str, object]) -> None:
        self._write_meta(run_id, payload)

    def delete_run_dir(self, run_id: int) -> None:
        run_dir = self.run_dir(run_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)

    def _write_meta(self, run_id: int, payload: Mapping[str, object]) -> None:
        meta_path = self.create_run_dir(run_id) / "meta.json"
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
