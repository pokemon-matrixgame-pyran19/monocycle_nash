"""Artifact storage for run metadata."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArtifactStore:
    """In-memory artifact key registry per run."""

    _artifacts: dict[str, list[str]] = field(default_factory=dict)

    def add(self, run_id: str, artifact_uri: str) -> None:
        """Register an artifact URI for a run."""

        self._artifacts.setdefault(run_id, []).append(artifact_uri)

    def list_for_run(self, run_id: str) -> list[str]:
        """Return artifact URIs associated with a run."""

        return list(self._artifacts.get(run_id, []))
