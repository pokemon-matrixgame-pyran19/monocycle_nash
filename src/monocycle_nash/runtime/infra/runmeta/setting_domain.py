from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunMetaSetting:
    sqlite_path: str = ".runmeta/run_history.db"


@dataclass(frozen=True)
class OutputSetting:
    base_dir: str = "results"


@dataclass(frozen=True)
class AnalysisProjectSetting:
    project_id: str | None = None
    project_path: str | None = None


@dataclass(frozen=True)
class RuntimeSetting:
    runmeta: RunMetaSetting = RunMetaSetting()
    output: OutputSetting = OutputSetting()
    analysis_project: AnalysisProjectSetting = AnalysisProjectSetting()

    def to_toml_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "runmeta": {"sqlite_path": self.runmeta.sqlite_path},
            "output": {"base_dir": self.output.base_dir},
        }
        analysis_project: dict[str, str] = {}
        if self.analysis_project.project_id is not None:
            analysis_project["project_id"] = self.analysis_project.project_id
        if self.analysis_project.project_path is not None:
            analysis_project["project_path"] = self.analysis_project.project_path
        if analysis_project:
            payload["analysis_project"] = analysis_project
        return payload


class RuntimeSettingParser(ABC):
    @abstractmethod
    def parse(self, raw_setting: dict[str, Any]) -> RuntimeSetting:
        raise NotImplementedError
