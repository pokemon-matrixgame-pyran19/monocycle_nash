from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from monocycle_nash.matrix.base import PayoffMatrix


@dataclass(frozen=True)
class ApproximationConfig:
    source_matrix_name: str | None
    reference_matrix_name: str | None
    approximation_name: str
    distance_name: str
    raw_input: dict


@dataclass(frozen=True)
class RandomMatrixConfig:
    size: int
    generation_count: int
    acceptance_condition: str
    low: float
    high: float
    max_attempts: int
    random_seed: int | None
    raw_input: dict


@dataclass(frozen=True)
class PayoffGraphConfig:
    threshold: float
    canvas_size: int


@dataclass(frozen=True)
class CharacterGraphConfig:
    canvas_size: int
    margin: int


@dataclass(frozen=True)
class LoadedFeatureInputs:
    matrix: PayoffMatrix
    graph_config: PayoffGraphConfig | CharacterGraphConfig | None
    approximation_config: ApproximationConfig | None
    random_matrix_config: RandomMatrixConfig | None
    setting_data: dict


class FeatureWorkflowInputPort(Protocol):
    def load_features(self) -> list[str]: ...

    def load_inputs_for_feature(self, feature: str) -> LoadedFeatureInputs: ...

    def load_matrix(self, matrix_name: str) -> PayoffMatrix: ...
