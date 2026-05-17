from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from monocycle_nash.application.matrix_nodes import (
    GeneralFromTeamMatchupsNode,
    NodeResolutionContext,
    OutputEmission,
    OutputNode,
)
from monocycle_nash.application.node_spec import OutputSpec
from monocycle_nash.application.ports import OutputPathPort
from monocycle_nash.domain.matrix.base import PayoffMatrix
from monocycle_nash.domain.team import Team
from monocycle_nash.domain.visualization import TeamFeatureVectorScatterPlotter


@dataclass(frozen=True)
class TeamMatchupExperimentCsvOutputNode(
    OutputNode["GeneralFromTeamMatchupsNode"],
    output_method="team_matchup_experiment_csv",
):
    """固定チーム i と各チーム j の比較情報を CSV 出力する実験ノード。"""

    runner: str | None = None
    filename: str = "team_matchup_experiment.csv"
    focus_team: int | str = 0

    @classmethod
    def _from_output_spec(cls, spec: OutputSpec) -> "TeamMatchupExperimentCsvOutputNode":
        return cls(
            runner=spec.runner,
            filename=spec.params.get("filename", "team_matchup_experiment.csv"),
            focus_team=spec.params.get("focus_team", 0),
        )

    def emit(
        self,
        *,
        node_name: str,
        node_path: tuple[str, ...],
        node: "GeneralFromTeamMatchupsNode",
    ) -> OutputEmission:
        return OutputEmission(
            output_node=self,
            node_name=node_name,
            node_path=node_path,
            runner=self.runner,
            node=node,
        )

    def execute(
        self,
        *,
        output_path_port: OutputPathPort,
        run_id: str,
        node_path: tuple[str, ...],
        node: "GeneralFromTeamMatchupsNode",
        ctx: NodeResolutionContext,
    ) -> Path:
        path = output_path_port.resolve_output_path(
            run_id=run_id,
            node_path=node_path,
            output_method=self.output_method,
            filename=self.filename,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        team_matrix = node.provide_object(ctx=ctx)
        character_matrix = node.resolve_character_matrix(ctx=ctx)
        teams = node.teams.get_teams(ctx=ctx)
        focus_index = self._resolve_focus_team_index(teams)
        focus_team = teams[focus_index]
        i1, i2 = self._resolve_team_member_indices(focus_team, character_matrix)
        i1_label, i1_vec = self._resolve_strategy_label_and_vector(character_matrix, i1)
        i2_label, i2_vec = self._resolve_strategy_label_and_vector(character_matrix, i2)
        base_diff = (i1_vec[0] - i2_vec[0], i1_vec[1] - i2_vec[1])

        rows: list[dict[str, str | int | float]] = []
        for j, team_j in enumerate(teams):
            if j == focus_index:
                continue
            j1, j2 = self._resolve_team_member_indices(team_j, character_matrix)
            j1_label, j1_vec = self._resolve_strategy_label_and_vector(character_matrix, j1)
            j2_label, j2_vec = self._resolve_strategy_label_and_vector(character_matrix, j2)
            (j3_label, j3_vec), (j4_label, j4_vec) = self._to_counter_clockwise_pair(
                (j1_label, j1_vec), (j2_label, j2_vec)
            )
            j_diff = (j3_vec[0] - j4_vec[0], j3_vec[1] - j4_vec[1])
            angle_rad = self._signed_angle(base_diff, j_diff)
            bij = float(team_matrix.matrix[focus_index, j])
            rows.append(
                {
                    "fixed_team_index": focus_index,
                    "fixed_team_label": focus_team.label,
                    "fixed_member_1_label": i1_label,
                    "fixed_member_1_x": i1_vec[0],
                    "fixed_member_1_y": i1_vec[1],
                    "fixed_member_2_label": i2_label,
                    "fixed_member_2_x": i2_vec[0],
                    "fixed_member_2_y": i2_vec[1],
                    "j_team_index": j,
                    "j_team_label": team_j.label,
                    "j3_label": j3_label,
                    "j3_x": j3_vec[0],
                    "j3_y": j3_vec[1],
                    "j4_label": j4_label,
                    "j4_x": j4_vec[0],
                    "j4_y": j4_vec[1],
                    "j3_minus_j4_x": j_diff[0],
                    "j3_minus_j4_y": j_diff[1],
                    "angle_v12_to_v34_rad": angle_rad,
                    "angle_v12_to_v34_deg": math.degrees(angle_rad),
                    "v34_is_counter_clockwise_from_v12": angle_rad > 0.0,
                    "bij": bij,
                }
            )

        fieldnames = [
            "fixed_team_index",
            "fixed_team_label",
            "fixed_member_1_label",
            "fixed_member_1_x",
            "fixed_member_1_y",
            "fixed_member_2_label",
            "fixed_member_2_x",
            "fixed_member_2_y",
            "j_team_index",
            "j_team_label",
            "j3_label",
            "j3_x",
            "j3_y",
            "j4_label",
            "j4_x",
            "j4_y",
            "j3_minus_j4_x",
            "j3_minus_j4_y",
            "angle_v12_to_v34_rad",
            "angle_v12_to_v34_deg",
            "v34_is_counter_clockwise_from_v12",
            "bij",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _resolve_focus_team_index(self, teams: tuple[Team, ...]) -> int:
        if isinstance(self.focus_team, int):
            if not 0 <= self.focus_team < len(teams):
                raise IndexError(f"focus_team index out of range: {self.focus_team}")
            return self.focus_team
        for i, team in enumerate(teams):
            if team.label == self.focus_team:
                return i
        raise ValueError(f"focus_team label not found: {self.focus_team}")

    @staticmethod
    def _resolve_team_member_indices(team: Team, character_matrix: PayoffMatrix) -> tuple[int, int]:
        indices = team.resolve_member_indices(character_matrix.row_strategies)
        if len(indices) != 2:
            raise ValueError(
                "team_matchup_experiment_csv は 2匹チームのみ対応します: "
                f"{team.label} has {len(indices)} members"
            )
        return indices[0], indices[1]

    @staticmethod
    def _resolve_strategy_label_and_vector(
        character_matrix: PayoffMatrix,
        index: int,
    ) -> tuple[str, tuple[float, float]]:
        strategy = character_matrix.row_strategies[index]
        if strategy.vector is None:
            raise TypeError("team_matchup_experiment_csv には Character 戦略が必要です")
        return strategy.label, (float(strategy.vector.x), float(strategy.vector.y))

    @staticmethod
    def _to_counter_clockwise_pair(
        left: tuple[str, tuple[float, float]],
        right: tuple[str, tuple[float, float]],
    ) -> tuple[tuple[str, tuple[float, float]], tuple[str, tuple[float, float]]]:
        angle = TeamMatchupExperimentCsvOutputNode._signed_angle(left[1], right[1])
        if angle > 0:
            return left, right
        if angle < 0:
            return right, left
        left_theta = math.atan2(left[1][1], left[1][0])
        right_theta = math.atan2(right[1][1], right[1][0])
        if left_theta <= right_theta:
            return left, right
        return right, left

    @staticmethod
    def _signed_angle(base: tuple[float, float], target: tuple[float, float]) -> float:
        base_x, base_y = base
        target_x, target_y = target
        return math.atan2(base_x * target_y - base_y * target_x, base_x * target_x + base_y * target_y)


@dataclass(frozen=True)
class TeamFeatureVectorCsvOutputNode(
    OutputNode["GeneralFromTeamMatchupsNode"],
    output_method="team_feature_vector_csv",
):
    """各チームの特徴ベクトルを CSV 出力する。"""

    runner: str | None = None
    filename: str = "team_feature_vectors.csv"

    @classmethod
    def _from_output_spec(cls, spec: OutputSpec) -> "TeamFeatureVectorCsvOutputNode":
        return cls(
            runner=spec.runner,
            filename=spec.params.get("filename", "team_feature_vectors.csv"),
        )

    def emit(
        self,
        *,
        node_name: str,
        node_path: tuple[str, ...],
        node: "GeneralFromTeamMatchupsNode",
    ) -> OutputEmission:
        return OutputEmission(
            output_node=self,
            node_name=node_name,
            node_path=node_path,
            runner=self.runner,
            node=node,
        )

    def execute(
        self,
        *,
        output_path_port: OutputPathPort,
        run_id: str,
        node_path: tuple[str, ...],
        node: "GeneralFromTeamMatchupsNode",
        ctx: NodeResolutionContext,
    ) -> Path:
        path = output_path_port.resolve_output_path(
            run_id=run_id,
            node_path=node_path,
            output_method=self.output_method,
            filename=self.filename,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        character_matrix = node.resolve_character_matrix(ctx=ctx)
        teams = node.teams.get_teams(ctx=ctx)
        rows: list[dict[str, str | int | float]] = []
        for i, team in enumerate(teams):
            i1, i2 = self._resolve_team_member_indices(team, character_matrix)
            i1_label, i1_vec = self._resolve_strategy_label_and_vector(
                character_matrix,
                i1,
            )
            i2_label, i2_vec = self._resolve_strategy_label_and_vector(
                character_matrix,
                i2,
            )
            feature = team.calculate_feature_vector(character_matrix.row_strategies)
            rows.append(
                {
                    "team_index": i,
                    "team_label": team.label,
                    "member_1_label": i1_label,
                    "member_1_x": i1_vec[0],
                    "member_1_y": i1_vec[1],
                    "member_2_label": i2_label,
                    "member_2_x": i2_vec[0],
                    "member_2_y": i2_vec[1],
                    "feature_x": feature.x,
                    "feature_y": feature.y,
                    "feature_distance": feature.distance_from_origin,
                    "feature_angle_rad": feature.angle_rad,
                    "feature_angle_deg": feature.angle_deg,
                }
            )

        fieldnames = [
            "team_index",
            "team_label",
            "member_1_label",
            "member_1_x",
            "member_1_y",
            "member_2_label",
            "member_2_x",
            "member_2_y",
            "feature_x",
            "feature_y",
            "feature_distance",
            "feature_angle_rad",
            "feature_angle_deg",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    @staticmethod
    def _resolve_team_member_indices(team: Team, character_matrix: PayoffMatrix) -> tuple[int, int]:
        indices = team.resolve_member_indices(character_matrix.row_strategies)
        if len(indices) != 2:
            raise ValueError(
                "team_feature_vector_csv only supports 2-member teams: "
                f"{team.label} has {len(indices)} members"
            )
        return indices[0], indices[1]

    @staticmethod
    def _resolve_strategy_label_and_vector(
        character_matrix: PayoffMatrix,
        index: int,
    ) -> tuple[str, tuple[float, float]]:
        strategy = character_matrix.row_strategies[index]
        if strategy.vector is None:
            raise TypeError("team_feature_vector_csv requires character strategies with vectors")
        return strategy.label, (float(strategy.vector.x), float(strategy.vector.y))


@dataclass(frozen=True)
class TeamFeatureVectorScatterPlotOutputNode(
    OutputNode["GeneralFromTeamMatchupsNode"],
    output_method="team_feature_vector_directed_graph",
):
    """各チームの特徴ベクトルを2次元散布図として SVG 出力する。"""

    runner: str | None = None
    filename: str = "team_feature_vector_scatter_plot.svg"
    canvas_size: int = 840

    @classmethod
    def _from_output_spec(cls, spec: OutputSpec) -> "TeamFeatureVectorScatterPlotOutputNode":
        return cls(
            runner=spec.runner,
            filename=spec.params.get("filename", "team_feature_vector_scatter_plot.svg"),
            canvas_size=spec.params.get("canvas_size", 840),
        )

    def emit(
        self,
        *,
        node_name: str,
        node_path: tuple[str, ...],
        node: "GeneralFromTeamMatchupsNode",
    ) -> OutputEmission:
        return OutputEmission(
            output_node=self,
            node_name=node_name,
            node_path=node_path,
            runner=self.runner,
            node=node,
        )

    def execute(
        self,
        *,
        output_path_port: OutputPathPort,
        run_id: str,
        node_path: tuple[str, ...],
        node: "GeneralFromTeamMatchupsNode",
        ctx: NodeResolutionContext,
    ) -> Path:
        path = output_path_port.resolve_output_path(
            run_id=run_id,
            node_path=node_path,
            output_method=self.output_method,
            filename=self.filename,
        )
        character_matrix = node.resolve_character_matrix(ctx=ctx)
        teams = node.teams.get_teams(ctx=ctx)
        labels = [team.label for team in teams]
        vectors = [team.calculate_feature_vector(character_matrix.row_strategies) for team in teams]
        TeamFeatureVectorScatterPlotter(
            labels=labels,
            vectors=vectors,
        ).draw(path, canvas_size=self.canvas_size)
        return path


# Backward compatible alias for existing imports.
TeamFeatureVectorDirectedGraphOutputNode = TeamFeatureVectorScatterPlotOutputNode
