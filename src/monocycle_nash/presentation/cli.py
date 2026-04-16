"""CLI entry point and composition root for monocycle_nash.

対話的にユースケースを選択して実行する。
依存関係のワイヤリング（組み立て）はここで行う。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from monocycle_nash import __version__
from monocycle_nash.application.comparison import ComparisonUseCase
from monocycle_nash.application.direct_analysis import DirectAnalysisUseCase
from monocycle_nash.application.draw_character_plot import DrawCharacterPlotUseCase
from monocycle_nash.application.draw_payoff_graph import DrawPayoffGraphUseCase
from monocycle_nash.application.dto import RandomExperimentConfig
from monocycle_nash.application.matrix_build import MatrixBuildUseCase
from monocycle_nash.application.matrix_build_from_characters import BuildMatrixFromCharactersUseCase
from monocycle_nash.application.matrix_build_from_raw import BuildMatrixFromRawUseCase
from monocycle_nash.application.random_experiment import RandomExperimentUseCase
from monocycle_nash.application.solve_equilibrium import SolveEquilibriumUseCase
from monocycle_nash.infrastructure.input.config_reader import (
    FileExperimentDataReader,
    FileGraphConfigReader,
)
from monocycle_nash.infrastructure.input.matrix_reader import FileMatrixDataReader
from monocycle_nash.infrastructure.input.toml_loader import TomlTreeLoader
from monocycle_nash.infrastructure.output.run_context_factory import FileOutputAdapter
from monocycle_nash.infrastructure.visualization.adapter import SvgVisualizationAdapter

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _PROJECT_ROOT / "data" / "run_config" / "main.toml"
_HISTORY_FILE = _PROJECT_ROOT / ".monocycle_nash_history"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_run_config(config_path: Path) -> dict[str, Any]:
    """run_config TOML を読み込む。"""
    loader = TomlTreeLoader()
    return loader.load(config_path)


def _merge_feature_config(
    run_config: dict[str, Any],
    feature_name: str,
) -> dict[str, Any]:
    """shared セクションと feature 固有セクションをマージする。"""
    shared = dict(run_config.get("shared", {}))
    feature_section = dict(run_config.get(feature_name, {}))
    merged = {**shared, **feature_section}
    return merged


# ---------------------------------------------------------------------------
# Interactive selection
# ---------------------------------------------------------------------------


def _load_last_used(history_file: Path) -> str | None:
    """前回使用したフィーチャー名を読み込む。"""
    try:
        text = history_file.read_text(encoding="utf-8").strip()
        return text if text else None
    except FileNotFoundError:
        return None


def _save_last_used(history_file: Path, feature_name: str) -> None:
    """前回使用したフィーチャー名を保存する。"""
    try:
        history_file.write_text(feature_name + "\n", encoding="utf-8")
    except OSError:
        pass


def interactive_select(features: list[str], history_file: Path) -> str:
    """対話的にユースケースを選択する。"""
    last_used = _load_last_used(history_file)

    ordered = sorted(features, key=lambda f: f != last_used)

    print("実行するユースケースを選択してください:")
    for i, feature in enumerate(ordered):
        marker = " (前回)" if feature == last_used else ""
        print(f"  [{i + 1}] {feature}{marker}")

    if last_used and last_used in features:
        print(f"\n何も入力せずEnterで前回と同じ ({last_used}) を実行します")

    choice = input("\n> ").strip()

    if not choice and last_used and last_used in features:
        selected = last_used
    else:
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(ordered):
                raise IndexError
            selected = ordered[idx]
        except (ValueError, IndexError):
            print(f"無効な選択です: {choice}")
            return interactive_select(features, history_file)

    _save_last_used(history_file, selected)
    return selected


# ---------------------------------------------------------------------------
# Use-case runner (composition root logic)
# ---------------------------------------------------------------------------


class UseCaseRunner:
    """ユースケースの実行器 ― 依存をワイヤリングして feature を実行する。"""

    def __init__(self, data_dir: Path, result_dir: Path) -> None:
        self._data_dir = data_dir
        self._result_dir = result_dir

        # Infrastructure adapters
        self._matrix_reader = FileMatrixDataReader(data_dir)
        self._experiment_reader = FileExperimentDataReader(data_dir)
        self._graph_reader = FileGraphConfigReader(data_dir)
        self._output = FileOutputAdapter(result_dir)
        self._visualization = SvgVisualizationAdapter()

        # Mini use cases - matrix construction
        self._raw_uc = BuildMatrixFromRawUseCase()
        self._chars_uc = BuildMatrixFromCharactersUseCase()
        self._matrix_build_uc = MatrixBuildUseCase(
            port=self._matrix_reader,
            raw_uc=self._raw_uc,
            chars_uc=self._chars_uc,
        )

        # Mini use cases - analysis
        self._equilibrium_uc = SolveEquilibriumUseCase()
        self._payoff_graph_uc = DrawPayoffGraphUseCase(
            visualization=self._visualization,
            config_port=self._graph_reader,
        )
        self._character_plot_uc = DrawCharacterPlotUseCase(
            visualization=self._visualization,
            config_port=self._graph_reader,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_feature(self, feature_name: str, feature_config: dict[str, Any]) -> int:
        """指定されたフィーチャーを実行する。"""
        dispatch: dict[str, Any] = {
            "solve_payoff": self._run_direct_analysis,
            "graph_payoff": self._run_direct_analysis,
            "plot_characters": self._run_direct_analysis,
            "compare_approximation": self._run_comparison,
            "compare_product_formula": self._run_compare_product_formula,
            "compare_random_approximation": self._run_compare_random_approximation,
            "experiment_team_strict_spectrum": self._run_experiment,
        }
        handler = dispatch.get(feature_name)
        if handler is None:
            print(f"未対応の feature です: {feature_name}", file=sys.stderr)
            return 1
        return handler(feature_name, feature_config)

    # ------------------------------------------------------------------
    # Direct analysis  (solve_payoff / graph_payoff / plot_characters)
    # ------------------------------------------------------------------

    def _run_direct_analysis(
        self, feature_name: str, config: dict[str, Any],
    ) -> int:
        matrix_id = config.get("matrix")
        if matrix_id is None:
            print("設定に matrix が指定されていません", file=sys.stderr)
            return 1

        graph_id: str | None = config.get("graph")

        # 依存性注入でどの分析を行うかを決定する（AnalysisConfig フラグは使わない）
        equilibrium_uc = self._equilibrium_uc if feature_name == "solve_payoff" else None
        payoff_graph_uc = self._payoff_graph_uc if feature_name == "graph_payoff" else None
        character_plot_uc = (
            self._character_plot_uc if feature_name == "plot_characters" else None
        )

        uc = DirectAnalysisUseCase(
            matrix_build_uc=self._matrix_build_uc,
            output=self._output,
            equilibrium_uc=equilibrium_uc,
            payoff_graph_uc=payoff_graph_uc,
            character_plot_uc=character_plot_uc,
        )

        start = time.time()
        results = uc.execute(matrix_id, graph_id=graph_id)
        elapsed = time.time() - start

        for r in results:
            if r.equilibrium is not None:
                eq = r.equilibrium
                print(f"均衡解: {dict(zip(eq.mixed_strategy.strategy_ids, eq.mixed_strategy.probabilities.tolist(), strict=False))}")
            if r.payoff_graph_svg is not None:
                print("利得グラフ SVG を生成しました")
            if r.character_plot_svg is not None:
                print("キャラクタープロット SVG を生成しました")

        print(f"完了 ({elapsed:.2f}s)")
        return 0

    # ------------------------------------------------------------------
    # Comparison analysis  (compare_approximation)
    # ------------------------------------------------------------------

    def _run_comparison(
        self, feature_name: str, config: dict[str, Any],
    ) -> int:
        from monocycle_nash.domain.matrix.approximation import (
            MaxElementDifferenceDistance,
            MonocycleToGeneralApproximation,
        )

        matrix_id = config.get("matrix")
        if matrix_id is None:
            print("設定に matrix が指定されていません", file=sys.stderr)
            return 1

        approx_id = config.get("approximation", "default")
        approx_data = self._experiment_reader.load_experiment_data(
            "approximation", approx_id,
        )

        matrix_data = self._matrix_reader.load_matrix_data(matrix_id)
        source_matrix = self._data_to_payoff_matrix(matrix_data)

        approximation = MonocycleToGeneralApproximation()
        distance = MaxElementDifferenceDistance()

        approx_result = approximation.approximate(source_matrix)
        reference_matrix = approx_result.approximated_matrix

        comparison_uc = ComparisonUseCase(equilibrium_uc=self._equilibrium_uc)
        result = comparison_uc.compare_with_approximation(
            source_matrix, reference_matrix, approximation, distance,
        )

        ctx = self._output.create_run_context("compare_approximation")
        self._write_comparison_result(ctx, result)

        print(f"行列要素最大距離: {result.max_element_distance}")
        print(f"近似品質: {result.approximation_quality}")
        print("完了")
        return 0

    # ------------------------------------------------------------------
    # compare_product_formula
    # ------------------------------------------------------------------

    def _run_compare_product_formula(
        self, feature_name: str, config: dict[str, Any],
    ) -> int:
        from monocycle_nash.domain.matrix.cauchy_like import CauchyLikePayoffMatrix

        matrix_id = config.get("matrix")
        cauchy_id = config.get("cauchy_like")

        if cauchy_id is None:
            print("設定に cauchy_like が指定されていません", file=sys.stderr)
            return 1

        cauchy_data = self._experiment_reader.load_experiment_data(
            "cauchy_like", cauchy_id,
        )

        a = cauchy_data["a"]
        b = cauchy_data["b"]
        labels = cauchy_data.get("labels")
        params = list(zip(a, b, strict=True))
        cauchy_matrix = CauchyLikePayoffMatrix(params, labels=labels)

        # If a base matrix is provided, compare against it
        if matrix_id is not None:
            matrix_data = self._matrix_reader.load_matrix_data(matrix_id)
            source_matrix = self._data_to_payoff_matrix(matrix_data)

            comparison_uc = ComparisonUseCase(equilibrium_uc=self._equilibrium_uc)
            result = comparison_uc.compare(source_matrix, cauchy_matrix)

            ctx = self._output.create_run_context("compare_product_formula")
            self._write_comparison_result(ctx, result)

            print(f"行列要素最大距離: {result.max_element_distance}")
            print(f"均衡距離: {result.equilibrium_distance}")
        else:
            # Analyze the cauchy matrix alone
            eq_result = self._equilibrium_uc.execute(cauchy_matrix)
            ctx = self._output.create_run_context("compare_product_formula")
            self._output.write_json(ctx, "equilibrium.json", {
                "strategy_ids": eq_result.mixed_strategy.strategy_ids,
                "probabilities": eq_result.mixed_strategy.probabilities.tolist(),
            })
            print(f"均衡解: {dict(zip(eq_result.mixed_strategy.strategy_ids, eq_result.mixed_strategy.probabilities.tolist(), strict=False))}")

            if hasattr(cauchy_matrix, "theoretical_equilibrium"):
                theoretical = cauchy_matrix.theoretical_equilibrium()
                self._output.write_json(ctx, "theoretical_equilibrium.json", {
                    "strategy_ids": theoretical.strategy_ids,
                    "probabilities": theoretical.probabilities.tolist(),
                })
                print(f"理論均衡: {dict(zip(theoretical.strategy_ids, theoretical.probabilities.tolist(), strict=False))}")

        print("完了")
        return 0

    # ------------------------------------------------------------------
    # compare_random_approximation
    # ------------------------------------------------------------------

    def _run_compare_random_approximation(
        self, feature_name: str, config: dict[str, Any],
    ) -> int:
        from monocycle_nash.domain.matrix.approximation import (
            MaxElementDifferenceDistance,
            MonocycleToGeneralApproximation,
        )
        from monocycle_nash.domain.matrix.builder import PayoffMatrixBuilder

        random_cfg_id = config.get("random_matrix", "default")
        random_cfg = self._experiment_reader.load_experiment_data(
            "random_matrix", random_cfg_id,
        )
        size = int(random_cfg.get("size", 4))

        rng = np.random.default_rng(42)
        raw_matrix = rng.standard_normal((size, size))
        raw_matrix = raw_matrix - raw_matrix.T
        labels = [f"s{i}" for i in range(size)]

        source_matrix = PayoffMatrixBuilder.from_general_matrix(raw_matrix, labels=labels)

        approximation = MonocycleToGeneralApproximation()
        distance = MaxElementDifferenceDistance()

        approx_result = approximation.approximate(source_matrix)
        reference_matrix = approx_result.approximated_matrix

        comparison_uc = ComparisonUseCase(equilibrium_uc=self._equilibrium_uc)
        result = comparison_uc.compare_with_approximation(
            source_matrix, reference_matrix, approximation, distance,
        )

        ctx = self._output.create_run_context("compare_random_approximation")
        self._write_comparison_result(ctx, result)

        print(f"行列サイズ: {size}")
        print(f"行列要素最大距離: {result.max_element_distance}")
        print(f"近似品質: {result.approximation_quality}")
        print("完了")
        return 0

    # ------------------------------------------------------------------
    # Random experiment  (experiment_team_strict_spectrum)
    # ------------------------------------------------------------------

    def _run_experiment(
        self, feature_name: str, config: dict[str, Any],
    ) -> int:
        experiment_id = config.get("experiment", "default")
        experiment_data = self._experiment_reader.load_experiment_data(
            "experiment/team_strict_spectrum", experiment_id,
        )

        experiment_config = RandomExperimentConfig(
            character_count=int(experiment_data.get("character_count", 6)),
            team_size=int(experiment_data.get("team_size", 2)),
            generation_count=int(experiment_data.get("generation_count", 100)),
            random_seed=(
                int(experiment_data["random_seed"])
                if experiment_data.get("random_seed") is not None
                else None
            ),
            power_low=float(experiment_data.get("power_low", -1.0)),
            power_high=float(experiment_data.get("power_high", 1.0)),
            vector_low=float(experiment_data.get("vector_low", -1.0)),
            vector_high=float(experiment_data.get("vector_high", 1.0)),
            support_threshold=float(experiment_data.get("support_threshold", 1e-6)),
        )

        uc = RandomExperimentUseCase()

        start = time.time()
        result = uc.run_team_experiment(experiment_config)
        elapsed = time.time() - start

        ctx = self._output.create_run_context("experiment_team_strict_spectrum")
        self._output.write_json(ctx, "trials.json", {"trials": result.trials})
        self._output.write_json(ctx, "summary.json", result.summary)
        self._output.write_metadata(ctx, {
            "use_case": "experiment_team_strict_spectrum",
            "elapsed_seconds": elapsed,
        })

        s = result.summary
        print(f"試行数: {s.get('count')}")
        print(f"サポートサイズ==3 率: {s.get('support_size_eq_3_rate')}")
        print(f"完了 ({elapsed:.2f}s)")
        return 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _data_to_payoff_matrix(self, data: dict[str, Any]) -> Any:
        """data dict から PayoffMatrix を構築する。"""
        if "characters" in data:
            return self._chars_uc.execute(data)
        return self._raw_uc.execute(data)

    def _write_comparison_result(self, ctx: Any, result: Any) -> None:
        """比較結果を出力する。"""
        if result.source_analysis.equilibrium is not None:
            eq = result.source_analysis.equilibrium
            self._output.write_json(ctx, "source_equilibrium.json", {
                "strategy_ids": eq.mixed_strategy.strategy_ids,
                "probabilities": eq.mixed_strategy.probabilities.tolist(),
            })
        if result.reference_analysis.equilibrium is not None:
            eq = result.reference_analysis.equilibrium
            self._output.write_json(ctx, "reference_equilibrium.json", {
                "strategy_ids": eq.mixed_strategy.strategy_ids,
                "probabilities": eq.mixed_strategy.probabilities.tolist(),
            })
        comparison_data: dict[str, Any] = {}
        if result.max_element_distance is not None:
            comparison_data["max_element_distance"] = result.max_element_distance
        if result.equilibrium_distance is not None:
            comparison_data["equilibrium_distance"] = result.equilibrium_distance
        if result.approximation_quality is not None:
            comparison_data["approximation_quality"] = result.approximation_quality
        if comparison_data:
            self._output.write_json(ctx, "comparison.json", comparison_data)

        self._output.write_metadata(ctx, {"use_case": "comparison"})


# ---------------------------------------------------------------------------
# CLI argument parsing & entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="monocycle_nash",
        description="単相性モデルのナッシュ均衡ソルバー",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="run_config ファイルのパス (default: data/run_config/main.toml)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI エントリーポイント。"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path: Path = args.config if args.config is not None else _DEFAULT_CONFIG
    if not config_path.exists():
        print(f"設定ファイルが見つかりません: {config_path}", file=sys.stderr)
        return 1

    run_config = _load_run_config(config_path)
    features: list[str] = run_config.get("features", [])
    if not features:
        print("features が設定されていません", file=sys.stderr)
        return 1

    # Resolve data / result directories from setting
    setting_id = run_config.get("shared", {}).get("setting")
    data_dir = _PROJECT_ROOT / "data"
    result_dir = _PROJECT_ROOT / "result"
    if setting_id is not None:
        loader = TomlTreeLoader()
        setting_path = data_dir / "setting" / f"{setting_id}.toml"
        if setting_path.exists():
            setting = loader.load(setting_path)
            output_cfg = setting.get("output", {})
            base_dir = output_cfg.get("base_dir")
            if base_dir:
                result_dir = _PROJECT_ROOT / base_dir

    runner = UseCaseRunner(data_dir=data_dir, result_dir=result_dir)

    # If only one feature, run it directly; otherwise interactive select
    if len(features) == 1:
        selected = features[0]
    elif sys.stdin.isatty():
        selected = interactive_select(features, _HISTORY_FILE)
    else:
        selected = features[0]

    merged_config = _merge_feature_config(run_config, selected)

    return runner.run_feature(selected, merged_config)
