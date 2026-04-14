from __future__ import annotations

import json
from pathlib import Path

from monocycle_nash.analysis.app.compare_product_formula import run
from monocycle_nash.runtime.infra.loader.main_config import MainConfigLoader


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _write_setting(data_dir: Path, tmp_path: Path) -> None:
    _write(
        data_dir / "setting" / "local.toml",
        f'''
        [runmeta]
        sqlite_path = "{(tmp_path / '.runmeta' / 'run_history.db').as_posix()}"

        [output]
        base_dir = "{(tmp_path / 'result').as_posix()}"
        ''',
    )


def _write_main_config(data_dir: Path, cauchy_like_name: str = "default") -> None:
    _write(
        data_dir / "run_config" / "main.toml",
        f'''
        features = ["compare_product_formula"]

        [shared]
        setting = "local"

        [compare_product_formula]
        cauchy_like = "{cauchy_like_name}"
        ''',
    )


def _write_cauchy_like_3d(data_dir: Path, name: str = "default") -> None:
    _write(
        data_dir / "cauchy_like" / name / "data.toml",
        '''
        a = [1.0, -1.0, 1.0]
        b = [1.0, 2.0, 3.0]
        ''',
    )


def _write_cauchy_like_5d(data_dir: Path, name: str = "default") -> None:
    _write(
        data_dir / "cauchy_like" / name / "data.toml",
        '''
        a = [1.0, -1.0, 1.0, -1.0, 1.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        ''',
    )


# ---------------------------------------------------------------------------
# 正常系
# ---------------------------------------------------------------------------


def test_compare_product_formula_3d_writes_output(tmp_path: Path) -> None:
    """3次元: 出力 JSON が生成され期待フィールドが揃っていることを確認。"""
    data_dir = tmp_path / "data"
    _write_main_config(data_dir)
    _write_cauchy_like_3d(data_dir)
    _write_setting(data_dir, tmp_path)

    code = run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    assert code == 0

    run_dirs = [x for x in (tmp_path / "result").iterdir() if x.is_dir()]
    assert len(run_dirs) == 1

    result_path = run_dirs[0] / "output" / "compare_product_formula.json"
    assert result_path.exists()

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["n"] == 3
    assert len(payload["exact_equilibrium"]["probabilities"]) == 3
    assert len(payload["product_formula_raw"]) == 3
    assert "product_formula_normalized" in payload


def test_compare_product_formula_3d_exact_sums_to_one(tmp_path: Path) -> None:
    """3次元: 厳密解の確率の合計が 1 に近い。"""
    data_dir = tmp_path / "data"
    _write_main_config(data_dir)
    _write_cauchy_like_3d(data_dir)
    _write_setting(data_dir, tmp_path)

    run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    run_dirs = [x for x in (tmp_path / "result").iterdir() if x.is_dir()]
    assert len(run_dirs) == 1
    result_path = run_dirs[0] / "output" / "compare_product_formula.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    probs = payload["exact_equilibrium"]["probabilities"]
    assert abs(sum(probs) - 1.0) < 1e-6


def test_compare_product_formula_3d_normalized_valid(tmp_path: Path) -> None:
    """3次元: product_formula_normalized が有効な確率分布として認識される。"""
    data_dir = tmp_path / "data"
    _write_main_config(data_dir)
    _write_cauchy_like_3d(data_dir)
    _write_setting(data_dir, tmp_path)

    run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    run_dirs = [x for x in (tmp_path / "result").iterdir() if x.is_dir()]
    assert len(run_dirs) == 1
    result_path = run_dirs[0] / "output" / "compare_product_formula.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    norm = payload["product_formula_normalized"]
    assert norm["is_valid"] is True
    assert norm["probabilities"] is not None
    assert abs(sum(norm["probabilities"]) - 1.0) < 1e-6


def test_compare_product_formula_5d_writes_output(tmp_path: Path) -> None:
    """5次元: 出力 JSON が生成され期待フィールドが揃っていることを確認。"""
    data_dir = tmp_path / "data"
    _write_main_config(data_dir)
    _write_cauchy_like_5d(data_dir)
    _write_setting(data_dir, tmp_path)

    code = run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    assert code == 0

    result_path = next((tmp_path / "result").iterdir()) / "output" / "compare_product_formula.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["n"] == 5
    assert len(payload["exact_equilibrium"]["probabilities"]) == 5


def test_compare_product_formula_5d_residual_small(tmp_path: Path) -> None:
    """5次元: 厳密解の均衡条件残差 max|Bx| が十分に小さい。"""
    data_dir = tmp_path / "data"
    _write_main_config(data_dir)
    _write_cauchy_like_5d(data_dir)
    _write_setting(data_dir, tmp_path)

    run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    run_dirs = [x for x in (tmp_path / "result").iterdir() if x.is_dir()]
    assert len(run_dirs) == 1
    result_path = run_dirs[0] / "output" / "compare_product_formula.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["exact_equilibrium"]["max_residual"] < 1e-6


def test_compare_product_formula_5d_normalized_not_exact(tmp_path: Path) -> None:
    """5次元: product_formula の a 除算・正規化値が厳密解と一致しない。

    n≥5 では product_formula は均衡解を与えないため、
    max_diff_from_exact がゼロより大きいことを確認する。
    """
    data_dir = tmp_path / "data"
    _write_main_config(data_dir)
    _write_cauchy_like_5d(data_dir)
    _write_setting(data_dir, tmp_path)

    run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    run_dirs = [x for x in (tmp_path / "result").iterdir() if x.is_dir()]
    assert len(run_dirs) == 1
    result_path = run_dirs[0] / "output" / "compare_product_formula.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    norm = payload["product_formula_normalized"]
    assert norm["is_valid"] is True
    assert norm["max_diff_from_exact"] > 1e-4


# ---------------------------------------------------------------------------
# 異常系
# ---------------------------------------------------------------------------


def test_compare_product_formula_returns_failure_on_missing_config(tmp_path: Path) -> None:
    """cauchy_like セクション未設定でエラーコード 1 が返る。"""
    data_dir = tmp_path / "data"
    _write(
        data_dir / "run_config" / "main.toml",
        '''
        features = ["compare_product_formula"]

        [shared]
        setting = "local"

        [compare_product_formula]
        ''',
    )
    _write_setting(data_dir, tmp_path)

    code = run(MainConfigLoader(data_dir / "run_config" / "main.toml"))

    assert code == 1
