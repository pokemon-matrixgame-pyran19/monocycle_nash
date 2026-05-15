#!/usr/bin/env python3
"""方向分散を意識したキャラクターCSV生成の簡易スクリプト。

目的:
- 差分ベクトル V_{ij} = v_i - v_j (i<j) を
  V_{ij} = |V_{ij}|(x_{ij}, y_{ij}) と見たときの方向 (x_{ij}, y_{ij}) が
  なるべく均等になるように、元の点 v_1..v_n を探索する。
- 交互最適化 (角度→半径) とシミュレーテッドアニーリングを併用する。

出力:
- monocycle_nash の character CSV 形式
  label,power,vector_x,vector_y
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.random import Generator


@dataclass(frozen=True)
class Config:
    n_points: int
    output: Path
    seed: int
    power: float
    label_prefix: str
    outer_iters: int
    inner_iters: int
    init_radius_min: float
    init_radius_max: float
    radius_min: float
    radius_max: float
    theta_step: float
    radius_step: float
    temp_start: float
    temp_end: float
    penalty_alpha: float
    zero_distance_penalty: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "差分ベクトルの向きが偏りにくい2D点群を探索し、"
            "character入力CSVを生成する。"
        )
    )
    parser.add_argument("--n-points", type=int, default=12, help="生成する点数")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated_characters.csv"),
        help="出力CSVパス",
    )
    parser.add_argument("--seed", type=int, default=19, help="乱数シード")
    parser.add_argument("--power", type=float, default=0.0, help="power列の値")
    parser.add_argument(
        "--label-prefix",
        type=str,
        default="c",
        help="label列の接頭辞 (例: c -> c1,c2,...)",
    )

    parser.add_argument("--outer-iters", type=int, default=250, help="外側反復回数")
    parser.add_argument("--inner-iters", type=int, default=4, help="各変数の内側反復回数")

    parser.add_argument("--init-radius-min", type=float, default=0.5)
    parser.add_argument("--init-radius-max", type=float, default=2.0)
    parser.add_argument("--radius-min", type=float, default=0.2)
    parser.add_argument("--radius-max", type=float, default=3.0)

    parser.add_argument("--theta-step", type=float, default=0.30, help="角度更新幅")
    parser.add_argument("--radius-step", type=float, default=0.15, help="半径更新幅")

    parser.add_argument("--temp-start", type=float, default=0.30, help="初期温度")
    parser.add_argument("--temp-end", type=float, default=0.01, help="終端温度")

    parser.add_argument(
        "--penalty-alpha",
        type=float,
        default=3.0,
        help="f(dot)=exp(alpha*dot) の alpha",
    )
    parser.add_argument(
        "--zero-distance-penalty",
        type=float,
        default=50.0,
        help="点同士が近すぎる場合の追加ペナルティ",
    )

    args = parser.parse_args()

    if args.n_points < 2:
        raise ValueError("--n-points は 2 以上にしてください")
    if args.radius_min <= 0 or args.radius_max <= 0 or args.radius_min >= args.radius_max:
        raise ValueError("半径範囲が不正です")
    if args.init_radius_min <= 0 or args.init_radius_min >= args.init_radius_max:
        raise ValueError("初期半径範囲が不正です")
    if args.temp_start <= 0 or args.temp_end <= 0 or args.temp_end > args.temp_start:
        raise ValueError("温度設定が不正です")
    if args.outer_iters < 1 or args.inner_iters < 1:
        raise ValueError("反復回数は1以上にしてください")

    return Config(
        n_points=args.n_points,
        output=args.output,
        seed=args.seed,
        power=args.power,
        label_prefix=args.label_prefix,
        outer_iters=args.outer_iters,
        inner_iters=args.inner_iters,
        init_radius_min=args.init_radius_min,
        init_radius_max=args.init_radius_max,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        theta_step=args.theta_step,
        radius_step=args.radius_step,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        penalty_alpha=args.penalty_alpha,
        zero_distance_penalty=args.zero_distance_penalty,
    )


def temperature_at(step: int, total: int, start: float, end: float) -> float:
    if total <= 1:
        return end
    ratio = step / float(total - 1)
    # 幾何冷却
    return start * ((end / start) ** ratio)


def points_from(theta: np.ndarray, radius: np.ndarray) -> np.ndarray:
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def direction_vectors(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = points.shape[0]
    diffs: list[np.ndarray] = []
    norms: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            v = points[i] - points[j]
            norm = np.linalg.norm(v)
            diffs.append(v)
            norms.append(norm)
    diff_arr = np.asarray(diffs, dtype=float)
    norm_arr = np.asarray(norms, dtype=float)
    return diff_arr, norm_arr


def energy(points: np.ndarray, alpha: float, zero_distance_penalty: float) -> float:
    diffs, norms = direction_vectors(points)
    eps = 1e-9

    # 方向単位ベクトル U_j
    valid = norms > eps
    if not np.any(valid):
        return float("inf")

    unit = diffs[valid] / norms[valid, None]

    # E = sum_{j,k} f(U_j・U_k), ただし自己項は除外
    gram = unit @ unit.T
    m = gram.shape[0]
    mask = ~np.eye(m, dtype=bool)
    dots = gram[mask]
    pair_penalty = np.exp(alpha * dots).mean()

    # 点が重なる/近すぎると方向定義が崩れるため追加ペナルティ
    near = norms[~valid]
    collision_penalty = zero_distance_penalty * float(near.size)

    return float(pair_penalty + collision_penalty)


def metropolis_accept(delta: float, temp: float, rng: Generator) -> bool:
    if delta <= 0:
        return True
    return rng.random() < math.exp(-delta / max(temp, 1e-12))


def optimize(cfg: Config) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(cfg.seed)

    theta = rng.uniform(0.0, 2.0 * math.pi, size=cfg.n_points)
    radius = rng.uniform(cfg.init_radius_min, cfg.init_radius_max, size=cfg.n_points)

    points = points_from(theta, radius)
    current_e = energy(points, cfg.penalty_alpha, cfg.zero_distance_penalty)

    best_theta = theta.copy()
    best_radius = radius.copy()
    best_e = current_e

    total_steps = cfg.outer_iters

    for outer in range(cfg.outer_iters):
        temp = temperature_at(outer, total_steps, cfg.temp_start, cfg.temp_end)

        # 交互最適化1: 角度 theta を更新
        for i in range(cfg.n_points):
            for _ in range(cfg.inner_iters):
                cand_theta = theta.copy()
                cand_theta[i] = (cand_theta[i] + rng.normal(0.0, cfg.theta_step * temp)) % (
                    2.0 * math.pi
                )
                cand_points = points_from(cand_theta, radius)
                cand_e = energy(cand_points, cfg.penalty_alpha, cfg.zero_distance_penalty)
                if metropolis_accept(cand_e - current_e, temp, rng):
                    theta = cand_theta
                    current_e = cand_e

        # 交互最適化2: 半径 radius を更新
        for i in range(cfg.n_points):
            for _ in range(cfg.inner_iters):
                cand_radius = radius.copy()
                cand_radius[i] = np.clip(
                    cand_radius[i] + rng.normal(0.0, cfg.radius_step * temp),
                    cfg.radius_min,
                    cfg.radius_max,
                )
                cand_points = points_from(theta, cand_radius)
                cand_e = energy(cand_points, cfg.penalty_alpha, cfg.zero_distance_penalty)
                if metropolis_accept(cand_e - current_e, temp, rng):
                    radius = cand_radius
                    current_e = cand_e

        if current_e < best_e:
            best_e = current_e
            best_theta = theta.copy()
            best_radius = radius.copy()

    best_points = points_from(best_theta, best_radius)
    return best_points, best_e


def write_character_csv(points: np.ndarray, output: Path, power: float, label_prefix: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "power", "vector_x", "vector_y"])
        for i, (x, y) in enumerate(points, start=1):
            writer.writerow([f"{label_prefix}{i}", power, float(x), float(y)])


def summarize(points: np.ndarray, best_e: float) -> str:
    diffs, norms = direction_vectors(points)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    # -pi..pi を 0..2pi に正規化
    angles = np.mod(angles, 2.0 * math.pi)
    bins = 12
    hist, _ = np.histogram(angles, bins=bins, range=(0.0, 2.0 * math.pi))
    nonzero_norm = norms[norms > 1e-9]
    min_dist = float(nonzero_norm.min()) if nonzero_norm.size > 0 else 0.0
    max_dist = float(nonzero_norm.max()) if nonzero_norm.size > 0 else 0.0
    return (
        f"best_energy={best_e:.6f}\n"
        f"diff_vectors={len(diffs)}\n"
        f"pair_distance_min={min_dist:.6f}, pair_distance_max={max_dist:.6f}\n"
        f"angle_hist(12bins)={hist.tolist()}"
    )


def main() -> None:
    cfg = parse_args()
    points, best_e = optimize(cfg)
    write_character_csv(points, cfg.output, cfg.power, cfg.label_prefix)
    print(summarize(points, best_e))
    print(f"written: {cfg.output}")


if __name__ == "__main__":
    main()
