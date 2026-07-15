# ruff: noqa: F722
"""
エッジ重み行列 → グラフ距離 → 最短経路距離

設計方針
--------
- 重み→距離の変換は EdgeWeightFunction の実装から独立させる
  (どのエッジ重み関数を使っても同じ変換ロジックを共有できるようにするため)
- 変換方式は GraphDistanceConfig.method で切替可能にする
    neg_log    : d_ij = -log(rho_ij) , rho_ij = w_ij / sqrt(w_ii * w_jj) (正規化相関)
                 stationary カーネルでは rho_ij = w_ij / sigma_2 に一致する
                 Gudmundson の場合 d_ij = D_ij * ln2 / d_cor となり,
                 既存のユークリッド距離ベースの分析 (analyze_variogram.py) との
                 検算に使える
    reciprocal : d_ij = 1 / w_ij (正規化不要, 非定常カーネルでも単純に使える)
- 対角成分 (自己重み) は距離 0 に落ちるように正規化する
- 最短経路は密な完全グラフ上の Dijkstra (scipy.sparse.csgraph.shortest_path)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Float
from numpy import ndarray
from scipy.sparse.csgraph import shortest_path

_GraphDistanceMethod = Literal["neg_log", "reciprocal"]


@dataclass(frozen=True, slots=True)
class GraphDistanceConfig:
    """重み→距離変換の設定値

    Attributes
    ----------
    method : "neg_log" | "reciprocal"
    eps    : 数値安定化用の下限値 (0 除算 / log(0) を防ぐ)
    """

    method: _GraphDistanceMethod
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.method not in ("neg_log", "reciprocal"):
            raise ValueError(f"Unknown method: {self.method!r}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")


def weights_to_graph_distance(
    w: Float[ndarray, "N N"],
    cfg: GraphDistanceConfig,
) -> Float[ndarray, "N N"]:
    """重み行列 W をグラフ距離行列 D_edge に変換する (対角は 0)

    Parameters
    ----------
    W   : エッジ重み行列, shape (N, N). 対角成分 (自己重み) を正規化の基準に使う
    cfg : 変換方式の設定

    Returns
    -------
    D_edge : グラフ距離行列, shape (N, N). 非負, 対角は 0
    """
    match cfg.method:
        case "neg_log":
            diag = np.diag(w)  # (N,) 自己重み
            norm = np.sqrt(np.outer(diag, diag))  # (N, N) sqrt(w_ii * w_jj)
            rho = np.clip(w / norm, cfg.eps, 1.0)  # 正規化相関, (0, 1] にクリップ
            distance = -np.log(rho)
        case "reciprocal":
            distance = 1.0 / np.clip(w, cfg.eps, None)
        case _:
            raise NotImplementedError(f"method={cfg.method!r} は未実装")

    np.fill_diagonal(distance, 0.0)  # 自己距離は必ず 0
    return distance


def compute_graph_shortest_path(
    distance: Float[ndarray, "N N"],
) -> Float[ndarray, "N N"]:
    """密な完全グラフ上でグラフ最短経路距離を計算する (Dijkstra)

    Parameters
    ----------
    distance : weights_to_graph_distance() の出力. 非負, 対角は 0

    Returns
    -------
    graph_distance : 全ノード対のグラフ最短経路距離, shape (N, N), 対称
    """
    if np.any(distance < 0.0):
        raise ValueError("distance must be non-negative for Dijkstra shortest path")
    graph_distance = shortest_path(distance, method="D", directed=False)
    return graph_distance
