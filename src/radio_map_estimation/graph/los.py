# ruff: noqa: F722
"""
LOS/NLOS幾何接続判定 (グラフのエッジトポロジー決定)

設計方針
--------
- 「どのノードペアをエッジとして扱うか (トポロジー)」を
  「エッジの重みをいくつにするか (EdgeWeightFunction によるカーネル値)」から
  完全に独立させる. distance.py が重み→距離変換をEdgeWeightFunctionの実装から
  独立させた設計思想の延長線上にある
- 接続条件は2段階:
    1. 半径条件 : ユークリッド距離が max_radius_m 以内
                  (これを超えるペアはLOS判定すら行わずNLOS(非接続)扱いにする.
                   候補を絞ってからBresenhamを回すことで計算量を抑える)
    2. LOS条件  : Bresenhamラインで bldg_mask 上を1ピクセルずつ辿り、
                  建物ピクセルを1つも通過しなければLOS
- Bresenhamライン判定は utils/bresenham.py の汎用実装を使う
  (元々 pathloss/base.py の RX-TX 間遮蔽判定専用だったものを汎用化したもの)
- 孤立ノード (次数0) の「検出」のみをここで行う. 除去そのもの (座標・残差配列の
  フィルタリング) は呼び出し元 (graph/spectral.py の GraphNodeSet.filter) の責務とし,
  ここではマスクを返すだけに留める (単一責任)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float
from numpy import ndarray

from ..loader.dataset import GridInfo
from ..utils.bresenham import bresenham_line


@dataclass(frozen=True, slots=True)
class LosAdjacencyConfig:
    """LOS幾何接続判定の設定値

    Attributes
    ----------
    max_radius_m : 候補ペアの上限半径 [m].
                   これを超えるペアはLOS判定を行わず非接続 (False) とする
    """

    max_radius_m: float

    def __post_init__(self) -> None:
        if self.max_radius_m <= 0.0:
            raise ValueError(f"max_radius_m must be positive, got {self.max_radius_m}")


def _pairwise_distance(coords: Float[ndarray, "N 2"]) -> Float[ndarray, "N N"]:
    """全ノード対のユークリッド距離行列を計算する"""
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
    return np.linalg.norm(diff, axis=-1)


def _is_los(coords: Float[ndarray, "N 2"], grid_info: GridInfo, i: int, j: int) -> bool:
    """ノード i, j 間が LOS (建物遮蔽なし) かどうかを判定する

    compute_ray_crossing_count (pathloss/base.py) と同じ Bresenham ベースの
    走査を、「建物ピクセルを1つでも通過したらNLOS」という二値判定に単純化して使う
    """
    idx = grid_info.coord_to_bldg_index(coords[[i, j]])  # (2,2) row,col
    rows, cols = bresenham_line(int(idx[0, 0]), int(idx[0, 1]), int(idx[1, 0]), int(idx[1, 1]))
    hits = grid_info.bldg_mask[rows, cols]  # (line_len,) bool
    return not bool(np.any(hits))


def compute_los_adjacency(
    coords: Float[ndarray, "N 2"],
    grid_info: GridInfo,
    cfg: LosAdjacencyConfig,
) -> Bool[ndarray, "N N"]:
    """半径内 かつ LOS (建物遮蔽なし) のペアのみ True とする隣接行列を返す

    対角成分は常に False (自己ループなし). 対称行列.

    Parameters
    ----------
    coords    : 全ノード (train + held-out) の座標 (x, y) [m], shape (N, 2)
    grid_info : グリッド全体の静的情報 (bldg_mask を含む)
    cfg       : LOS幾何接続判定の設定

    Returns
    -------
    adjacency : 隣接行列, shape (N, N). 対称, 対角は False
    """
    n = coords.shape[0]
    distance = _pairwise_distance(coords)
    upper_triangle = np.triu(np.ones((n, n), dtype=bool), k=1)  # 対角除外・上三角のみ
    radius_candidates = (distance <= cfg.max_radius_m) & upper_triangle

    adjacency = np.zeros((n, n), dtype=bool)
    for i, j in np.argwhere(radius_candidates):
        if _is_los(coords, grid_info, int(i), int(j)):
            adjacency[i, j] = True

    adjacency |= adjacency.T  # 対称化 (下三角にも反映)
    return adjacency


def filter_isolated_nodes(adjacency: Bool[ndarray, "N N"]) -> Bool[ndarray, "N 1"]:
    """次数0 (孤立) のノードを検出する

    孤立ノードはエッジを一切持たないため他ノードの次数に寄与しない.
    したがってこのマスクによる1回のフィルタで十分であり、反復除去は不要
     (孤立ノードを取り除いても残りのノードの次数は変化しないため)

    Parameters
    ----------
    adjacency : compute_los_adjacency の出力, shape (N, N)

    Returns
    -------
    survive_mask : 生存ノードなら True, 孤立ノードなら False, shape (N,)
    """
    degree = adjacency.sum(axis=1)
    return degree > 0
