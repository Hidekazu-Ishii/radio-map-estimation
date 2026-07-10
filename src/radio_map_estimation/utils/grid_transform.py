# ruff: noqa: F722
"""汎用グリッド変換モジュール"""

import numpy as np
from jaxtyping import Float, Int
from numpy import ndarray


def snap_to_nearest_grid_point(
    points: Float[ndarray, "N 2"],
    cell_size_m: float,
) -> Float[ndarray, "N 2"]:
    """座標を最も近いグリッド点座標 (セル左下端, cell_size_m x 整数) に変換する

    グリッド点は x, y ともに 0 以上を前提とする

    Returns:
        グリッド点にスナップされた (x, y) 座標配列
    """
    return np.round(points / cell_size_m) * cell_size_m


def grid_point_to_index(
    grid_points: Float[ndarray, "N 2"],
    cell_size_m: float,
) -> Int[ndarray, "N 2"]:
    """snap_to_nearest_grid_point で得たグリッド点座標を (row, col) インデックスに変換する

    Parameters
    ----------
    grid_points : snap_to_nearest_grid_point の出力 (round 済み座標)
    cell_size_m : セルサイズ [m]

    Returns
    -------
    (row, col) インデックス配列shape (N, 2)
    """
    indices = np.round(grid_points / cell_size_m).astype(np.int64)  # (N, 2)
    row = indices[:, 1]
    col = indices[:, 0]
    return np.stack([row, col], axis=-1)


def grid_index_to_point(
    rows: Int[ndarray, "N 1"],
    cols: Int[ndarray, "N 1"],
    cell_size_m: float,
) -> Float[ndarray, "N 2"]:
    """セルインデックス (row, col) → グリッド点座標 (左下端) [m] に変換する"""
    x = cols.astype(np.float64) * cell_size_m
    y = rows.astype(np.float64) * cell_size_m
    return np.stack([x, y], axis=-1)


def margin_num_cells(margin_m: float, cell_size_m: float) -> int:
    """margin_m に対応するセル数 (round)"""
    return round(margin_m / cell_size_m)


def coord_to_bldg_index(
    points: Float[ndarray, "N 2"],
    cell_size_m: float,
    margin_m: float,
    grid_shape: tuple[int, int],
) -> Int[ndarray, "N 2"]:
    """物理座標 (x, y) → 最近傍グリッド点座標にスナップ → margin オフセット付き bldg_mask インデックス (row, col)

    範囲外は端にクリップする
    """
    offset = margin_num_cells(margin_m, cell_size_m)
    grid_points = snap_to_nearest_grid_point(points, cell_size_m)  # (N,2)
    idx = grid_point_to_index(grid_points, cell_size_m) + offset  # (N,2)

    h, w = grid_shape
    row = np.clip(idx[:, 0], 0, h - 1)
    col = np.clip(idx[:, 1], 0, w - 1)
    return np.stack([row, col], axis=-1)


def bldg_index_to_coord(
    rows: Int[ndarray, "N 1"],
    cols: Int[ndarray, "N 1"],
    cell_size_m: float,
    margin_m: float,
) -> Float[ndarray, "N 2"]:
    """margin オフセット付き bldg_mask インデックス (row, col) → グリッド点座標 (左下端) [m]

    インデックス k は x = (k - num_margin_cells) * cell_size_m に対応する
     (build_bldg_mask の設計に準拠)
    """
    offset = margin_num_cells(margin_m, cell_size_m)
    return grid_index_to_point(rows - offset, cols - offset, cell_size_m)
