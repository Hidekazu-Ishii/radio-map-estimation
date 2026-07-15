# ruff: noqa: F722
"""汎用グリッド変換モジュール"""

import numpy as np
from jaxtyping import Float, Int
from numpy import ndarray


def point_to_cell_index(
    points: Float[ndarray, "N 2"],
    cell_size_m: float,
) -> Int[ndarray, "N 2"]:
    """連続座標 (x, y) → 含まれるセルの (row, col) インデックスに変換する (floor ベース、包含判定)

    セルは半開区間 [col * cell_size_m, (col+1) * cell_size_m) x
    [row * cell_size_m, (row+1) * cell_size_m) として定義される

    Returns:
        座標が含まれるセルの (row, col) インデックス配列
    """
    indices = np.floor(points / cell_size_m).astype(np.int64)  # (N, 2)
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
    """物理座標 (x, y) → 含まれるセルの (row, col) インデックスに変換する
     (floor ベース、包含判定) → margin オフセット付き bldg_mask インデックス

    RSS値の対応 (point_to_cell_index) と同じ floor ベースのセル包含判定を用いる
    セルは半開区間 [x, x + cell_size_m) を代表座標 (左下端) に紐づける
    最近傍グリッド点への丸めは行わない (round ベースのスナップは最大 ±0.5 セル分の
    誤差を生むため廃止)

    範囲外は端にクリップする
    """
    offset = margin_num_cells(margin_m, cell_size_m)
    idx = point_to_cell_index(points, cell_size_m) + offset  # (N,2)

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
