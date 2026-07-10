# ruff: noqa: F722
"""
train / test 点をサンプリングするモジュール

サンプリング戦略:
    train: (0, 0) - (area_size_m, area_size_m) の範囲で連続座標を一様サンプリングする
           (不規則な座標配列)
    test : 値の入っているセルインデックスを直接サンプリング (または全件取得) し、
           左下端座標に変換する (セル格子に整列した座標配列)
           train で使用したセルは除外し、disjoint を保証する
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Int

from ..utils.grid_transform import grid_point_to_index, snap_to_nearest_grid_point


def cell_lower_left(
    rows: Int[np.ndarray, "N 1"],
    cols: Int[np.ndarray, "N 1"],
    cell_size_m: float,
) -> Float[np.ndarray, "N 2"]:
    """セルインデックス (row, col) → 左下端座標 (x, y) [m] に変換する

    Parameters
    ----------
    rows, cols  : shape (N,) セルの行・列インデックス
    cell_size_m : セルサイズ [m]

    Returns
    -------
    coords : shape (N, 2) 左下端座標 (x, y) [m]
    """
    x = cols.astype(np.float64) * cell_size_m  # (N,)
    y = rows.astype(np.float64) * cell_size_m  # (N,)
    return np.stack([x, y], axis=-1)  # (N, 2)


def _sample_train_points(
    rss_dbm_gt: Float[np.ndarray, "H W"],
    area_size_m: float,
    cell_size_m: float,
    train_size: int,
    rng: np.random.Generator,
) -> tuple[Float[np.ndarray, "N 2"], Float[np.ndarray, "N 1"], set[int]]:
    """連続座標を一様サンプリングし、有効な (座標, 値) を train_size 個集める (不規則座標)

    座標は連続値のまま保存する (train は格子に整列しない不規則座標)。
    セル所属判定のみ snap_to_nearest_grid_point + grid_point_to_index で行う。
    """
    height, width = rss_dbm_gt.shape
    collected_coords: list[np.ndarray] = []
    collected_values: list[float] = []
    used_flat_indices: set[int] = set()

    while len(collected_coords) < train_size:
        num_needed = train_size - len(collected_coords)
        batch_size = num_needed * 4  # nan / 重複による棄却を見込んで多めにサンプリング

        xy = rng.uniform(0.0, area_size_m, size=(batch_size, 2))  # (batch_size, 2) 連続座標

        grid_points = snap_to_nearest_grid_point(xy, cell_size_m)  # (batch_size, 2)
        cell_indices = grid_point_to_index(grid_points, cell_size_m)  # (batch_size, 2)
        rows = np.minimum(cell_indices[:, 0], height - 1)
        cols = np.minimum(cell_indices[:, 1], width - 1)
        flat_indices = rows * width + cols  # (batch_size,)

        values = rss_dbm_gt[rows, cols]  # (batch_size,)
        is_valid = ~np.isnan(values)

        for x, y, value, flat_idx, valid in zip(
            xy[:, 0], xy[:, 1], values, flat_indices, is_valid, strict=False
        ):
            if len(collected_coords) >= train_size:
                break
            if not valid or flat_idx in used_flat_indices:
                continue

            collected_coords.append(np.array([x, y]))  # 連続座標のまま保存
            collected_values.append(value)
            used_flat_indices.add(int(flat_idx))

    coords = np.stack(collected_coords, axis=0)  # (train_size, 2)
    rss_dbm = np.asarray(collected_values, dtype=np.float64).reshape(-1, 1)  # (train_size, 1)

    return coords, rss_dbm, used_flat_indices


def _sample_test_cells(
    rss_dbm_gt: Float[np.ndarray, "H W"],
    cell_size_m: float,
    test_size: int | None,
    rng: np.random.Generator,
    excluded_flat_indices: set[int],
) -> tuple[Float[np.ndarray, "M 2"], Float[np.ndarray, "M 1"]]:
    """train で使用したセルを除く有効セルから test 点を取得する (セル格子に整列した座標)

    test_size が None の場合は該当する全セルを、int の場合はその個数だけランダムに
    セルインデックスを直接サンプリングする。

    Parameters
    ----------
    rss_dbm_gt             : (H, W) 真値マップ (欠測は nan)
    cell_size_m            : セルサイズ [m]
    test_size              : 取得するセル数。None なら全件
    rng                    : 乱数生成器 (外部から受け取る)
    excluded_flat_indices  : train で採用済みのフラットセルインデックス

    Returns
    -------
    coords  : shape (M, 2) 左下端座標 (x, y) [m]
    rss_dbm : shape (M, 1) 対応する値
    """
    height, width = rss_dbm_gt.shape
    valid_flat_indices = np.flatnonzero(~np.isnan(rss_dbm_gt))  # (num_valid,)

    excluded_mask = np.isin(valid_flat_indices, np.fromiter(excluded_flat_indices, dtype=np.int64))
    remaining_flat_indices = valid_flat_indices[~excluded_mask]  # (num_remaining,)

    if test_size is not None:
        if test_size > len(remaining_flat_indices):
            raise ValueError(
                f"test_size ({test_size}) exceeds remaining observable cells ({len(remaining_flat_indices)})"
            )
        remaining_flat_indices = rng.choice(remaining_flat_indices, size=test_size, replace=False)
        remaining_flat_indices = np.sort(remaining_flat_indices)

    rows, cols = np.unravel_index(remaining_flat_indices, (height, width))
    coords = cell_lower_left(rows, cols, cell_size_m)  # (M, 2)
    rss_dbm = rss_dbm_gt[rows, cols].reshape(-1, 1)  # (M, 1)

    return coords, rss_dbm


def sample_train_test_points(
    rss_dbm_gt: Float[np.ndarray, "H W"],
    area_size_m: float,
    cell_size_m: float,
    train_size: int,
    test_size: int | None,
    rng: np.random.Generator,
) -> tuple[
    Float[np.ndarray, "N 2"],  # train_coords (x, y) 不規則な連続座標
    Float[np.ndarray, "N 1"],  # train_rss_dbm
    Float[np.ndarray, "M 2"],  # test_coords (x, y) 左下端座標 (セル格子に整列)
    Float[np.ndarray, "M 1"],  # test_rss_dbm
]:
    """(0, 0)-(area_size_m, area_size_m) から train / test 点をサンプリングする

    Parameters
    ----------
    rss_dbm_gt  : (H, W) 真値マップ (欠測は nan)
    area_size_m : サンプリング範囲の一辺 [m]
    cell_size_m : セルサイズ [m]
    train_size  : 学習点数
    test_size   : 予測 (評価) 点数。None の場合は、train で使用したセルを除く
                  全ての有効セルを test とする
    rng         : 乱数生成器 (外部から受け取る)

    Returns
    -------
    train_coords, train_rss_dbm : 学習用の不規則な連続座標と値
    test_coords, test_rss_dbm   : 評価用の左下端座標(セル格子に整列)と値
                                   (train と使用セルが disjoint)
    """
    train_coords, train_rss_dbm, train_flat_indices = _sample_train_points(
        rss_dbm_gt, area_size_m, cell_size_m, train_size, rng
    )

    test_coords, test_rss_dbm = _sample_test_cells(
        rss_dbm_gt, cell_size_m, test_size, rng, excluded_flat_indices=train_flat_indices
    )

    return train_coords, train_rss_dbm, test_coords, test_rss_dbm
