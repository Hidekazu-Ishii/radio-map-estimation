# ruff: noqa: F722
"""
train / test 点をサンプリングするモジュール

サンプリング戦略:
    train: (0, 0) - (area_size_m, area_size_m) の範囲で連続座標を一様サンプリングする
           (不規則な座標配列)
    test : 値の入っているセルインデックスを直接サンプリング (または全件取得) し、
           左下端座標に変換する (セル格子に整列した座標配列)
           train で使用したセルは除外し、disjoint を保証する

pool 制約について:
    test_prod (本番評価用、固定) を一度確定した後、チューニング用の train_tune /
    test_tune のサンプリング、および本番学習用の train_prod のサンプリングは
    pool_flat_indices の範囲内のみに制限しなければならない。
    候補セルを絞る役割は `allowed_flat_indices` / `candidate_flat_indices` 引数が担う。
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


def create_pool_test_split(
    rss_dbm_gt: Float[np.ndarray, "H W"],
    test_size: int,
    rng: np.random.Generator,
) -> tuple[Int[np.ndarray, "T 1"], Int[np.ndarray, "P 1"]]:
    """全有効セルから test_prod 用セルを一度だけ確定し、残りを pool として返す

    ここで確定した test_flat_indices は、以後チューニング処理には一切渡さないこと。

    Parameters
    ----------
    rss_dbm_gt : (H, W) 真値マップ (欠測は nan)
    test_size  : test_prod のセル数
    rng        : 乱数生成器 (外部から受け取る)

    Returns
    -------
    test_flat_indices : test_prod に属するフラットインデックス (昇順)
    pool_flat_indices  : それ以外の有効セルのフラットインデックス (昇順)

    Raises
    ------
    ValueError
        test_size が有効セル数を超える場合
    """
    valid_flat_indices = np.flatnonzero(~np.isnan(rss_dbm_gt))  # (num_valid,)
    if test_size > len(valid_flat_indices):
        raise ValueError(f"test_size ({test_size}) exceeds observable cells ({len(valid_flat_indices)})")

    shuffled = rng.permutation(valid_flat_indices)
    test_flat_indices = np.sort(shuffled[:test_size])
    pool_flat_indices = np.sort(shuffled[test_size:])
    return test_flat_indices, pool_flat_indices


def _sample_train_points(
    rss_dbm_gt: Float[np.ndarray, "H W"],
    area_size_m: float,
    cell_size_m: float,
    train_size: int,
    rng: np.random.Generator,
    allowed_flat_indices: Int[np.ndarray, "K 1"] | None = None,
) -> tuple[Float[np.ndarray, "N 2"], Float[np.ndarray, "N 1"], set[int]]:
    """連続座標を一様サンプリングし、有効な (座標, 値) を train_size 個集める (不規則座標)

    座標は連続値のまま保存する (train は格子に整列しない不規則座標)。
    セル所属判定のみ snap_to_nearest_grid_point + grid_point_to_index で行う。

    allowed_flat_indices が与えられた場合、その集合に含まれるセルのみ採用する
    (pool 制約。test_prod のセルを絶対に踏まないようにするための唯一の窓口)。
    """
    height, width = rss_dbm_gt.shape
    allowed_set = None if allowed_flat_indices is None else {int(i) for i in allowed_flat_indices}

    collected_coords: list[np.ndarray] = []
    collected_values: list[float] = []
    used_flat_indices: set[int] = set()

    while len(collected_coords) < train_size:
        num_needed = train_size - len(collected_coords)
        batch_size = num_needed * 4  # nan / 重複 / pool外による棄却を見込んで多めにサンプリング

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
            if allowed_set is not None and int(flat_idx) not in allowed_set:
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
    candidate_flat_indices: Int[np.ndarray, "K 1"] | None = None,
) -> tuple[Float[np.ndarray, "M 2"], Float[np.ndarray, "M 1"]]:
    """train で使用したセルを除く有効セルから test 点を取得する (セル格子に整列した座標)

    candidate_flat_indices が与えられた場合、その集合の中からのみ選ぶ
    (pool 制約。None なら全有効セルが候補になる)。

    Parameters
    ----------
    rss_dbm_gt              : (H, W) 真値マップ (欠測は nan)
    cell_size_m             : セルサイズ [m]
    test_size               : 取得するセル数。None なら該当する全件
    rng                     : 乱数生成器 (外部から受け取る)
    excluded_flat_indices   : train で採用済みのフラットセルインデックス
    candidate_flat_indices  : 選択候補を制限する場合のフラットインデックス集合 (pool 制約)

    Returns
    -------
    coords  : shape (M, 2) 左下端座標 (x, y) [m]
    rss_dbm : shape (M, 1) 対応する値
    """
    height, width = rss_dbm_gt.shape
    valid_flat_indices = np.flatnonzero(~np.isnan(rss_dbm_gt))  # (num_valid,)

    if candidate_flat_indices is not None:
        valid_flat_indices = np.intersect1d(valid_flat_indices, candidate_flat_indices, assume_unique=False)

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
    Float[np.ndarray, "N 2"],
    Float[np.ndarray, "N 1"],
    Float[np.ndarray, "M 2"],
    Float[np.ndarray, "M 1"],
]:
    """(0, 0)-(area_size_m, area_size_m) から train / test 点をサンプリングする (pool 制約なし)

    注意: このバージョンは全有効セルを対象にするため、test_prod のリークを防ぐ
    仕組みを持たない。チューニング・本番実験からは呼ばず、代わりに
    sample_train_test_points_from_pool / sample_train_points_from_pool を使うこと。
    """
    train_coords, train_rss_dbm, train_flat_indices = _sample_train_points(
        rss_dbm_gt, area_size_m, cell_size_m, train_size, rng
    )
    test_coords, test_rss_dbm = _sample_test_cells(
        rss_dbm_gt, cell_size_m, test_size, rng, excluded_flat_indices=train_flat_indices
    )
    return train_coords, train_rss_dbm, test_coords, test_rss_dbm


def sample_train_test_points_from_pool(
    rss_dbm_gt: Float[np.ndarray, "H W"],
    area_size_m: float,
    cell_size_m: float,
    train_size: int,
    test_size: int | None,
    rng: np.random.Generator,
    pool_flat_indices: Int[np.ndarray, "P 1"],
) -> tuple[
    Float[np.ndarray, "N 2"],
    Float[np.ndarray, "N 1"],
    Float[np.ndarray, "M 2"],
    Float[np.ndarray, "M 1"],
]:
    """pool_flat_indices の範囲内だけで train / test 点をサンプリングする

    チューニング (train_tune / test_tune) がこれを通る。test_prod のセルには絶対に触れない。

    Parameters
    ----------
    pool_flat_indices : PoolTestSplit.pool_flat_indices (test_prod を除いた候補セル)
    その他は sample_train_test_points と同じ
    """
    train_coords, train_rss_dbm, train_flat_indices = _sample_train_points(
        rss_dbm_gt, area_size_m, cell_size_m, train_size, rng, allowed_flat_indices=pool_flat_indices
    )
    test_coords, test_rss_dbm = _sample_test_cells(
        rss_dbm_gt,
        cell_size_m,
        test_size,
        rng,
        excluded_flat_indices=train_flat_indices,
        candidate_flat_indices=pool_flat_indices,
    )
    return train_coords, train_rss_dbm, test_coords, test_rss_dbm


def sample_train_points_from_pool(
    rss_dbm_gt: Float[np.ndarray, "H W"],
    area_size_m: float,
    cell_size_m: float,
    train_size: int,
    rng: np.random.Generator,
    pool_flat_indices: Int[np.ndarray, "P 1"],
) -> tuple[Float[np.ndarray, "N 2"], Float[np.ndarray, "N 1"]]:
    """pool_flat_indices の範囲内だけで train 点のみをサンプリングする (test は作らない)

    train_prod 用。test_prod (PoolTestSplit.test_flat_indices) はこの関数のスコープに
    一切現れない。

    Parameters
    ----------
    pool_flat_indices : PoolTestSplit.pool_flat_indices (test_prod を除いた候補セル)
    """
    coords, rss_dbm, _used_flat_indices = _sample_train_points(
        rss_dbm_gt, area_size_m, cell_size_m, train_size, rng, allowed_flat_indices=pool_flat_indices
    )
    return coords, rss_dbm


def sample_all_valid_cells(
    rss_dbm_gt: Float[np.ndarray, "H W"],
    cell_size_m: float,
) -> tuple[Float[np.ndarray, "V 2"], Float[np.ndarray, "V 1"]]:
    """欠測でない全セル (pool + test_prod の両方) を取得する (可視化専用)

    train/test の区別を一切行わない。この関数の戻り値を評価 (RMSE計算) に
    使ってはならない。あくまでマップ補間・可視化のための全点予測用。
    """
    height, width = rss_dbm_gt.shape
    valid_flat_indices = np.flatnonzero(~np.isnan(rss_dbm_gt))
    rows, cols = np.unravel_index(valid_flat_indices, (height, width))
    coords = cell_lower_left(rows, cols, cell_size_m)
    rss_dbm = rss_dbm_gt[rows, cols].reshape(-1, 1)
    return coords, rss_dbm
