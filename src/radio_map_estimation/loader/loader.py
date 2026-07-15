# ruff: noqa: F722
"""
学習・予測用データセットおよび静的情報を構築する

読み込み済みデータを受け取り、TrainData / TestData / GridInfo を返す。

責務の分離:
    load_grid_info_and_maps : npz 読み込みと GridInfo 構築のみ (共通処理)
    load_tuning_dataset     : pool 内のみで train_tune / test_tune を作る (チューニング専用)
    load_production_data    : train_prod は pool からサンプリング、
                               test_prod は PoolTestSplit の全件をそのまま使う (再サンプリングしない)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from .dataset import GridInfo, PoolTestSplit, TestData, TrainData
from .sampler import (
    sample_all_valid_cells,
    sample_train_points_from_pool,
    sample_train_test_points_from_pool,
)


def _broadcast_scalar(value: float, num_points: int) -> Float[np.ndarray, "N 1"]:
    """スカラー値を (num_points, 1) に展開する"""
    return np.full((num_points, 1), value, dtype=np.float64)


def _broadcast_tx_coords(
    tx_positions: Float[np.ndarray, "1 3"],
    num_points: int,
) -> Float[np.ndarray, "N 3"]:
    """単一 TX 座標を (num_points, 3) に展開する"""
    return np.tile(tx_positions, (num_points, 1))


@dataclass(frozen=True, slots=True)
class _RadioMapArrays:
    """npz から取り出した生配列と静的情報 (loader 内部専用の受け渡し用)"""

    grid_info: GridInfo
    rss_dbm_gt: Float[np.ndarray, "H W"]
    tx_positions: Float[np.ndarray, "1 3"]
    cell_size_m: float
    freq_hz: float
    tx_power_dbm: float
    rx_height_m: float


def load_grid_info_and_maps(
    bldgmap_data: np.lib.npyio.NpzFile,
    radiomap_data: np.lib.npyio.NpzFile,
) -> _RadioMapArrays:
    """npz データから GridInfo と生配列を取り出す (train/test サンプリングは行わない)"""
    bldg_mask: np.ndarray = bldgmap_data["bldg_mask"]
    bldg_cell_size_m: float = float(bldgmap_data["bldg_cell_size_m"])
    area_size_m: float = float(bldgmap_data["area_size_m"])
    margin_m: float = float(bldgmap_data["margin_m"])

    rss_dbm_gt: np.ndarray = radiomap_data["rss_dbm_gt"]  # (H, W) float64, 未検出は nan
    tx_positions: np.ndarray = radiomap_data["tx_positions"]  # (1, 3) float64
    cell_size_m: float = float(radiomap_data["cell_size_m"])
    freq_hz: float = float(radiomap_data["freq_hz"])
    tx_power_dbm: float = float(radiomap_data["tx_power_dbm"])
    rx_height_m: float = float(radiomap_data["rx_height_m"])

    grid_info = GridInfo(
        bldg_mask=bldg_mask,
        bldg_cell_size_m=bldg_cell_size_m,
        cell_size_m=cell_size_m,
        area_size_m=area_size_m,
        margin_m=margin_m,
    )

    return _RadioMapArrays(
        grid_info=grid_info,
        rss_dbm_gt=rss_dbm_gt,
        tx_positions=tx_positions,
        cell_size_m=cell_size_m,
        freq_hz=freq_hz,
        tx_power_dbm=tx_power_dbm,
        rx_height_m=rx_height_m,
    )


def _build_train_data(
    coords: Float[np.ndarray, "N 2"],
    rss_dbm: Float[np.ndarray, "N 1"],
    arrays: _RadioMapArrays,
) -> TrainData:
    n = len(coords)
    return TrainData(
        coords=coords,
        tx_coords=_broadcast_tx_coords(arrays.tx_positions, n),
        rss_dbm_obs=rss_dbm,
        freq_hz=_broadcast_scalar(arrays.freq_hz, n),
        tx_power_dbm=_broadcast_scalar(arrays.tx_power_dbm, n),
        rx_height_m=_broadcast_scalar(arrays.rx_height_m, n),
    )


def _build_test_data(
    coords: Float[np.ndarray, "M 2"],
    rss_dbm: Float[np.ndarray, "M 1"],
    arrays: _RadioMapArrays,
) -> TestData:
    m = len(coords)
    return TestData(
        coords=coords,
        tx_coords=_broadcast_tx_coords(arrays.tx_positions, m),
        rss_dbm_gt=rss_dbm,
        freq_hz=_broadcast_scalar(arrays.freq_hz, m),
        tx_power_dbm=_broadcast_scalar(arrays.tx_power_dbm, m),
        rx_height_m=_broadcast_scalar(arrays.rx_height_m, m),
    )


def load_tuning_dataset(
    bldgmap_data: np.lib.npyio.NpzFile,
    radiomap_data: np.lib.npyio.NpzFile,
    split: PoolTestSplit,
    train_size: int,
    test_size: int,
    rng: np.random.Generator,
) -> tuple[TrainData, TestData, GridInfo]:
    """pool 内のみで train_tune / test_tune を作る (チューニング専用)

    Parameters
    ----------
    split      : PoolTestSplit。split.pool_flat_indices のみが使われ、
                 split.test_flat_indices はこの関数のスコープに一切現れない
    train_size : train_tune の点数
    test_size  : test_tune の点数 (None 不可。pool を使い切らないよう明示指定する)
    rng        : 乱数生成器 (外部から受け取る)

    Raises
    ------
    ValueError
        split.grid_shape が実データの rss_dbm_gt.shape と一致しない場合
        (誤った split ファイルを渡した可能性が高いため fail loudly)
    """
    arrays = load_grid_info_and_maps(bldgmap_data, radiomap_data)

    if arrays.rss_dbm_gt.shape != split.grid_shape:
        raise ValueError(
            f"PoolTestSplit.grid_shape {split.grid_shape} does not match "
            f"rss_dbm_gt.shape {arrays.rss_dbm_gt.shape}. Wrong split file?"
        )

    train_coords, train_rss_dbm, test_coords, test_rss_dbm = sample_train_test_points_from_pool(
        arrays.rss_dbm_gt,
        arrays.grid_info.area_size_m,
        arrays.cell_size_m,
        train_size=train_size,
        test_size=test_size,
        rng=rng,
        pool_flat_indices=split.pool_flat_indices,
    )

    train = _build_train_data(train_coords, train_rss_dbm, arrays)
    test = _build_test_data(test_coords, test_rss_dbm, arrays)
    return train, test, arrays.grid_info


def load_production_data(
    bldgmap_data: np.lib.npyio.NpzFile,
    radiomap_data: np.lib.npyio.NpzFile,
    split: PoolTestSplit,
    train_size: int,
    rng: np.random.Generator,
) -> tuple[TrainData, TestData, GridInfo]:
    """本番実験用データを作る

    train_prod : pool からサンプリング
    test_prod  : split.test_coords / split.test_rss_dbm をそのまま使う (再サンプリングしない、1回だけ評価する前提)
    """
    arrays = load_grid_info_and_maps(bldgmap_data, radiomap_data)

    if arrays.rss_dbm_gt.shape != split.grid_shape:
        raise ValueError(
            f"PoolTestSplit.grid_shape {split.grid_shape} does not match "
            f"rss_dbm_gt.shape {arrays.rss_dbm_gt.shape}. Wrong split file?"
        )

    train_coords, train_rss_dbm = sample_train_points_from_pool(
        arrays.rss_dbm_gt,
        arrays.grid_info.area_size_m,
        arrays.cell_size_m,
        train_size,
        rng,
        split.pool_flat_indices,
    )

    train = _build_train_data(train_coords, train_rss_dbm, arrays)
    test = _build_test_data(split.test_coords, split.test_rss_dbm, arrays)
    return train, test, arrays.grid_info


def load_full_map_data(
    bldgmap_data: np.lib.npyio.NpzFile,
    radiomap_data: np.lib.npyio.NpzFile,
) -> tuple[TestData, GridInfo]:
    """可視化専用: 全有効セル (pool + test_prod の両方) への予測用データを作る

    train_prod に選ばれなかった点には、test_prod でない点 (どちらの集合にも
    属さない残りのプール点) も含まれる。これらは学習に使われておらず、かつ
    正式な評価対象でもないため、モデルの真の汎化性能を代表する点ではない。

    したがって、この関数の戻り値 (TestData.rss_dbm_gt を含む) を
    RMSE 等の性能評価に使ってはならない。あくまで見た目のなめらかな
    マップを作るための補間・可視化専用。
    """
    arrays = load_grid_info_and_maps(bldgmap_data, radiomap_data)
    coords, rss_dbm = sample_all_valid_cells(arrays.rss_dbm_gt, arrays.cell_size_m)
    full_map = _build_test_data(coords, rss_dbm, arrays)
    return full_map, arrays.grid_info
