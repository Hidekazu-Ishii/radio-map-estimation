# ruff: noqa: F722
"""
学習・予測用データセットおよび静的情報を構築する

読み込み済みデータを受け取り、TrainData / TestData / GridInfo を返す。
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float

from .dataset import GridInfo, TestData, TrainData
from .sampler import sample_train_test_points


def _broadcast_scalar(value: float, num_points: int) -> Float[np.ndarray, "N 1"]:
    """スカラー値を (num_points, 1) に展開する"""
    return np.full((num_points, 1), value, dtype=np.float64)


def _broadcast_tx_coords(
    tx_positions: Float[np.ndarray, "1 3"],
    num_points: int,
) -> Float[np.ndarray, "N 3"]:
    """単一 TX 座標を (num_points, 3) に展開する"""
    return np.tile(tx_positions, (num_points, 1))


def load_dataset(
    bldgmap_data: np.lib.npyio.NpzFile,
    radiomap_data: np.lib.npyio.NpzFile,
    train_size: int,
    test_size: int | None,
    rng: np.random.Generator,
) -> tuple[TrainData, TestData, GridInfo]:
    """読み込み済み npz データから train / test データを返す

    Parameters
    ----------
    bldgmap_data  : np.load() で読み込んだ建物マップの NpzFile オブジェクト
    radiomap_data : np.load() で読み込んだ電波マップの NpzFile オブジェクト
    train_size    : 学習点数 (エントリポイントで train_rate から計算済み)
    test_size     : 予測 (評価) 点数。None なら train 使用セルを除く全有効セル
    rng           : 乱数生成器 (外部から受け取る、モジュール内で固定しない)

    Returns
    -------
    train     : TrainData
    test      : TestData
    grid_info : GridInfo

    Raises
    ------
    ValueError
        train_size + test_size が観測可能点数を超える場合
    """
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

    grid_info = GridInfo(bldg_mask, bldg_cell_size_m, cell_size_m, area_size_m, margin_m)

    # train / test 点のサンプリング
    train_coords, train_rss_dbm, test_coords, test_rss_dbm = sample_train_test_points(
        rss_dbm_gt, area_size_m, cell_size_m, train_size=train_size, test_size=test_size, rng=rng
    )

    num_train = len(train_coords)
    num_test = len(test_coords)

    train = TrainData(
        coords=train_coords,
        tx_coords=_broadcast_tx_coords(tx_positions, num_train),
        rss_dbm_obs=train_rss_dbm,
        freq_hz=_broadcast_scalar(freq_hz, num_train),
        tx_power_dbm=_broadcast_scalar(tx_power_dbm, num_train),
        rx_height_m=_broadcast_scalar(rx_height_m, num_train),
    )
    test = TestData(
        coords=test_coords,
        tx_coords=_broadcast_tx_coords(tx_positions, num_test),
        rss_dbm_gt=test_rss_dbm,
        freq_hz=_broadcast_scalar(freq_hz, num_test),
        tx_power_dbm=_broadcast_scalar(tx_power_dbm, num_test),
        rx_height_m=_broadcast_scalar(rx_height_m, num_test),
    )

    return train, test, grid_info
