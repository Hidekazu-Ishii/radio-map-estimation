"""
radio_map.npz から学習・予測用データセットを構築する

処理フロー:
    1. 読み込み済み npz データから配列を取得
    2. rss_dbm_gt の nan マスクから観測可能セルを再構成
    3. sampler でセルインデックスをサンプリング (train / test disjoint)
    4. セルインデックス → グリッド左下端座標 (x, y) に変換
    5. tx_association + tx_positions → 接続TX座標を付与
    6. TrainData / TestData を返す

Note:
    npz の読み込み (np.load) とtrain_size の計算はエントリポイントで行う。
    本モジュールは読み込み済みデータを受け取り、TrainData / TestData を返す。
"""

from __future__ import annotations

import logging

import numpy as np

from .dataset import TestData, TrainData
from .sampler import cell_lower_left, resolve_tx_positions, sample_train_test_indices

logger = logging.getLogger(__name__)


def load_dataset(
    data: np.lib.npyio.NpzFile,
    train_size: int,
    test_size: int,
    rng: np.random.Generator,
) -> tuple[TrainData, TestData]:
    """読み込み済み npz データから train / test データを返す

    Parameters
    ----------
    data       : np.load() で読み込んだ NpzFile オブジェクト
    train_size : 学習点数 (エントリポイントで train_rate から計算済み)
    test_size  : 予測 (評価) 点数
    rng        : 乱数生成器 (外部から受け取る、モジュール内で固定しない)

    Returns
    -------
    train : TrainData
    test  : TestData

    Raises
    ------
    ValueError
        train_size + test_size が観測可能点数を超える場合
    """
    rss_dbm_gt: np.ndarray = data["rss_dbm_gt"]  # (H, W) float64, 建物上・未検出は nan
    tx_association: np.ndarray = data["tx_association"]  # (H, W) int32
    tx_positions: np.ndarray = data["tx_positions"]  # (T, 3) float64
    cell_size_m: float = float(data["cell_size_m"])

    # 観測可能マスクを nan 埋めから再構成 (nan でない = 観測可能)
    mask_observed: np.ndarray = ~np.isnan(rss_dbm_gt)  # (H, W) bool

    logger.info(
        "Observable cells: %d  (train=%d, test=%d)",
        int(mask_observed.sum()),
        train_size,
        test_size,
    )

    # セルインデックスのサンプリング (train / test disjoint)
    train_rows, train_cols, test_rows, test_cols = sample_train_test_indices(
        mask_observed, train_size, test_size, rng
    )

    # グリッド左下端座標 (x, y) に変換
    train_coords = cell_lower_left(train_rows, train_cols, cell_size_m)  # (N, 2)
    test_coords = cell_lower_left(test_rows, test_cols, cell_size_m)  # (M, 2)

    # 接続TX座標・インデックスを付与
    train_tx_coords, train_tx_indices = resolve_tx_positions(
        train_rows, train_cols, tx_association, tx_positions
    )
    test_tx_coords, test_tx_indices = resolve_tx_positions(test_rows, test_cols, tx_association, tx_positions)

    # rows/cols は (N, 1) のため squeeze して 2D インデックスとして使用
    train = TrainData(
        coords=train_coords,
        tx_coords=train_tx_coords,
        tx_indices=train_tx_indices,
        rss_dbm_obs=rss_dbm_gt[train_rows.squeeze(1), train_cols.squeeze(1)].reshape(-1, 1),
    )
    test = TestData(
        coords=test_coords,
        tx_coords=test_tx_coords,
        tx_indices=test_tx_indices,
        rss_dbm_gt=rss_dbm_gt[test_rows.squeeze(1), test_cols.squeeze(1)].reshape(-1, 1),
    )

    logger.info(
        "Dataset loaded: train=%d, test=%d, num_tx=%d",
        len(train),
        len(test),
        len(tx_positions),
    )

    return train, test
