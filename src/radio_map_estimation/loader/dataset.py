# ruff: noqa: F722
"""
学習・予測に渡すデータ構造の定義

TrainData : 学習点 (座標 + 接続TX座標 + TXインデックス + RSS観測値)
TestData  : 予測点 (座標 + 接続TX座標 + TXインデックス + RSS真値)

座標系:
    coords は 10m x 10m グリッドの左下端座標 (x, y) [m] (ローカル座標)
    x = col * cell_size_m
    y = row * cell_size_m
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int


@dataclass(frozen=True, slots=True)
class TrainData:
    """学習用データ

    Attributes
    ----------
    coords      : グリッド左下端座標 (x, y) [m] (ローカル座標)
    tx_coords   : 接続TX座標 (tx_x, tx_y, tx_z) [m] (ローカル座標)
    tx_indices  : 接続TXインデックス (モデルの振り分けキー)
    rss_dbm_obs : RSS観測値 [dBm] (rss_dbm_gt からサンプリング、ノイズ付加済み)
    """

    coords: Float[np.ndarray, "N 2"]
    tx_coords: Float[np.ndarray, "N 3"]
    tx_indices: Int[np.ndarray, "N 1"]
    rss_dbm_obs: Float[np.ndarray, "N 1"]

    def __len__(self) -> int:
        return len(self.rss_dbm_obs)


@dataclass(frozen=True, slots=True)
class TestData:
    """予測・評価用データ

    Attributes
    ----------
    coords      : グリッド左下端座標 (x, y) [m] (ローカル座標)
    tx_coords   : 接続TX座標 (tx_x, tx_y, tx_z) [m] (ローカル座標)
    tx_indices  : 接続TXインデックス (モデルの振り分けキー)
    rss_dbm_gt  : RSS真値 [dBm] (評価用、ノイズ付加済み)
    """

    coords: Float[np.ndarray, "M 2"]
    tx_coords: Float[np.ndarray, "M 3"]
    tx_indices: Int[np.ndarray, "M 1"]
    rss_dbm_gt: Float[np.ndarray, "M 1"]

    def __len__(self) -> int:
        return len(self.rss_dbm_gt)
