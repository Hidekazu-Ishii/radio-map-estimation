# ruff: noqa: F722
"""
学習・予測に渡すデータ構造の定義

TrainData : 学習点 (座標 + 接続TX座標 + TXインデックス + 周波数 + 送信電力 + 受信機高さ + RSS観測値)
TestData  : 予測点 (座標 + 接続TX座標 + TXインデックス + 周波数 + 送信電力 + 受信機高さ + RSS真値)

座標系:
    coords はグリッドの左下端座標 (x, y) [m] (ローカル座標)
    x = col * cell_size_m
    y = row * cell_size_m
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int
from numpy import ndarray

from ..utils.grid_transform import bldg_index_to_coord, coord_to_bldg_index


@dataclass(frozen=True, slots=True)
class TrainData:
    """学習用データ

    Attributes
    ----------
    coords       : 学習点の座標 (x, y)
    tx_coords    : 接続TX座標 (tx_x, tx_y, tx_z) [m] (ローカル座標)
    rss_dbm_obs  : RSS観測値 [dBm] (rss_dbm_gt からサンプリング、ノイズ付加済み)
    freq_hz      : 搬送波周波数 [Hz] (全点共通値を (M, 1) に展開)
    tx_power_dbm : 送信電力 [dBm] (全点共通値を (M, 1) に展開)
    rx_height_m  : 受信機高さ [m] (全点共通値を (M, 1) に展開)
    """

    coords: Int[np.ndarray, "N 2"]
    tx_coords: Float[np.ndarray, "N 3"]
    rss_dbm_obs: Float[np.ndarray, "N 1"]
    freq_hz: Float[np.ndarray, "N 1"]
    tx_power_dbm: Float[np.ndarray, "N 1"]
    rx_height_m: Float[np.ndarray, "N 1"]

    def __len__(self) -> int:
        return len(self.rss_dbm_obs)


@dataclass(frozen=True, slots=True)
class TestData:
    """予測・評価用データ

    Attributes
    ----------
    coords       : 予測点の座標 (x, y)
    tx_coords    : 接続TX座標 (tx_x, tx_y, tx_z) [m] (ローカル座標)
    rss_dbm_gt   : RSS真値 [dBm] (評価用、ノイズ付加済み)
    以下 TrainData と同じ値だが明示的に保持
    freq_hz      : 搬送波周波数 [Hz] (全点共通値を (M, 1) に展開)
    tx_power_dbm : 送信電力 [dBm] (全点共通値を (M, 1) に展開)
    rx_height_m  : 受信機高さ [m] (全点共通値を (M, 1) に展開)
    """

    coords: Int[np.ndarray, "M 2"]
    tx_coords: Float[np.ndarray, "M 3"]
    rss_dbm_gt: Float[np.ndarray, "M 1"]
    freq_hz: Float[np.ndarray, "M 1"]
    tx_power_dbm: Float[np.ndarray, "M 1"]
    rx_height_m: Float[np.ndarray, "M 1"]

    def __len__(self) -> int:
        return len(self.rss_dbm_gt)


@dataclass(frozen=True, slots=True)
class GridInfo:
    """グリッド全体の静的情報

    Attributes
    ----------
    bldg_mask        : 建物マスク (True = 建物上)margin 込みで拡張済み
    bldg_cell_size_m : bldg_mask のセルサイズ [m]
    cell_size_m      : セルサイズ [m]
    area_size_m      : エリア一辺の長さ [m]
    margin_m         : マージンの長さ [m]
    """

    bldg_mask: Bool[ndarray, "H W"]
    bldg_cell_size_m: float
    cell_size_m: float
    area_size_m: float
    margin_m: float

    @property
    def bldg_num_margin_cells(self) -> int:
        """margin_m に対応する bldg_mask 側のセル数"""
        return round(self.margin_m / self.bldg_cell_size_m)

    def coord_to_bldg_index(
        self,
        points: Float[ndarray, "N 2"],
    ) -> Int[ndarray, "N 2"]:
        """物理座標 (x, y) → 最近傍グリッド点にスナップ → bldg_mask の (row, col) インデックス

        margin オフセット・範囲外クリップの実体は grid_transform.coord_to_bldg_index に
        一元化されている (build_bldg_mask と同じインデックス体系)
        """
        return coord_to_bldg_index(points, self.bldg_cell_size_m, self.margin_m, self.bldg_mask.shape)

    def bldg_index_to_coord(
        self,
        rows: Int[ndarray, "N 1"],
        cols: Int[ndarray, "N 1"],
    ) -> Float[ndarray, "N 2"]:
        """bldg_mask の (row, col) インデックス → 物理座標 (x, y) [m] (左下端点)

        margin オフセットの実体は grid_transform.bldg_index_to_coord に一元化されている
         (coord_to_bldg_index の逆変換)
        """
        return bldg_index_to_coord(rows, cols, self.bldg_cell_size_m, self.margin_m)
