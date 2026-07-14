# ruff: noqa: F722
"""
学習・予測に渡すデータ構造の定義

TrainData : 学習点 (座標 + 接続TX座標 + TXインデックス + 周波数 + 送信電力 + 受信機高さ + RSS観測値)
TestData  : 予測点 (座標 + 接続TX座標 + TXインデックス + 周波数 + 送信電力 + 受信機高さ + RSS真値)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int
from numpy import ndarray

from ..utils.grid_transform import bldg_index_to_coord, coord_to_bldg_index
from .sampler import create_pool_test_split


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


@dataclass(frozen=True, slots=True)
class PoolTestSplit:
    """test_prod (本番評価用、固定) と pool (チューニング + 本番学習用) の分割

    一度 create() で確定したら save() でファイルに永続化し、以後は load() で
    読み込むだけにする (乱数で毎回作り直さない)。

    test_flat_indices はチューニング処理のどの関数にも渡してはならない。
    渡してよいのは pool_flat_indices のみ。

    Attributes
    ----------
    test_flat_indices : test_prod に属する有効セルのフラットインデックス (固定・不変)
    pool_flat_indices  : それ以外の有効セル (チューニング + 本番学習で使用可能)
    grid_shape         : (H, W) rss_dbm_gt の形状。読み込み時の整合性チェック用
    """

    test_flat_indices: Int[ndarray, "T 1"]
    pool_flat_indices: Int[ndarray, "P 1"]
    grid_shape: tuple[int, int]

    @classmethod
    def create(
        cls,
        rss_dbm_gt: Float[ndarray, "H W"],
        test_size: int,
        rng: np.random.Generator,
    ) -> PoolTestSplit:
        """全有効セルから test_prod を一度だけ確定する (新規生成)

        Parameters
        ----------
        rss_dbm_gt : (H, W) 真値マップ (欠測は nan)
        test_size  : test_prod のセル数
        rng        : 乱数生成器 (外部から受け取る、この呼び出しの中でのみ使う)
        """
        test_flat_indices, pool_flat_indices = create_pool_test_split(rss_dbm_gt, test_size, rng)
        return cls(
            test_flat_indices=test_flat_indices,
            pool_flat_indices=pool_flat_indices,
            grid_shape=rss_dbm_gt.shape,
        )

    def save(self, path: Path) -> None:
        """分割結果を npz に保存する (再現性確保のため一度だけ実行する想定)"""
        np.savez(
            path,
            test_flat_indices=self.test_flat_indices,
            pool_flat_indices=self.pool_flat_indices,
            grid_shape=np.array(self.grid_shape),
        )

    @classmethod
    def load(cls, path: Path) -> PoolTestSplit:
        """保存済みの分割結果を読み込む

        Raises
        ------
        FileNotFoundError
            path が存在しない場合 (fail loudly: 黙って新規生成しない)
        """
        if not path.exists():
            raise FileNotFoundError(
                f"PoolTestSplit not found at {path}. "
                "Run the split-creation entry point once before tuning/production."
            )
        data = np.load(path)
        return cls(
            test_flat_indices=data["test_flat_indices"],
            pool_flat_indices=data["pool_flat_indices"],
            grid_shape=tuple(int(x) for x in data["grid_shape"]),  # type: ignore
        )
