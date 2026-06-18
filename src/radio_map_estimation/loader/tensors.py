# ruff: noqa: F722
"""
TrainData / TestData の numpy 配列を cuda Tensor に変換するモジュール
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

_DEVICE = torch.device("cuda")


@dataclass(frozen=True, slots=True)
class TrainTensors:
    """学習用 cuda Tensor

    Attributes
    ----------
    coords      : グリッド左下端座標 (x, y) [m]
    tx_coords   : 接続TX座標 (tx_x, tx_y, tx_z) [m] (モデルへの入力特徴量)
    tx_indices  : 接続TXインデックス (モデルの振り分けキー)
    rss_dbm_obs : RSS観測値 [dBm] (ノイズ付加済み)
    """

    coords: Float[Tensor, "N 2"]
    tx_coords: Float[Tensor, "N 3"]
    tx_indices: Int[Tensor, "N 1"]
    rss_dbm_obs: Float[Tensor, "N 1"]

    @classmethod
    def from_train_data(
        cls, train_data
    ) -> TrainTensors:  # TrainData (循環 import を避けるため型注釈は文字列回避)
        """TrainData (numpy) → TrainTensors (cuda Tensor) に変換する"""
        return cls(
            coords=torch.as_tensor(train_data.coords, dtype=torch.float32).to(_DEVICE),
            tx_coords=torch.as_tensor(train_data.tx_coords, dtype=torch.float32).to(_DEVICE),
            tx_indices=torch.as_tensor(train_data.tx_indices, dtype=torch.long).to(_DEVICE),
            rss_dbm_obs=torch.as_tensor(train_data.rss_dbm_obs, dtype=torch.float32).to(_DEVICE),
        )

    def __len__(self) -> int:
        return self.rss_dbm_obs.shape[0]


@dataclass(frozen=True, slots=True)
class TestTensors:
    """予測・評価用 cuda Tensor

    Attributes
    ----------
    coords      : グリッド左下端座標 (x, y) [m]
    tx_coords   : 接続TX座標 (tx_x, tx_y, tx_z) [m] (モデルへの入力特徴量)
    tx_indices  : 接続TXインデックス (モデルの振り分けキー)
    rss_dbm_gt  : RSS真値 [dBm] (評価用、ノイズ付加済み)
    """

    coords: Float[Tensor, "M 2"]
    tx_coords: Float[Tensor, "M 3"]
    tx_indices: Int[Tensor, "M 1"]
    rss_dbm_gt: Float[Tensor, "M 1"]

    @classmethod
    def from_test_data(cls, test_data) -> TestTensors:  # TestData
        """TestData (numpy) → TestTensors (cuda Tensor) に変換する"""
        return cls(
            coords=torch.as_tensor(test_data.coords, dtype=torch.float32).to(_DEVICE),
            tx_coords=torch.as_tensor(test_data.tx_coords, dtype=torch.float32).to(_DEVICE),
            tx_indices=torch.as_tensor(test_data.tx_indices, dtype=torch.long).to(_DEVICE),
            rss_dbm_gt=torch.as_tensor(test_data.rss_dbm_gt, dtype=torch.float32).to(_DEVICE),
        )

    def __len__(self) -> int:
        return self.rss_dbm_gt.shape[0]
