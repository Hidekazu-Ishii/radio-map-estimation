# ruff: noqa: F722
"""
FFNN (Feedforward Neural Network) パスロスモデル

入力特徴量 (3次元) :
    log10(d)   : 3D距離の対数 [m] → min-max正規化 → [0, 1]
    cos(θ)    : 水平角の余弦 → (x+1)/2 → [0, 1]
    sin(θ)    : 水平角の正弦 → (x+1)/2 → [0, 1]

出力:
    pathloss [dB] → min-max正規化 → [0, 1] (訓練時)
    予測時は逆正規化して rss_mean = tx_power_dbm - pathloss を返す

ネットワーク構造:
    Linear(3, n_neurons) → ReLU → [...] → Linear(n_neurons, 1)
    隠れ層数は n_layers で指定 (1〜3)

最適化:
    Adam, lr=0.01, batch_size=64
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ..loader.dataset import GridInfo
from .base import FitResult, PathLossModel

# ------------------------------------------------------------------
# ネットワーク定義 (変更なし)
# ------------------------------------------------------------------


class _FFNet(nn.Module):
    """n_layers 個の隠れ層をもつ feedforward ネットワーク

    Parameters
    ----------
    n_neurons : 各隠れ層のニューロン数 (全層共通)
    n_layers  : 隠れ層の数 (1〜3)
    """

    def __init__(self, n_neurons: int, n_layers: int) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        layers: list[nn.Module] = []
        layers += [nn.Linear(3, n_neurons), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.ReLU()]
        layers.append(nn.Linear(n_neurons, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ------------------------------------------------------------------
# FFNNModel
# ------------------------------------------------------------------


class FFNNModel(PathLossModel):
    """FFNNパスロスモデル

    bldg_mask (建物マスク) は使用しない. fit/predict_mean は grid_info を
    受け取るがインターフェース準拠のためのみで、中身では参照しない.

    Attributes
    ----------
    _n_neurons  : 各隠れ層のニューロン数
    _n_layers   : 隠れ層の数 (1〜3)
    _n_epochs   : 学習エポック数
    _batch_size : ミニバッチサイズ
    _lr         : Adam 初期学習率
    _net        : 学習済みネットワーク (fit() 後に設定)
    _norm       : 正規化パラメータ (fit() 後に設定)
    """

    def __init__(
        self,
        n_neurons: int,
        n_layers: int,
        n_epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        self._n_neurons = n_neurons
        self._n_layers = n_layers
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._lr = lr
        self._net: _FFNet | None = None
        self._norm: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # PathLossModel インターフェース
    # ------------------------------------------------------------------

    def fit(
        self,
        coords: Float[ndarray, "N 2"],
        tx_coords: Float[ndarray, "N 3"],
        rx_height_m: Float[ndarray, "N 1"],
        freq_hz: Float[ndarray, "N 1"],
        tx_power_dbm: Float[ndarray, "N 1"],
        rss_dbm_obs: Float[ndarray, "N 1"],
        grid_info: GridInfo,
        rng: np.random.Generator,
    ) -> FitResult:
        """Adam で FFNN をフィットする (grid_info は未使用)"""
        # --- 特徴量・ターゲット計算 ---
        d = self.compute_3d_distance(coords, tx_coords, rx_height_m)  # (N,1)
        azimuth = self.compute_azimuth(coords, tx_coords)  # (N,1)
        pl_obs = tx_power_dbm - rss_dbm_obs  # (N,1) pathloss観測値

        log10_d = np.log10(d)  # (N,1)
        cos_az = np.cos(azimuth)  # (N,1)
        sin_az = np.sin(azimuth)  # (N,1)

        # --- 正規化パラメータをフィット (訓練データから推定) ---
        log10_d_min = float(log10_d.min())
        log10_d_max = float(log10_d.max())
        pl_min = float(pl_obs.min())
        pl_max = float(pl_obs.max())
        self._norm = {
            "log10_d_min": log10_d_min,
            "log10_d_max": log10_d_max,
            "pl_min": pl_min,
            "pl_max": pl_max,
        }

        # --- 正規化 ---
        x = self._normalize_features(log10_d, cos_az, sin_az)  # (N,3)
        y = self._normalize_pl(pl_obs)  # (N,1)

        # --- Tensor 変換 ---
        x_t = torch.from_numpy(x).float()
        y_t = torch.from_numpy(y).float()

        # --- 学習 (再現性のため rng から seed を派生) ---
        seed = int(rng.integers(0, 2**31))
        torch.manual_seed(seed)  # 重み初期化の再現性
        self._net = _FFNet(self._n_neurons, self._n_layers)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        criterion = nn.MSELoss()

        torch_gen = torch.Generator()
        torch_gen.manual_seed(seed)  # シャッフル順序の再現性
        dataset = TensorDataset(x_t, y_t)
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            generator=torch_gen,
        )

        self._net.train()
        for _ in range(self._n_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()

        # --- 訓練RMSE (元スケール) ---
        self._net.eval()
        with torch.no_grad():
            pl_pred_norm = self._net(x_t).numpy()
        pl_pred = self._denormalize_pl(pl_pred_norm)
        rss_pred = tx_power_dbm - pl_pred
        residuals = rss_dbm_obs - rss_pred
        rmse = float(np.sqrt(np.mean(residuals**2)))

        return FitResult(
            model_name="ffnn",
            params={
                "n_neurons": float(self._n_neurons),
                "n_layers": float(self._n_layers),
                "n_epochs": float(self._n_epochs),
                "batch_size": float(self._batch_size),
                "lr": float(self._lr),
            },
            norm_stats=self._norm,
            n_samples=int(coords.shape[0]),
            rmse_db=rmse,
        )

    def predict_mean(
        self,
        coords: Float[ndarray, "M 2"],
        tx_coords: Float[ndarray, "M 3"],
        rx_height_m: Float[ndarray, "M 1"],
        freq_hz: Float[ndarray, "M 1"],
        tx_power_dbm: Float[ndarray, "M 1"],
        grid_info: GridInfo,
    ) -> Float[ndarray, "M 1"]:
        """平均RSS [dBm] を返す (grid_info は未使用)"""
        if self._net is None or self._norm is None:
            raise RuntimeError("FFNNModel.fit() must be called before predict_mean()")

        d = self.compute_3d_distance(coords, tx_coords, rx_height_m)
        azimuth = self.compute_azimuth(coords, tx_coords)

        x = self._normalize_features(np.log10(d), np.cos(azimuth), np.sin(azimuth))
        x_t = torch.from_numpy(x).float()

        self._net.eval()
        with torch.no_grad():
            pl_pred_norm = self._net(x_t).numpy()

        pl_pred = self._denormalize_pl(pl_pred_norm)
        return tx_power_dbm - pl_pred

    @property
    def params(self) -> dict[str, float]:
        if self._net is None:
            raise RuntimeError("FFNNModel.fit() must be called before accessing params")
        return {
            "n_neurons": float(self._n_neurons),
            "n_layers": float(self._n_layers),
            "n_epochs": float(self._n_epochs),
            "batch_size": float(self._batch_size),
            "lr": float(self._lr),
        }

    # ------------------------------------------------------------------
    # 正規化ユーティリティ (変更なし)
    # ------------------------------------------------------------------

    def _normalize_features(
        self,
        log10_d: Float[ndarray, "N 1"],
        cos_az: Float[ndarray, "N 1"],
        sin_az: Float[ndarray, "N 1"],
    ) -> Float[ndarray, "N 3"]:
        """特徴量を [0, 1] に正規化して結合する"""
        assert self._norm is not None
        eps = 1e-8
        d_range = self._norm["log10_d_max"] - self._norm["log10_d_min"]
        x_d = (log10_d - self._norm["log10_d_min"]) / (d_range + eps)
        x_cos = (cos_az + 1.0) / 2.0
        x_sin = (sin_az + 1.0) / 2.0
        return np.hstack([x_d, x_cos, x_sin])  # (N,3)

    def _normalize_pl(self, pl: Float[ndarray, "N 1"]) -> Float[ndarray, "N 1"]:
        """pathloss [dB] を [0, 1] に正規化する (訓練データの min/max を使用)"""
        assert self._norm is not None
        eps = 1e-8
        pl_range = self._norm["pl_max"] - self._norm["pl_min"]
        return (pl - self._norm["pl_min"]) / (pl_range + eps)

    def _denormalize_pl(self, pl_norm: Float[ndarray, "N 1"]) -> Float[ndarray, "N 1"]:
        """正規化された pathloss を元スケールに戻す"""
        assert self._norm is not None
        pl_range = self._norm["pl_max"] - self._norm["pl_min"]
        return pl_norm * pl_range + self._norm["pl_min"]
