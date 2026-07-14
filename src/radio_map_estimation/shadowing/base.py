# ruff: noqa: F722
"""
シャドウイングモデルの抽象基底クラス

パスロスモデルの残差 (シャドウイング成分) を
GPで推定するためのインターフェースを定義する
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from radio_map_estimation.pathloss.base import FitResult


class ShadowingModel(ABC):
    """シャドウイングモデルの抽象基底クラス

    すべてのシャドウイングモデルはこのインターフェースを実装する
    fit() で学習し、predict_mean() / predict_with_uncertainty() で推定を行う
    """

    @abstractmethod
    def fit(
        self,
        coords: Float[ndarray, "N 2"],
        tx_coords: Float[ndarray, "N 3"],
        freq_hz: Float[ndarray, "N 1"],
        residuals: Float[ndarray, "N 1"],
        rng: np.random.Generator,
    ) -> FitResult:
        """残差 (シャドウイング観測値) からモデルをフィットする

        Parameters
        ----------
        coords    : セル左下端座標 (x, y) [m]
        tx_coords : 接続TX座標 (x, y, z) [m]
        freq_hz   : 搬送波周波数 [Hz] (将来の周波数依存カーネル用)
        residuals : パスロスモデルの残差 = rss_obs - rss_mean [dB]

        Returns
        -------
        FitResult : フィット結果 (ハイパーパラメータ・誤差・サンプル数)
        """
        ...

    @abstractmethod
    def predict_mean(
        self,
        coords: Float[ndarray, "M 2"],
        tx_coords: Float[ndarray, "M 3"],
        freq_hz: Float[ndarray, "M 1"],
    ) -> Float[ndarray, "M 1"]:
        """シャドウイング成分の事後平均を返す

        Parameters
        ----------
        coords    : セル左下端座標 (x, y) [m]
        tx_coords : 接続TX座標 (x, y, z) [m]
        freq_hz   : 搬送波周波数 [Hz]

        Returns
        -------
        shadowing_mean : 事後平均 [dB]、shape (M, 1)
        """
        ...

    @abstractmethod
    def predict_with_uncertainty(
        self,
        coords: Float[ndarray, "M 2"],
        tx_coords: Float[ndarray, "M 3"],
        freq_hz: Float[ndarray, "M 1"],
    ) -> tuple[Float[ndarray, "M 1"], Float[ndarray, "M 1"]]:
        """シャドウイング成分の事後平均と事後分散を返す

        Parameters
        ----------
        coords    : セル左下端座標 (x, y) [m]
        tx_coords : 接続TX座標 (x, y, z) [m]
        freq_hz   : 搬送波周波数 [Hz]

        Returns
        -------
        shadowing_mean : 事後平均 [dB]、shape (M, 1)
        shadowing_var  : 事後分散 [dB²]、shape (M, 1)
        """
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, float]:
        """現在のハイパーパラメータを返す (フィット後に有効)"""
        ...
