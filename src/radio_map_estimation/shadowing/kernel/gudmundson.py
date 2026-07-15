# ruff: noqa: F722
"""
Gudmundson カーネル

k(x_i, x_j) = σ_2 · exp(−||x_i − x_j|| · ln2 / d_cor)

ハイパーパラメータ (log スケールで最適化) :
    sigma_2 : シャドウイング分散 σ_2 [dB²]
    d_cor  : 相関距離 d_cor [m]

make_input の出力:
    ユークリッド距離行列 D (M, N)
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from .base import Kernel


class GudmundsonKernel(Kernel):
    """Gudmundson の経験的シャドウイングカーネル (距離ベース)"""

    def __init__(
        self,
        sigma_2_init: float,
        d_cor_init: float,
    ) -> None:
        self._sigma_2 = sigma_2_init
        self._d_cor = d_cor_init

    # ------------------------------------------------------------------
    # Kernel インターフェース
    # ------------------------------------------------------------------

    def make_input(
        self,
        coords_a: Float[ndarray, "M 2"],
        coords_b: Float[ndarray, "N 2"],
        **kwargs,  # TX 座標など — このカーネルでは不使用
    ) -> Float[ndarray, "M N"]:
        """ユークリッド距離行列 D (M, N) を返す"""
        delta = coords_a[:, None, :] - coords_b[None, :, :]  # (M, N, 2)
        return np.sqrt(np.sum(delta**2, axis=-1))  # (M, N)

    def eval(
        self,
        kernel_input: Float[ndarray, "M N"],
        log_params: Float[ndarray, 2],
    ) -> Float[ndarray, "M N"]:
        """内部状態を変更せずにカーネル行列を計算する"""
        sigma_2, d_cor = np.exp(log_params)
        return sigma_2 * np.exp(-kernel_input * np.log(2.0) / d_cor)

    def grad_at(
        self,
        kernel_input: Float[ndarray, "M N"],
        log_params: Float[ndarray, 2],
    ) -> dict[str, Float[ndarray, "M N"]]:
        """∂K/∂log(sigma_2) と ∂K/∂log(d_cor) を返す (log スケール勾配)

        log スケール chain rule:
            ∂K/∂log(σ_2) = ∂K/∂σ_2  · σ_2  = K
            ∂K/∂log(d)  = ∂K/∂d   · d   = K · D · ln2 / d_cor
        """
        K = self.eval(kernel_input, log_params)
        _, d_cor = np.exp(log_params)
        return {
            "sigma_2": K,  # ∂K/∂log(σ_2)
            "d_cor": K * kernel_input * np.log(2.0) / d_cor,  # ∂K/∂log(d)
        }

    def set_log_params(self, log_params: Float[ndarray, 2]) -> None:
        """fit 完了後に内部状態を更新する"""
        self._sigma_2, self._d_cor = np.exp(log_params)

    def __call__(self, kernel_input: Float[ndarray, "M N"]) -> Float[ndarray, "M N"]:
        """fit 後の内部パラメータでカーネル行列を計算する"""
        return self.eval(kernel_input, self.log_params_init)

    # ------------------------------------------------------------------
    # プロパティ
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        return 2

    @property
    def log_params_init(self) -> Float[ndarray, 2]:
        return np.log(np.array([self._sigma_2, self._d_cor]))

    @property
    def param_bounds(self) -> list[tuple[float | None, float | None]]:
        return [(None, None), (None, None)]  # sigma_2, d_cor: 上下限なし

    @property
    def params(self) -> dict[str, float]:
        return {"sigma_2": self._sigma_2, "d_cor": self._d_cor}
