# ruff: noqa: F722
"""
カーネル抽象基底クラス

設計方針
--------
- カーネルは「入力構築」と「行列計算」の両方に責任を持つ
- GP エンジン (gp.py) は isinstance 分岐を一切持たない
- 最適化ループ中は内部状態を変更しない (eval / grad_at で純粋計算)
- fit 完了後のみ set_log_params で内部状態を更新する

サブクラスの実装義務
--------------------
make_input  : 座標 → カーネル入力 (距離行列 or 差分ベクトルなど)
eval        : カーネル入力 x log_params → カーネル行列 K (内部状態変更なし)
grad_at     : カーネル入力 x log_params → ∂K/∂θ の辞書 (内部状態変更なし)
set_log_params : fit 完了後に内部状態を log_params で更新する
n_params    : ハイパーパラメータ数
log_params_init : 最適化初期値 (log スケール)
param_bounds    : L-BFGS-B 用の上下限 (log スケール)
params      : 現在のハイパーパラメータ (元スケール)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from jaxtyping import Float
from numpy import ndarray


class Kernel(ABC):
    """GP カーネルの抽象基底クラス

    GP エンジンはこのインターフェースのみを通じてカーネルを扱う。
    入力の種類 (距離行列・差分ベクトル・特徴量など) は
    サブクラスの make_input() に隠蔽される。

    make_input の **kwargs 設計
    ---------------------------
    追加情報 (TX 座標など) を必要とするカーネルは kwargs 経由で受け取る。
    不要なカーネルは kwargs を無視するだけでよく、呼び出し側は
    常に同じシグネチャで make_input() を呼べる。

        # TX 座標不要 (GudmundsonKernel など)
        kernel.make_input(coords_a, coords_b)

        # TX 座標必要 (DirectionalKernel など)
        kernel.make_input(coords_a, coords_b, tx_coords_a=..., tx_coords_b=...)
    """

    @abstractmethod
    def make_input(
        self,
        coords_a: Float[ndarray, "M 2"],
        coords_b: Float[ndarray, "N 2"],
        **kwargs,
    ) -> np.ndarray:
        """座標ペアからカーネルへの入力を構築する

        Parameters
        ----------
        coords_a : 受信点座標 (x, y) [m]、shape (M, 2)
        coords_b : 受信点座標 (x, y) [m]、shape (N, 2)
        **kwargs : カーネル固有の追加引数 (TX 座標など)

        Returns
        -------
        カーネルが要求する形式の入力
            例: 距離行列 D (M, N)、{"dist": ..., "cos_theta": ...}
        """
        ...

    @abstractmethod
    def eval(
        self,
        kernel_input: np.ndarray,
        log_params: Float[ndarray, "P 1"],
    ) -> Float[ndarray, "M N"]:
        """内部状態を変更せずにカーネル行列を計算する

        最適化ループ内で呼ばれるため、self の状態を書き換えてはならない。

        Parameters
        ----------
        kernel_input : make_input() の出力
        log_params   : log スケールのハイパーパラメータベクトル

        Returns
        -------
        K : カーネル行列 (M, N)
        """
        ...

    @abstractmethod
    def grad_at(
        self,
        kernel_input: np.ndarray,
        log_params: Float[ndarray, "P 1"],
    ) -> dict[str, Float[ndarray, "M N"]]:
        """内部状態を変更せずに ∂K/∂log(θ) を計算する

        規約: log スケール勾配を返す (∂K/∂log(θ) = ∂K/∂θ · θ)
        GP エンジン側では chain rule の θ 乗算を行わない。

        Parameters
        ----------
        kernel_input : make_input() の出力
        log_params   : log スケールのハイパーパラメータベクトル

        Returns
        -------
        grads : パラメータ名 → ∂K/∂log(θ) の辞書
        """
        ...

    @abstractmethod
    def set_log_params(self, log_params: Float[ndarray, "P 1"]) -> None:
        """fit 完了後に内部状態を更新する

        最適化ループ内では呼ばず、fit() の最後に一度だけ呼ぶこと。
        """
        ...

    @property
    @abstractmethod
    def n_params(self) -> int:
        """ハイパーパラメータ数"""
        ...

    @property
    @abstractmethod
    def log_params_init(self) -> Float[ndarray, "P 1"]:
        """最適化初期値 (log スケール)"""
        ...

    @property
    @abstractmethod
    def param_bounds(self) -> list[tuple[float | None, float | None]]:
        """L-BFGS-B 用の上下限 (log スケール) 、長さ n_params"""
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, float]:
        """現在のハイパーパラメータ (元スケール、fit 完了後に有効)"""
        ...

    @abstractmethod
    def __call__(self, kernel_input: np.ndarray) -> Float[ndarray, "M N"]:
        """現在の内部状態 (fit 後のパラメータ) でカーネル行列を計算する

        fit() → set_log_params() 完了後に _posterior() から呼ばれる。
        各サブクラスが自身の log_params を使って eval() を呼ぶ。
        """
        ...
