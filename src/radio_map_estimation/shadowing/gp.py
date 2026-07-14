# ruff: noqa: RUF002, F722
"""
GP シャドウイング推論エンジン

カーネルに依存しない密ガウス過程推論を実装する
カーネルを差し替えることで任意のシャドウイングモデルを実現できる

設計方針
--------
- カーネルが make_input() を持ち、GP は isinstance 分岐を一切持たない
- コレスキー分解は np.linalg.cholesky (密行列) を使用
  スパース戦略 (CHOLMOD など) は後から gp.py のみを変更して差し替え可能
- 最適化ループ中はカーネルの内部状態を変更しない (eval / grad_at を使用)
- fit 完了後に set_log_params で内部状態を確定する

最適化パラメータ (log スケール)
---------------------------------
[カーネルハイパーパラメータ..., log(σ_n_2)]
→ L-BFGS-B による NLML 最小化 (マルチスタート)

数値安定化
----------
コレスキー分解の安定化に jitter (対角に微小量加算) を使用する
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float
from numpy import ndarray
from scipy.optimize import minimize

from radio_map_estimation.pathloss.base import FitResult

from .base import ShadowingModel
from .kernels.base import Kernel

# 数値安定化用 jitter (対角加算)
_JITTER = 1e-6


class GPShadowingModel(ShadowingModel):
    """GP シャドウイング推論エンジン (密コレスキー)

    Attributes
    ----------
    _kernel          : カーネル (make_input / eval / grad_at を持つ)
    _sigma_n_2_init  : 観測ノイズ分散の初期値 [dB²]
    _sigma_n_2_min   : 観測ノイズ分散の下限 (None なら制限なし)
    _sigma_n_2_max   : 観測ノイズ分散の上限 (None なら制限なし)
    _n_restarts      : マルチスタート試行数 (1 = シングルスタート)
    _max_iter        : L-BFGS-B の最大反復数
    _ftol / _gtol    : 収束判定閾値
    _sigma_n_2       : fit 後の観測ノイズ分散 (fit 前は None)
    _coords_train    : fit 時の訓練座標 (predict で再利用)
    _tx_coords_train : fit 時の訓練 TX 座標 (predict で再利用)
    _alpha           : (K + σ_n_2I)^{-1} y (predict で再利用)
    _L               : コレスキー因子 L (predict の分散計算で再利用)
    """

    def __init__(
        self,
        kernel: Kernel,
        sigma_n_2_init: float,
        sigma_n_2_min: float,
        sigma_n_2_max: float,
        n_restarts: int,
        max_iter: int,
        ftol: float,
        gtol: float,
    ) -> None:
        self._kernel = kernel
        self._sigma_n_2_init = sigma_n_2_init
        self._sigma_n_2_min = sigma_n_2_min
        self._sigma_n_2_max = sigma_n_2_max
        self._n_restarts = n_restarts
        self._max_iter = max_iter
        self._ftol = ftol
        self._gtol = gtol

        # fit 後に設定される内部状態
        self._sigma_n_2: float | None = None
        self._coords_train: Float[ndarray, "N 2"] | None = None
        self._tx_coords_train: Float[ndarray, "N 3"] | None = None
        self._alpha: Float[ndarray, "N 1"] | None = None
        self._L: Float[ndarray, "N N"] | None = None  # コレスキー因子

    # ------------------------------------------------------------------
    # ShadowingModel インターフェース
    # ------------------------------------------------------------------

    def fit(
        self,
        coords: Float[ndarray, "N 2"],
        tx_coords: Float[ndarray, "N 3"],
        freq_hz: Float[ndarray, "N 1"],
        residuals: Float[ndarray, "N 1"],
        rng: np.random.Generator,
    ) -> FitResult:
        """L-BFGS-B で NLML を最小化してハイパーパラメータを推定する

        マルチスタート戦略
        ------------------
        1 試行目は指定初期値をそのまま使用する
        2 試行目以降は log スケールで [-1, +1] の一様乱数を加えて初期値を撹乱する
        全試行のうち NLML が最小の結果を採用する

        最適化ベクトル: [カーネルパラメータ (log) ..., log(σ_n_2)]
        """
        y = residuals.ravel()  # (N,)
        k_input = self._kernel.make_input(coords, coords, tx_coords_a=tx_coords, tx_coords_b=tx_coords)

        # 基準初期値: [カーネルの log 初期値..., log(σ_n_2_init)]
        x0_base = np.append(self._kernel.log_params_init, np.log(self._sigma_n_2_init))

        # σ_n_2 の上下限 (log スケール)
        log_sn_min = np.log(self._sigma_n_2_min) if self._sigma_n_2_min is not None else None
        log_sn_max = np.log(self._sigma_n_2_max) if self._sigma_n_2_max is not None else None
        bounds = [*self._kernel.param_bounds, (log_sn_min, log_sn_max)]

        # マルチスタート初期値リストを構築
        # 1 試行目: 基準初期値、2 試行目以降: log スケールで撹乱
        if self._n_restarts > 1:
            perturbations = rng.uniform(-1.0, 1.0, size=(self._n_restarts - 1, len(x0_base)))
            x0_list = [x0_base] + [x0_base + p for p in perturbations]
        else:
            x0_list = [x0_base]

        # マルチスタート最適化ループ
        best_x: Float[ndarray, "P 1"] | None = None
        best_nlml = np.inf

        for x0 in x0_list:
            result = minimize(
                fun=self._nlml,
                x0=x0,
                args=(k_input, y),
                method="L-BFGS-B",
                jac=self._nlml_grad,
                bounds=bounds,
                options={"maxiter": self._max_iter, "ftol": self._ftol, "gtol": self._gtol},
            )
            # NLML を再評価して最良解を選択 (result.fun は数値誤差を含む場合があるため)
            nlml_val = self._nlml(result.x, k_input, y)
            if nlml_val < best_nlml:
                best_nlml = nlml_val
                best_x = result.x

        assert best_x is not None  # n_restarts >= 1 が保証されているため到達可能

        # 最良解で内部状態を確定
        n_k = self._kernel.n_params
        self._kernel.set_log_params(best_x[:n_k])
        self._sigma_n_2 = float(np.exp(best_x[n_k]))
        self._coords_train = coords
        self._tx_coords_train = tx_coords

        # predict で再利用するコレスキー因子と α をキャッシュ
        K = self._kernel.eval(k_input, best_x[:n_k])
        self._L, self._alpha = self._cholesky_solve(K, self._sigma_n_2, y)

        # 訓練 RMSE
        shadowing_train = (K @ self._alpha).reshape(-1, 1)
        rmse = float(np.sqrt(np.mean((residuals - shadowing_train) ** 2)))

        return FitResult(
            model_name=f"gp_{type(self._kernel).__name__.lower()}",
            params={**self._kernel.params, "sigma_n_2": self._sigma_n_2},
            norm_stats={},
            n_samples=int(coords.shape[0]),
            rmse_db=rmse,
        )

    def predict_mean(
        self,
        coords: Float[ndarray, "M 2"],
        tx_coords: Float[ndarray, "M 3"],
        freq_hz: Float[ndarray, "M 1"],
    ) -> Float[ndarray, "M 1"]:
        """シャドウイング成分の事後平均を返す (分散計算は行わない)"""
        k_star = self._cross_kernel(coords, tx_coords)
        return (k_star @ self._alpha).reshape(-1, 1)

    def predict_with_uncertainty(
        self,
        coords: Float[ndarray, "M 2"],
        tx_coords: Float[ndarray, "M 3"],
        freq_hz: Float[ndarray, "M 1"],
    ) -> tuple[Float[ndarray, "M 1"], Float[ndarray, "M 1"]]:
        """シャドウイング成分の事後平均と事後分散を返す"""
        return self._posterior(coords, tx_coords)

    @property
    def params(self) -> dict[str, float]:
        if self._sigma_n_2 is None:
            raise RuntimeError("fit() を呼んでから params にアクセスしてください")
        return {**self._kernel.params, "sigma_n_2": self._sigma_n_2}

    def covariance_matrix(
        self,
        coords: Float[ndarray, "M 2"],
        tx_coords: Float[ndarray, "M 3"],
    ) -> Float[ndarray, "M M"]:
        """fit 済みカーネルで任意の点集合間の共分散行列を計算する

        可視化用途 (plot_scalar_field_3d 等) で、学習点＋予測点を結合した
        任意の座標集合に対するフルカーネル行列を得るために使う
        predict 時の内部キャッシュ (_L, _alpha) は使わず、fit 済みカーネル
        パラメータのみで K = k(coords, coords) を直接計算する
         (観測ノイズ σ_n_2 は加算しない)

        Parameters
        ----------
        coords    : 対象点群の座標 (x, y) [m]、shape (M, 2)
        tx_coords : 対象点群の TX 座標 (x, y, z) [m]、shape (M, 3)

        Returns
        -------
        K : 共分散行列、shape (M, M)
        """
        if self._sigma_n_2 is None:
            raise RuntimeError("fit() を呼んでから covariance_matrix にアクセスしてください")
        kernel_input = self._kernel.make_input(coords, coords, tx_coords_a=tx_coords, tx_coords_b=tx_coords)
        return self._kernel(kernel_input)

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    def _cross_kernel(
        self,
        coords_test: Float[ndarray, "M 2"],
        tx_coords_test: Float[ndarray, "M 3"],
    ) -> Float[ndarray, "M N"]:
        """テスト点と学習点のクロスカーネル K*N を計算する (predict_mean / _posterior で共有)"""
        if self._coords_train is None:
            raise RuntimeError("fit() を呼んでから predict を呼んでください")
        cross_input = self._kernel.make_input(
            coords_test, self._coords_train, tx_coords_a=tx_coords_test, tx_coords_b=self._tx_coords_train
        )
        return self._kernel(cross_input)

    def _posterior(
        self,
        coords_test: Float[ndarray, "M 2"],
        tx_coords_test: Float[ndarray, "M 3"],
    ) -> tuple[Float[ndarray, "M 1"], Float[ndarray, "M 1"]]:
        """GP 事後分布 (平均・分散) を計算する"""
        if self._alpha is None or self._L is None:
            raise RuntimeError("fit() を呼んでから predict を呼んでください")

        k_star = self._cross_kernel(coords_test, tx_coords_test)  # (M, N) — 共通

        # 事後平均
        mean = (k_star @ self._alpha).reshape(-1, 1)

        # 自己共分散の対角 (テスト点ごとの事前分散)
        self_input = self._kernel.make_input(
            coords_test, coords_test, tx_coords_a=tx_coords_test, tx_coords_b=tx_coords_test
        )
        k_diag = np.diag(self._kernel(self_input))

        # 事後分散: k** − diag(v^T v),  v = L⁻¹ K*N^T  (N, M)
        v = np.linalg.solve(self._L, k_star.T)
        var = np.clip(k_diag - np.sum(v**2, axis=0), 0.0, None).reshape(-1, 1)

        return mean, var

    @staticmethod
    def _cholesky_solve(
        k: Float[ndarray, "N N"],
        sigma_n_2: float,
        y: Float[ndarray, "N 1"],
    ) -> tuple[Float[ndarray, "N N"], Float[ndarray, "N 1"]]:
        """(K + σ_n_2I) のコレスキー分解と α = (K + σ_n_2I)^{-1} y を返す

        数値安定化のため jitter を対角に加算する

        Returns
        -------
        L     : 下三角コレスキー因子 (N, N)
        alpha : (K + σ_n_2I)^{-1} y (N, 1)
        """
        n = k.shape[0]
        k_noisy = k + (sigma_n_2 + _JITTER) * np.eye(n)
        l = np.linalg.cholesky(k_noisy)  # (N, N) 下三角
        alpha = np.linalg.solve(l.T, np.linalg.solve(l, y))  # (N,)
        return l, alpha.reshape(-1, 1)

    def _nlml(
        self,
        log_params: Float[ndarray, "P 1"],
        k_input: np.ndarray,
        y: Float[ndarray, "N 1"],
    ) -> float:
        """負の対数周辺尤度 (NLML)

        NLML = ½ yᵀ α + ½ log|K_noisy| + ½N log(2π)

        カーネルの内部状態を変更しない (eval を使用)
        コレスキー分解が失敗した場合は大きな値を返して最適化を継続する
        """
        n_k = self._kernel.n_params
        sigma_n_2 = float(np.exp(log_params[n_k]))
        n = len(y)

        k = self._kernel.eval(k_input, log_params[:n_k])
        k_noisy = k + (sigma_n_2 + _JITTER) * np.eye(n)

        try:
            l = np.linalg.cholesky(k_noisy)
        except np.linalg.LinAlgError:
            return 1e10

        alpha = np.linalg.solve(l.T, np.linalg.solve(l, y))
        log_det = 2.0 * np.sum(np.log(np.diag(l)))  # log|K_noisy| = 2 Σ log(L_ii)
        nlml = 0.5 * float(y @ alpha) + 0.5 * log_det + 0.5 * n * np.log(2.0 * np.pi)
        return float(nlml)

    def _nlml_grad(
        self,
        log_params: Float[ndarray, "P 1"],
        k_input: np.ndarray,
        y: Float[ndarray, "N 1"],
    ) -> Float[ndarray, "P 1"]:
        """NLML の log_params に関する解析的勾配

        ∂NLML/∂log(θ) = ½ tr(W · ∂K/∂log(θ))
        W = (K+σ_n_2I)^{-1} − α αᵀ   (∂NLML/∂K の係数)

        カーネルの grad_at は log スケール勾配を返す規約なので、
        GP 側で chain rule の θ 乗算は行わない

        σ_n_2 の勾配:
            ∂NLML/∂log(σ_n_2) = ½ tr(W) · σ_n_2
        """
        n_k = self._kernel.n_params
        sigma_n_2 = float(np.exp(log_params[n_k]))
        n = len(y)

        k = self._kernel.eval(k_input, log_params[:n_k])
        k_noisy = k + (sigma_n_2 + _JITTER) * np.eye(n)

        try:
            L = np.linalg.cholesky(k_noisy)
        except np.linalg.LinAlgError:
            return np.zeros(len(log_params))

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))  # (N,)
        k_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))  # (N, N)
        w = k_inv - np.outer(alpha, alpha)  # ∂NLML/∂K = ½ W

        # カーネルパラメータの勾配 (log スケール、grad_at 側で chain rule 適用済み)
        kernel_grads = self._kernel.grad_at(k_input, log_params[:n_k])
        grad_k = np.array([0.5 * float(np.sum(w * dK)) for dK in kernel_grads.values()])

        # σ_n_2 の勾配: ∂K_noisy/∂σ_n_2 = I → ∂NLML/∂log(σ_n_2) = ½ tr(W) · σ_n_2
        grad_sn = 0.5 * float(np.trace(w)) * sigma_n_2

        return np.append(grad_k, grad_sn)
