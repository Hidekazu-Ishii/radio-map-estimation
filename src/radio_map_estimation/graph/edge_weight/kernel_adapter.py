# ruff: noqa: F722
"""
既存の Kernel (shadowing/kernels/base.py) をエッジ重み関数として使うアダプタ

設計方針
--------
- train のみでフィット済みの Kernel (例: GudmundsonKernel) を
  全ノード座標に適用して重み行列を作るだけの薄いラッパー
- フィット (NLML 最適化) は既存の GPShadowingModel.fit() に任せる.
  ここでは fitted_log_params を固定値として受け取るのみで,
  最適化ロジックを一切持たない
- 合成: Kernel オブジェクトをコンストラクタで受け取り, 内部に保持する

使用例
------
    >>> kernel = GudmundsonKernel(sigma_2_init=50.0, d_cor_init=50.0)
    >>> fit_result = gp_model.fit(rng=rng, coords=train_coords, ...)  # train のみでフィット
    >>> fitted_log_params = np.log(np.array([fit_result.params["sigma_2"], fit_result.params["d_cor"]]))
    >>> edge_weight_fn = KernelEdgeWeight(kernel=kernel, fitted_log_params=fitted_log_params)
    >>> W = edge_weight_fn.compute_weights(coords_all)  # train + held-out 全ノードに適用
"""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float
from numpy import ndarray

from radio_map_estimation.shadowing.kernel.base import Kernel

from .base import EdgeWeightFunction


@dataclass(frozen=True, slots=True)
class KernelEdgeWeight(EdgeWeightFunction):
    """フィット済み Kernel をエッジ重み関数として扱うアダプタ

    Attributes
    ----------
    kernel            : フィット済みの Kernel (make_input / eval のみ使用)
    fitted_log_params : train のみで推定された log スケールのハイパーパラメータ
                         (Kernel.eval にそのまま渡す固定値. ここでは最適化しない)
    """

    kernel: Kernel
    fitted_log_params: Float[ndarray, "P 1"]

    def compute_weights(
        self,
        coords: Float[ndarray, "N 2"],
        **kwargs,
    ) -> Float[ndarray, "N N"]:
        """全ノード座標に fitted_log_params 固定のカーネルを適用して重み行列を返す"""
        kernel_input = self.kernel.make_input(coords, coords, **kwargs)
        return self.kernel.eval(kernel_input, self.fitted_log_params)
