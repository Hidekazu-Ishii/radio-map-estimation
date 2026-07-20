# ruff: noqa: F722
"""
Gudmundson 型の距離減衰をエッジ重み関数として使う (初期値固定, 自己完結)

設計方針
--------
- graph/ は shadowing/ 配下のモジュールに一切依存しない (責務分離のため).
  shadowing/kernel/gudmundson.py の GudmundsonKernel と数式は同じだが,
  あちらは GP フィット (grad_at, set_log_params など最適化関連の責務) を
  持つ別概念のクラスであり, ここでは意図的に式のみを複製して自己完結させる
- 重み関数 W → グラフラプラシアン L → スペクトルフィルタ h(Λ) を同時最適化する
  本番仕様の NLML 最適化ロジックはまだ実装できていない. そのため fit 済み
  パラメータを外部から注入する経路は持たず, コンストラクタで渡した
  sigma_2_init / d_cor_init をそのまま重み計算に使う
  ("エッジ重み関数のパラメータは最適化せず, ヒューリスティックな初期値を
  設定して本番の代用とする" というタスク方針に対応する)

使用例
------
    >>> edge_weight_fn = GudmundsonEdgeWeight(sigma_2_init=50.0, d_cor_init=20.0)
    >>> W = edge_weight_fn.compute_weights(coords_all)  # train + held-out 全ノードに適用
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from numpy import ndarray

from .base import EdgeWeightFunction


@dataclass(frozen=True, slots=True)
class GudmundsonEdgeWeight(EdgeWeightFunction):
    """初期値固定の Gudmundson 型距離減衰をエッジ重み関数として扱う自己完結クラス

    Attributes
    ----------
    sigma_2_init : シャドウイング分散 σ_2 [dB²] のヒューリスティックな初期値
    d_cor_init   : 相関距離 d_cor [m] のヒューリスティックな初期値
    """

    sigma_2_init: float
    d_cor_init: float

    def __post_init__(self) -> None:
        if self.sigma_2_init <= 0.0:
            raise ValueError(f"sigma_2_init must be positive, got {self.sigma_2_init}")
        if self.d_cor_init <= 0.0:
            raise ValueError(f"d_cor_init must be positive, got {self.d_cor_init}")

    def compute_weights(
        self,
        coords: Float[ndarray, "N 2"],
        **kwargs,  # TX 座標など — このエッジ重み関数では不使用
    ) -> Float[ndarray, "N N"]:
        """全ノード座標に Gudmundson 型距離減衰を適用して重み行列を返す"""
        delta = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
        dist = np.sqrt(np.sum(delta**2, axis=-1))  # (N, N)
        return self.sigma_2_init * np.exp(-dist * np.log(2.0) / self.d_cor_init)
