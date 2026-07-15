# ruff: noqa: F722
"""
エッジ重み関数の抽象基底クラス

設計方針
--------
- グラフ距離バリオグラム診断 (Stage 1) では「フィット済みパラメータを
  全ノードに適用して重み行列を作る」ことだけが責務であり,
  shadowing/kernels/base.py の Kernel が持つ最適化関連の責務
  (grad_at, set_log_params, param_bounds など) は不要
- そのため Kernel を継承・拡張するのではなく, 薄い専用インタフェースを新設する
- 既存の Kernel を使うエッジ重み関数は KernelEdgeWeight アダプタで包む
  (kernel_adapter.py). 将来 k-NN グラフや方向依存グラフなど,
  Kernel の枠に収まらないエッジ重み関数もこの ABC を実装すれば
  同じパイプライン (distance.py, variogram.py) に接続できる

サブクラスの実装義務
--------------------
compute_weights : 全ノード座標 → 重み行列 W (N, N) (対称, 対角は自己重み)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from jaxtyping import Float
from numpy import ndarray


class EdgeWeightFunction(ABC):
    """グラフのエッジ重み行列を計算する抽象基底クラス

    グラフ構築・距離変換 (distance.py) はこのインタフェースのみを通じて
    エッジ重み関数を扱う. 重みの意味 (カーネル値, 類似度スコアなど) は
    サブクラスの内部に隠蔽される.
    """

    @abstractmethod
    def compute_weights(
        self,
        coords: Float[ndarray, "N 2"],
        **kwargs,
    ) -> Float[ndarray, "N N"]:
        """全ノード座標からエッジ重み行列 W を計算する

        Parameters
        ----------
        coords   : 全ノード (train + held-out) の座標 (x, y) [m], shape (N, 2)
        **kwargs : エッジ重み関数固有の追加引数 (TX 座標など)

        Returns
        -------
        W : 重み行列, shape (N, N). 対称, 対角成分は自己重み (距離変換で正規化に使う)
        """
        ...
