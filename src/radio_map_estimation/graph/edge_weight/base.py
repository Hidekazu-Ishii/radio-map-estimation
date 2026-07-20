# ruff: noqa: F722
"""
エッジ重み関数の抽象基底クラス

設計方針
--------
- graph/ はグラフ構築 (エッジ重み・距離変換・ラプラシアン・スペクトル解析) にのみ
  責務を持ち, shadowing/ 配下のモジュール (GP最適化・grad_at・set_log_params など)
  には一切依存しない. shadowing/kernel/gudmundson.py の Kernel は GP フィットの
  ための責務を持つクラスであり, グラフ構築側が必要とするのは
  「座標 → 重み行列」という薄い写像だけなので, 別概念として切り離す
- エッジ重み関数は Kernel を継承・合成せず, この ABC を直接実装した
  自己完結クラスとして graph/edge_weight/ 以下に個別実装する
  (例: gudmundson.py の GudmundsonEdgeWeight)
- shadowing 側と数式が重複する場合 (Gudmundsonの距離減衰式など) でも,
  モジュール境界を優先してあえて複製する. 将来 k-NN グラフ・方向依存グラフ・
  異方性ヒューリスティック重み (w0..w3, Fresnel拡張) など shadowing/ に
  対応物がないエッジ重み関数も, 同じ ABC さえ実装すれば同じパイプライン
  (distance.py, variogram.py, spectral.py) に接続できる

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
    エッジ重み関数を扱う. 重みの意味 (距離減衰, 類似度スコアなど) は
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
