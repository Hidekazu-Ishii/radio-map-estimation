# ruff: noqa: F722
"""
グラフ距離に対する経験的セミバリオグラムの計算 (train/held-out 群分け)

設計方針
--------
- ノード集合は「train + held-out」のみで構成する (test_prod は呼び出し元で
  最初から座標・残差に含めないこと. ここでは train_mask による群分けのみを行い,
  test を除外する責務は負わない = 呼び出し元の責務として明確に分離する)
- 群分けは 2 群のみ:
    train_train      : 両端が train ノード
    heldout_heldout   : 両端が held-out ノード
  train-held-out (片方が train, 片方が held-out) のペアはどちらの群にも
  含めない. train の影響を一切受けていない heldout_heldout 群だけを
  train_train 群と直接比較することで, 「train が見ていない領域まで
  エッジ重み関数が汎化しているか」を最も厳密に検証できる
- 対角成分 (自己ペア, 距離 0) は常に除外する
- 対称行列の上三角のみを使い, ペアの二重カウントを避ける
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int
from numpy import ndarray


@dataclass(frozen=True, slots=True)
class VariogramConfig:
    """セミバリオグラムのビン分割設定

    Attributes
    ----------
    n_bins       : 距離ビンの数
    max_distance : ビンの上限距離. None ならグループ内の最大グラフ距離を使う
    """

    n_bins: int
    max_distance: float | None = None

    def __post_init__(self) -> None:
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")
        if self.max_distance is not None and self.max_distance <= 0.0:
            raise ValueError(f"max_distance must be positive, got {self.max_distance}")


@dataclass(frozen=True, slots=True)
class VariogramResult:
    """1 群分の経験的セミバリオグラム計算結果

    Attributes
    ----------
    group       : 群名 ("train_train" | "involves_heldout")
    bin_centers : 各ビンの中心距離, shape (B,)
    gamma       : 各ビンの経験的セミバリオグラム値, shape (B,). ペアが 0 件のビンは nan
    counts      : 各ビンに属するペア数, shape (B,)
    """

    group: str
    bin_centers: Float[ndarray, "B 1"]
    gamma: Float[ndarray, "B 1"]
    counts: Int[ndarray, "B 1"]


def compute_pairwise_semivariance(
    residuals: Float[ndarray, "N 1"],
) -> Float[ndarray, "N N"]:
    """全ノード対の半二乗差 0.5*(f_i - f_j)^2 を計算する"""
    diff = residuals - residuals.T  # (N, N)
    return 0.5 * diff**2


def classify_pairs(
    train_mask: Bool[ndarray, "N 1"],
) -> dict[str, Bool[ndarray, "N N"]]:
    """ノード対を train_train / heldout_heldout の 2 群に分けるマスクを作る

    train-held-out (片方が train, 片方が held-out) のペアはどちらの群にも
    含めない (train の影響を受けたペアを比較から除外し, heldout_heldout 群を
    「train を一切通らない純粋な held-out 同士」に保つため).
    対角成分 (自己ペア) と下三角 (二重カウント) は両群とも False にする.

    Parameters
    ----------
    train_mask : ノードが train なら True, held-out なら False, shape (N,)

    Returns
    -------
    {"train_train": ..., "heldout_heldout": ...} のブールマスク辞書, 各 shape (N, N)
    """
    n = train_mask.shape[0]
    upper_triangle = np.triu(np.ones((n, n), dtype=bool), k=1)  # 対角除外・上三角のみ

    train_train_all = np.outer(train_mask, train_mask)  # 両端 train
    heldout_heldout_all = np.outer(~train_mask, ~train_mask)  # 両端 held-out
    return {
        "train_train": train_train_all & upper_triangle,
        "heldout_heldout": heldout_heldout_all & upper_triangle,
    }


def _bin_variogram(
    graph_distance: Float[ndarray, "N N"],
    semivariance: Float[ndarray, "N N"],
    pair_mask: Bool[ndarray, "N N"],
    cfg: VariogramConfig,
    group: str,
) -> VariogramResult:
    """1 群分のペアを距離ビンに割り当て, ビンごとの平均セミバリアンスを計算する"""
    d = graph_distance[pair_mask]  # (K,)
    gamma_pairs = semivariance[pair_mask]  # (K,)

    max_distance = cfg.max_distance if cfg.max_distance is not None else float(d.max())
    bin_edges = np.linspace(0.0, max_distance, cfg.n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    bin_indices = np.clip(np.digitize(d, bin_edges) - 1, 0, cfg.n_bins - 1)

    gamma = np.full(cfg.n_bins, np.nan)
    counts = np.zeros(cfg.n_bins, dtype=np.int64)
    for b in range(cfg.n_bins):
        in_bin = bin_indices == b
        counts[b] = int(np.count_nonzero(in_bin))
        if counts[b] > 0:
            gamma[b] = float(np.mean(gamma_pairs[in_bin]))

    return VariogramResult(group=group, bin_centers=bin_centers, gamma=gamma, counts=counts)


def compute_grouped_variogram(
    graph_distance: Float[ndarray, "N N"],
    residuals: Float[ndarray, "N 1"],
    train_mask: Bool[ndarray, "N 1"],
    cfg: VariogramConfig,
) -> dict[str, VariogramResult]:
    """train_train / heldout_heldout の 2 群それぞれで経験的セミバリオグラムを計算する

    train-held-out ペアは比較対象から除外される (classify_pairs を参照).

    Parameters
    ----------
    graph_distance : グラフ最短経路距離行列 (distance.compute_graph_shortest_path の出力), shape (N, N)
    residuals      : シャドウイング残差 (真値), shape (N, 1)
    train_mask     : ノードが train なら True, held-out なら False, shape (N,)
    cfg            : ビン分割設定

    Returns
    -------
    {"train_train": VariogramResult, "heldout_heldout": VariogramResult}
    """
    semivariance = compute_pairwise_semivariance(residuals)
    pair_masks = classify_pairs(train_mask)
    return {
        group: _bin_variogram(graph_distance, semivariance, mask, cfg, group)
        for group, mask in pair_masks.items()
    }
