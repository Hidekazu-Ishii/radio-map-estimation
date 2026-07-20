# ruff: noqa: F722
"""
グラフラプラシアン構築・スペクトル分解・グラフフーリエ変換によるパワースペクトル診断

設計方針
--------
- W (エッジ重み) → L=D-W → 固有分解 → GFT → パワースペクトル、という
  頂点領域からスペクトル領域への一連の変換を担う
- 孤立ノード (LOS隣接ゼロ) は固有値0に局在する無意味な固有ベクトルを生むため、
  Laplacian構築前に必ず除去しておく (graph/los.py の filter_isolated_nodes と対で使う.
  除去そのものは GraphNodeSet.filter が担う)
- 次数計算は W の対角成分を無視する (対角込み/対角抜きのどちらで次数を定義しても
  L=D-W の値自体は数学的に一致するが、"対角を引き忘れる" ような非一貫な実装による
  バグを避けるため、最初から対角を0とみなす形に統一する)
- train / heldout はそれぞれ独立にGFTする (混ぜて1本のベクトルにしない、
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float, Int
from numpy import ndarray

_LaplacianMethod = Literal["unnormalized", "normalized"]


@dataclass(frozen=True, slots=True)
class GraphNodeSet:
    """スペクトル診断用のノード集合 (座標・残差・train/heldout区分を対応づけて保持)

    孤立ノード除去などのフィルタ操作で複数配列間の対応がずれることを構造的に防ぐ

    Attributes
    ----------
    coords     : 全ノードの座標 (x, y) [m], shape (N, 2)
    residuals  : シャドウイング残差 (真値), shape (N, 1)
    train_mask : ノードが train なら True, held-out なら False, shape (N,)
    """

    coords: Float[ndarray, "N 2"]
    residuals: Float[ndarray, "N 1"]
    train_mask: Bool[ndarray, "N 1"]

    def filter(self, keep_mask: Bool[ndarray, "N 1"]) -> GraphNodeSet:
        """keep_mask で3配列をまとめて絞り込んだ新しいインスタンスを返す"""
        return GraphNodeSet(
            coords=self.coords[keep_mask],
            residuals=self.residuals[keep_mask],
            train_mask=self.train_mask[keep_mask],
        )

    def __len__(self) -> int:
        return self.coords.shape[0]


@dataclass(frozen=True, slots=True)
class LaplacianConfig:
    """グラフラプラシアンの構築方式の設定値

    Attributes
    ----------
    method : "unnormalized" | "normalized"
        unnormalized : L = D - W. 絶対スケールを持つ (次数の大小、つまりWの
                       パラメータ設定や接続密度に応じて固有値の範囲が大きく変わる)
        normalized   : L_sym = I - D^(-1/2) W D^(-1/2). 固有値が必ず [0, 2] に
                       収まるため、Wの候補間・trial間でスケールに依存せず比較できる
    eps    : 次数が0に極めて近いノードに対する数値安定化用の下限値
             (孤立ノードは filter_isolated_nodes で事前に除去済みのはずだが、
             フェイルセーフとして設ける)
    """

    method: _LaplacianMethod
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.method not in ("unnormalized", "normalized"):
            raise ValueError(f"Unknown method: {self.method!r}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")


def build_graph_laplacian(
    w: Float[ndarray, "N N"],
    cfg: LaplacianConfig,
) -> Float[ndarray, "N N"]:
    """エッジ重み行列 W からグラフラプラシアンを構築する (cfg.method で方式を切替)

    次数 D_ii は W の対角成分 (自己重み) を無視して計算する

    Parameters
    ----------
    w   : エッジ重み行列 (孤立ノード除去・LOSマスク適用済みを想定), shape (N, N).
          対称, 非負
    cfg : ラプラシアンの構築方式の設定

    Returns
    -------
    laplacian : グラフラプラシアン, shape (N, N)
    """
    w_off_diag = w - np.diag(np.diag(w))  # 対角を無視 (次数計算に使わない)
    degree = w_off_diag.sum(axis=1)

    match cfg.method:
        case "unnormalized":
            return np.diag(degree) - w_off_diag
        case "normalized":
            d_inv_sqrt = 1.0 / np.sqrt(np.clip(degree, cfg.eps, None))
            scaled_w = d_inv_sqrt[:, None] * w_off_diag * d_inv_sqrt[None, :]
            return np.eye(w.shape[0]) - scaled_w
        case _:
            raise NotImplementedError(f"method={cfg.method!r} は未実装")


def eigendecompose_laplacian(
    laplacian: Float[ndarray, "N N"],
) -> tuple[Float[ndarray, "N 1"], Float[ndarray, "N N"]]:
    """グラフラプラシアンを固有分解する (L は対称なので eigh を使う)

    Returns
    -------
    eigvals : 固有値 λ (昇順), shape (N,)
    eigvecs : 対応する固有ベクトル Φ (列ベクトル), shape (N, N)
    """
    return np.linalg.eigh(laplacian)


@dataclass(frozen=True, slots=True)
class SpectralVarianceConfig:
    """スペクトル領域でのパワー分散診断のビン分割設定

    Attributes
    ----------
    n_bins     : 固有値 λ のビン数
    max_lambda : ビンの上限 λ. None なら最大固有値を使う
    """

    n_bins: int
    max_lambda: float | None = None

    def __post_init__(self) -> None:
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")
        if self.max_lambda is not None and self.max_lambda <= 0.0:
            raise ValueError(f"max_lambda must be positive, got {self.max_lambda}")


@dataclass(frozen=True, slots=True)
class SpectralVarianceResult:
    """1群分のスペクトルパワー診断結果

    Attributes
    ----------
    group       : 群名 ("train" | "heldout")
    bin_centers : 各ビンの中心固有値, shape (B, 1)
    mean_power  : 各ビンの平均パワー, shape (B, 1). モードが0件のビンはnan
    std_power   : 各ビンのパワー標準偏差, shape (B, 1). モードが0/1件のビンはnan
    counts      : 各ビンに属するモード数, shape (B, 1)
    """

    group: str
    bin_centers: Float[ndarray, "B 1"]
    mean_power: Float[ndarray, "B 1"]
    std_power: Float[ndarray, "B 1"]
    counts: Int[ndarray, "B 1"]


def compute_graph_fourier_transform(
    eigvecs: Float[ndarray, "N N"],
    signal: Float[ndarray, "N 1"],
) -> Float[ndarray, "N 1"]:
    """グラフフーリエ変換 f_hat = Phi^T f を計算する"""
    return eigvecs.T @ signal


def _bin_power_spectrum(
    eigvals: Float[ndarray, "N 1"],
    power: Float[ndarray, "N 1"],
    cfg: SpectralVarianceConfig,
    group: str,
) -> SpectralVarianceResult:
    """1群分のモードを λ ビンに割り当て、ビンごとの平均・標準偏差パワーを計算する"""
    max_lambda = cfg.max_lambda if cfg.max_lambda is not None else float(eigvals.max())
    bin_edges = np.linspace(0.0, max_lambda, cfg.n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    bin_indices = np.clip(np.digitize(eigvals, bin_edges) - 1, 0, cfg.n_bins - 1)

    mean_power = np.full(cfg.n_bins, np.nan)
    std_power = np.full(cfg.n_bins, np.nan)
    counts = np.zeros(cfg.n_bins, dtype=np.int64)
    for b in range(cfg.n_bins):
        in_bin = bin_indices == b
        counts[b] = int(np.count_nonzero(in_bin))
        if counts[b] > 0:
            mean_power[b] = float(np.mean(power[in_bin]))
        if counts[b] > 1:
            std_power[b] = float(np.std(power[in_bin]))

    return SpectralVarianceResult(
        group=group, bin_centers=bin_centers, mean_power=mean_power, std_power=std_power, counts=counts
    )


def compute_grouped_spectral_variance(
    eigvals: Float[ndarray, "N 1"],
    eigvecs: Float[ndarray, "N N"],
    node_set: GraphNodeSet,
    cfg: SpectralVarianceConfig,
) -> dict[str, SpectralVarianceResult]:
    """train / heldout それぞれ独立にGFTし、λビンごとのパワー統計を計算する

    train, heldout を混ぜて1本のベクトルにせず、群ごとに「対象外ノードは0埋め」した
    信号として別々に変換する (train が見ていない領域まで W の設計が汎化しているかを
    比較するため)

    Parameters
    ----------
    eigvals  : グラフラプラシアンの固有値, shape (N,)
    eigvecs  : 対応する固有ベクトル, shape (N, N)
    node_set : 孤立ノード除去済みのノード集合 (train_mask で群分け)
    cfg      : ビン分割設定

    Returns
    -------
    {"train": SpectralVarianceResult, "heldout": SpectralVarianceResult}
    """
    results: dict[str, SpectralVarianceResult] = {}
    for group, mask in (("train", node_set.train_mask), ("heldout", ~node_set.train_mask)):
        signal = np.where(mask, node_set.residuals[:, 0], 0.0).reshape(-1, 1)
        f_hat = compute_graph_fourier_transform(eigvecs, signal)
        power = f_hat[:, 0] ** 2
        results[group] = _bin_power_spectrum(eigvals, power, cfg, group)
    return results
