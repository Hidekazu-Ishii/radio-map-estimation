# ruff: noqa: F722
"""汎用可視化モジュール"""

from __future__ import annotations

from pathlib import Path

import matplotlib

from ..utils.grid_transform import point_to_cell_index

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from numpy import ndarray


def scatter_to_grid(
    coords: Float[ndarray, "N 2"],
    values: Float[ndarray, "N 1"],
    area_size_m: float,
    cell_size_m: float,
) -> Float[ndarray, "H W"]:
    """疎なグリッド (N, 2) を密グリッド (H, W) に変換する

    未観測セルは nan で埋める

    Parameters
    ----------
    coords      : (N, 2) 連続 or グリッド点座標 [m]
    values      : (N, 1) 各観測点の値 [dBm]
    area_size_m : エリアサイズ [m]
    cell_size_m : グリッドセルサイズ [m]

    Returns
    -------
    grid : (H, W) ndarray。未観測セルは nan
    """
    n = int(area_size_m / cell_size_m)
    grid = np.full((n, n), np.nan)

    cell_indices = point_to_cell_index(coords, cell_size_m)  # (N, 2) floor ベースの包含判定
    row_idx = np.minimum(cell_indices[:, 0], n - 1)
    col_idx = np.minimum(cell_indices[:, 1], n - 1)

    grid[row_idx, col_idx] = values[:, 0]
    return grid


def save_rss_png(
    tx_coords: np.ndarray,
    area_size_m: float,
    output_path: Path,
    title: str,
    values_db: np.ndarray | None = None,
    bldg_mask: np.ndarray | None = None,
    vmin: float = -120.0,
    vmax: float = -60.0,
) -> None:
    """RSS [dBm] グリッド (H, W) を imshow で保存する

    疎データを渡す場合は事前に scatter_to_grid で変換すること
    bldg_mask が与えられた場合、True のセルを灰色でオーバーレイする

    Parameters
    ----------
    values_db      : (H, W) ndarray, 未観測セルは nan
    tx_coords    : (num_tx, 3) TX 座標 [m] (重複なし)
    area_size_m  : エリアサイズ [m] (軸ラベル用)
    output_path  : 保存先
    title        : プロットタイトル
    bldg_mask : (H, W) bool, True のセルを灰色オーバーレイ
    vmin, vmax   : カラーバー範囲 [dBm]
    """
    extent = [0, area_size_m, 0, area_size_m]

    fig, ax = plt.subplots(figsize=(7, 6))

    if values_db is not None:
        im = ax.imshow(
            values_db,
            origin="lower",
            extent=extent,  # type: ignore[arg-type]
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="RSS [dBm]")

    # 建物セルを灰色でオーバーレイ
    if bldg_mask is not None:
        bldg_rgba = np.zeros((*bldg_mask.shape, 4), dtype=float)
        bldg_rgba[bldg_mask] = [0.5, 0.5, 0.5, 1.0]  # gray, alpha=1.0
        ax.imshow(
            bldg_rgba,
            origin="lower",
            extent=extent,  # type: ignore[arg-type]
            interpolation="none",
        )

    # TX 位置をプロット
    for tx_coord in tx_coords:
        ax.plot(tx_coord[0], tx_coord[1], "r+", markersize=10, markeredgewidth=2)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
