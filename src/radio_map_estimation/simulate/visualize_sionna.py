# src/radio_map_estimation/simulate/visualize_sionna.py
"""
電波マップ・建物データの可視化 (公式準拠, numpy 配列から復元できないため, 利用はしない)
"""

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..scene.osm_buildings import BuildingData

# Sionna が型スタブを提供しないため Any で受ける
# PlanarRadioMap の実体は sionna.rt.PlanarRadioMap
PlanarRadioMap = Any

RadioMapMetric = Literal["path_gain", "rss", "sinr"]


def show_building_data(data: BuildingData) -> Figure:
    """BuildingData (マスク + 高さ) を2パネルで可視化する."""
    spec = data.area_spec
    s = spec.area_size_m
    grid_size = spec.grid_size

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- パネル左: 建物マスク ---
    ax = axes[0]
    rgb = np.ones((grid_size, grid_size, 3))
    rgb[data.building_mask] = [0.2, 0.2, 0.6]
    ax.imshow(rgb, origin="lower", extent=(0, s, 0, s))
    ax.set_title(
        f"OSM Building Map\n"
        f"center=({spec.center_lat:.4f}, {spec.center_lon:.4f}), "
        f"{s:.0f}m x {s:.0f}m, cell={spec.cell_size_m:.1f}m"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # --- パネル右: 建物高さ ---
    ax = axes[1]
    im = ax.imshow(data.building_heights, origin="lower", extent=(0, s, 0, s), cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Building height [m]")
    ax.set_title("Building Heights")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    fig.tight_layout()
    return fig


def show_radio_map(
    radio_map: PlanarRadioMap,
    metric: RadioMapMetric = "rss",
    tx: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_tx: bool = True,
    show_rx: bool = False,
) -> Figure:
    """電波マップを可視化する.

    Parameters
    ----------
    radio_map : PlanarRadioMap (sionna.rt.PlanarRadioMap)
        Sionna RT の RadioMapSolver が返すオブジェクト (全 TX 分).
    metric : {"path_gain", "rss", "sinr"}
        表示する指標.
    tx : int | None
        TX インデックス. None のとき全 TX の最大値 (best-server) を表示する.
    vmin, vmax : float | None
        カラーバーの範囲 [dB / dBm]. None のとき自動スケール.
    show_tx : bool
        TX 位置を "+" でプロットするか.
    show_rx : bool
        RX 位置を "x" でプロットするか.
    """
    fig: Figure = radio_map.show(
        metric=metric,
        tx=tx,
        vmin=vmin,
        vmax=vmax,
        show_tx=show_tx,
        show_rx=show_rx,
    )
    return fig


def show_radio_map_cdf(
    radio_map: PlanarRadioMap,
    metric: RadioMapMetric = "rss",
    tx: int | None = None,
    bins: int = 200,
) -> Figure:
    """電波マップ指標の CDF を可視化する.

    tx=None のとき best-server (全 TX 最大値) の CDF を返す.
    """
    fig, *_ = radio_map.cdf(metric=metric, tx=tx, bins=bins)
    return fig


def show_tx_association(
    radio_map: PlanarRadioMap,
    metric: RadioMapMetric = "rss",
    show_tx: bool = True,
    show_rx: bool = False,
) -> Figure:
    """セルと TX の対応を可視化する."""
    fig: Figure = radio_map.show_association(
        metric=metric,
        show_tx=show_tx,
        show_rx=show_rx,
    )
    return fig


def save_figure(fig: Figure, save_path: Path, dpi: int = 150) -> None:
    """Figure をファイルに保存してクローズする.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    save_path : Path
    dpi : int
        解像度 (デフォルト 150).
    """
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {save_path}")
