# src/radio_map_estimation/simulate/visualize_selfmade.py
"""
電波マップ・建物データの可視化 (自作)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from ..scene.datasource.osm_buildings import BuildingData


def _make_radio_map_cmap() -> Colormap:
    """公式 rm.show() に準拠した viridis ベースのカラーマップを返す."""
    cmap = plt.get_cmap("viridis").copy()
    return cmap


def _apply_building_overlay(ax: Axes, building_mask: np.ndarray, extent: tuple) -> None:
    """建物セルを薄いグレーでオーバーレイする."""
    overlay: np.ndarray = np.zeros((*building_mask.shape, 4), dtype=np.float32)
    overlay[building_mask] = [0.8, 0.8, 0.8, 1.0]
    ax.imshow(overlay, origin="lower", extent=extent, zorder=2)


def _scatter_tx(ax: Axes, tx_positions: np.ndarray) -> None:
    """TX 位置をプロットする."""
    ax.scatter(
        tx_positions[:, 0],
        tx_positions[:, 1],
        c="red",
        s=120,
        marker="*",
        zorder=3,
        linewidths=1.5,
        label=f"TX (n={len(tx_positions)})",
    )


def plot_building_data(data: BuildingData, save_path: Path) -> None:
    """BuildingData (マスク + 高さ) を2パネルで可視化・保存する."""
    spec = data.area_spec
    s = spec.area_size_m
    grid_size = spec.grid_size

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- パネル左: 建物マスク ---
    ax = axes[0]
    img = np.ones((grid_size, grid_size, 3))
    img[data.building_mask] = [0.2, 0.2, 0.6]
    ax.imshow(img, origin="lower", extent=(0, s, 0, s))
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

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] saved → {save_path}")


def plot_radio_map(
    rss_dbm: np.ndarray,
    building_mask: np.ndarray,
    area_size_m: float,
    tx_positions: np.ndarray,
    frequency_hz: float,
    save_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """RSS マップ [dBm] と建物マスク・TX位置を重ねて可視化・保存する."""
    _, ax = plt.subplots(figsize=(7, 6))
    extent = (0.0, area_size_m, 0.0, area_size_m)

    rss_masked = np.ma.masked_invalid(rss_dbm)
    _vmin = vmin if vmin is not None else float(np.nanmin(rss_dbm))
    _vmax = vmax if vmax is not None else float(np.nanmax(rss_dbm))

    cmap = _make_radio_map_cmap()

    im = ax.imshow(
        rss_masked,
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="bilinear",
        zorder=1,
        vmin=_vmin,
        vmax=_vmax,
    )
    plt.colorbar(im, ax=ax, label="RSS [dBm]")

    _apply_building_overlay(ax, building_mask, extent)
    _scatter_tx(ax, tx_positions)
    ax.legend(loc="upper right", fontsize=8)

    freq_ghz = frequency_hz / 1e9
    ax.set_title(
        f"Radio Map (Best-Server RSS)\nf={freq_ghz:.2f} GHz, {area_size_m:.0f}m x {area_size_m:.0f}m"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] saved → {save_path}")


def plot_tx_association(
    rss_dbm_per_tx: np.ndarray,
    building_mask: np.ndarray,
    area_size_m: float,
    tx_positions: np.ndarray,
    save_path: Path,
) -> None:
    """各セルが RSS を提供する TX のインデックスを可視化・保存する."""
    num_tx = rss_dbm_per_tx.shape[0]
    extent = (0.0, area_size_m, 0.0, area_size_m)

    # best-server TX インデックス (H, W): 0 ~ num_tx-1
    assoc: np.ndarray = np.argmax(rss_dbm_per_tx, axis=0)

    # TX ごとに異なる色を割り当て
    cmap_assoc = plt.get_cmap("cool", num_tx).copy()

    _, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        assoc,
        origin="lower",
        extent=extent,
        cmap=cmap_assoc,
        vmin=-0.5,
        vmax=num_tx - 0.5,
        zorder=1,
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(num_tx))
    cbar.set_label("TX index")

    _apply_building_overlay(ax, building_mask, extent)
    _scatter_tx(ax, tx_positions)
    ax.legend(loc="upper right", fontsize=8)

    ax.set_title(f"TX Association (Best-Server)\n{num_tx} TX, {area_size_m:.0f}m x {area_size_m:.0f}m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] saved → {save_path}")
