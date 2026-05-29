# src/radio_map_estimation/scene/radiomap_solver.py
"""
Mitsuba XML シーン → Sionna RT → RSS マップ [dBm]

Sionna / Mitsuba はスタブ (.pyi) を提供しないため関数内でインポートし, 位置・方向引数はすべて list[float] で渡す.
Sionna 内部で対応する mi 型に自動変換される.
"""

from pathlib import Path
from typing import Any

import numpy as np

from ..osm.osm_schema import BuildingData

# Sionna が型スタブを提供しないため Any で受ける
PlanarRadioMap = Any

Vec3 = tuple[float, float, float]


def run_radio_map(
    scene_xml: Path,
    tx_positions: list[Vec3],
    frequency_hz: float,
    tx_power_dbm: float,
    building_data: BuildingData,
    rx_height_m: float,
    max_depth: int,
    cell_size_m: float,
    samples_per_tx: int,
    seed: int,
) -> PlanarRadioMap:
    """
    Sionna RT で RSS マップ [dBm] を計算して返す.

    Parameters
    ----------
    scene_xml : Path
    tx_position_local : Vec3
        ローカル座標系 (bbox 左下を原点) での TX 位置 (x, y, z) [m].
    frequency_hz : float
    tx_power_dbm : float
    building_data : BuildingData
        building_mask と AreaSpec を参照する.
    rx_height_m : float
        受信面の高さ [m].
    max_depth : int
        最大反射回数.
    cell_size_m : float
        RadioMapSolver のセルサイズ [m].
    samples_per_tx : int
        TX あたりのレイサンプル数.

    Returns
    -------
    radio_map : PlanarRadioMap (sionna.rt.PlanarRadioMap)
        .rss         : shape [num_tx, H, W] [W]
        .path_gain   : shape [num_tx, H, W]
        .sinr        : shape [num_tx, H, W]
        .show(tx=None): best-server マップ
        .show(tx=i)  : TX i のマップ
    """
    import mitsuba as mi
    from sionna.rt import (
        PlanarArray,
        RadioMapSolver,
        Transmitter,
        load_scene,
    )

    spec = building_data.area_spec
    half: float = spec.area_size_m / 2.0

    # remove_duplicate_vertices=False: 隣接建物間の頂点 merge を防ぐ
    scene = load_scene(str(scene_xml), remove_duplicate_vertices=False)
    scene.frequency = frequency_hz

    iso_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.tx_array = iso_array
    scene.rx_array = iso_array

    # 全 TX を同一 scene に登録
    for i, pos in enumerate(tx_positions):
        tx = Transmitter(
            name=f"tx{i}",
            position=mi.Point3f([float(pos[0]), float(pos[1]), float(pos[2])]),
            orientation=mi.Point3f([0.0, 0.0, 0.0]),
            power_dbm=int(tx_power_dbm),
        )
        scene.add(tx)

    rm_solver = RadioMapSolver()
    radio_map = rm_solver(
        scene=scene,
        max_depth=max_depth,
        cell_size=mi.Point2f([cell_size_m, cell_size_m]),
        samples_per_tx=samples_per_tx,
        center=mi.Point3f([half, half, rx_height_m]),
        orientation=mi.Point3f([0.0, 0.0, 0.0]),
        size=mi.Point2f([spec.area_size_m, spec.area_size_m]),
        diffraction=True,
        seed=seed,
    )
    return radio_map


def radio_map_to_rss_dbm(
    radio_map: PlanarRadioMap,
) -> np.ndarray:
    """
    PlanarRadioMap → RSS [dBm] の numpy 配列に変換する.

    レイ未到達セル (rss_w == 0.0) は -120 dBm (1e-15 W) に置換する.

    Parameters
    ----------
    radio_map : PlanarRadioMap

    Returns
    -------
    rss_dbm : np.ndarray of shape (H, W)
        RSS [dBm]. nan は存在せず, 未到達セルは -120 dBm.
    """

    rss_w: np.ndarray = radio_map.rss.numpy()  # (num_tx, H, W) [W]
    rss_w_safe: np.ndarray = np.where(rss_w > 0.0, rss_w, 1e-15)
    rss_dbm: np.ndarray = 10.0 * np.log10(rss_w_safe) + 30.0
    return rss_dbm  # (num_tx, H, W)
