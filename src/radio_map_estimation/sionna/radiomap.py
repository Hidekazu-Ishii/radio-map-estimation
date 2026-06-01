"""
Sionna RT シーンのロード・送信機配置・RadioMapSolver 実行

役割
----
1. scene.xml をロード
2. DEM オブジェクトを measurement_surface として設定
3. アンテナアレイ・送信機を配置
4. MeshRadioMap / PlanarRadioMap を計算して返す

設計方針
--------
- I/O なし (保存は visualize.py へ)
- dataclass は持たない (SionnaConfig はエントリポイント run.py に定義)
- Sionna RT のインポートは関数内で行う (Mitsuba variant の自動設定のため)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_radio_maps(
    xml_path: Path,
    tx_positions: list[tuple[float, float, float]],
    cfg,
    area_size_m: float,
):
    """
    Sionna RT シーンをロードし、MeshRadioMap と PlanarRadioMap を計算する

    Parameters
    ----------
    xml_path     : scene.xml のパス
    tx_positions : TX 位置のリスト (tx_placement.build_tx_positions の出力)
    cfg          : SionnaConfig (run.py で定義)
    area_size_m  : 対象エリアの一辺の長さ [m] (PlanarRadioMap の range に使用)

    Returns
    -------
    (mesh_radio_map, planar_radio_map) のタプル
    """
    from sionna.rt import (
        PlanarArray,
        RadioMapSolver,
        Transmitter,
        load_scene,
        transform_mesh,
    )

    # 1. シーンロード
    logger.info("Loading scene: %s", xml_path)
    scene = load_scene(str(xml_path))
    scene.frequency = cfg.frequency_hz
    logger.info("Scene loaded: %d objects", len(scene.objects))

    # 2. DEM オブジェクトを measurement_surface として使用 (z + 1.5m)
    if "dem" not in scene.objects:
        raise KeyError(f"Object 'dem' not found in scene. Available objects: {list(scene.objects.keys())}")
    measurement_surface = scene.objects["dem"].clone(as_mesh=True)
    transform_mesh(measurement_surface, translation=[0.0, 0.0, 1.5])  # type: ignore
    logger.info("Measurement surface: dem (z + 1.5m)")

    # 3. アンテナアレイ設定
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern=cfg.tx_pattern,
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern="iso",
        polarization="V",
    )

    # 4. 送信機配置
    for i, tx_pos in enumerate(tx_positions):
        tx = Transmitter(
            name=f"tx{i}",
            position=list(tx_pos),  # type: ignore
            power_dbm=cfg.tx_power_dbm,
        )
        if cfg.tx_pattern == "tr38901":
            tilt_rad = math.radians(cfg.tx_tilt_deg)
            d_horizontal = 500.0
            look_z = tx_pos[2] - d_horizontal * math.tan(tilt_rad)
            tx.look_at(np.array([tx_pos[0] + d_horizontal, tx_pos[1], look_z]))  # type: ignore
        scene.add(tx)

    logger.info(
        "Added %d transmitter(s), power=%.1f dBm, pattern=%s%s",
        len(tx_positions),
        cfg.tx_power_dbm,
        cfg.tx_pattern,
        f", tilt={cfg.tx_tilt_deg}deg" if cfg.tx_pattern == "tr38901" else "",
    )

    # 5. RadioMapSolver
    logger.info(
        "Computing radio maps: cell_size=%.1fm, max_depth=%d, samples=%d",
        cfg.cell_size_m,
        cfg.max_depth,
        cfg.num_samples,
    )
    rm_solver = RadioMapSolver()

    # 5a. MeshRadioMap: DEM メッシュ上 (3D 俯瞰レンダリング用)
    mesh_radio_map = rm_solver(
        scene,
        measurement_surface=measurement_surface,
        max_depth=cfg.max_depth,
        samples_per_tx=cfg.num_samples,
    )
    logger.info("MeshRadioMap computed.")

    # 5b. PlanarRadioMap: エリア [0, area_size_m] x [0, area_size_m] の水平面
    # シミュレーションはシーン全体 (マージン込み) で実行し、
    # 測定グリッドのみ対象エリアに限定する
    planar_radio_map = rm_solver(
        scene,
        max_depth=cfg.max_depth,
        samples_per_tx=cfg.num_samples,
        cell_size=[cfg.cell_size_m, cfg.cell_size_m],  # type: ignore
        center=[area_size_m / 2, area_size_m / 2, cfg.rx_height_m],  # type: ignore
        orientation=[0.0, 0.0, 0.0],  # type: ignore
        size=[area_size_m, area_size_m],  # type: ignore
    )
    logger.info("PlanarRadioMap computed.")

    return scene, mesh_radio_map, planar_radio_map
