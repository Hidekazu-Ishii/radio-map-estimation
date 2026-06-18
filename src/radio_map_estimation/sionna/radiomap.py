"""
Sionna RT シーンのロード・送信機配置・RadioMapSolver 実行

役割
----
1. scene.xml をロード (1回のみ)
2. DEM オブジェクトを measurement_surface として設定
3. アンテナアレイ・送信機を配置
4. 周波数リストをループして scene.frequency を更新しながら
   MeshRadioMap / PlanarRadioMap を計算して返す

設計方針
--------
- I/O なし (保存は visualize.py へ)
- dataclass は持たない (SionnaConfig はエントリポイント simulate.py に定義)
- Sionna RT のインポートは関数内で行う (Mitsuba variant の自動設定のため)
- シーンロードは1回のみ。周波数ごとに scene.frequency を差し替えて再計算する
   (load_scene はコストが高いため、TX 配置等はロード直後に1回だけ行う)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _load_scene(
    xml_path: Path,
    tx_positions: list[tuple[float, float, float]],
    cfg,
):
    """
    シーンを1回ロードし、アンテナ・TX を配置して返す。
    周波数は呼び出し元でループしながら scene.frequency に設定する。

    Parameters
    ----------
    xml_path     : scene.xml のパス
    tx_positions : TX 位置のリスト
    cfg          : SionnaConfig

    Returns
    -------
    (scene, measurement_surface) のタプル
    """
    from sionna.rt import (
        PlanarArray,
        Transmitter,
        load_scene,
        transform_mesh,
    )

    # 1. シーンロード (周波数は後でループ内で設定するため未設定)
    logger.info("Loading scene: %s", xml_path)
    scene = load_scene(str(xml_path))
    logger.info("Scene loaded: %d objects", len(scene.objects))

    # 2. DEM オブジェクトを measurement_surface として使用 (z + rx_height_m)
    if "dem" not in scene.objects:
        raise KeyError(f"Object 'dem' not found in scene. Available objects: {list(scene.objects.keys())}")
    measurement_surface = scene.objects["dem"].clone(as_mesh=True)
    transform_mesh(measurement_surface, translation=[0.0, 0.0, cfg.rx_height_m])  # type: ignore
    logger.info("Measurement surface: dem (z + %.1fm)", cfg.rx_height_m)

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

    return scene, measurement_surface


def build_radio_maps(
    xml_path: Path,
    tx_positions: list[tuple[float, float, float]],
    cfg,
    area_size_m: float,
) -> list[tuple[float, object, object, object]]:
    """
    Sionna RT シーンをロードし、各周波数の MeshRadioMap と PlanarRadioMap を計算する。

    シーンロードと TX 配置は1回のみ行い、cfg.frequency_hz のリストをループして
    scene.frequency を更新しながらシミュレーションを実行する。

    Parameters
    ----------
    xml_path     : scene.xml のパス
    tx_positions : TX 位置のリスト (tx_placement.build_tx_positions の出力)
    cfg          : SionnaConfig (frequency_hz: tuple[float, ...])
    area_size_m  : 対象エリアの一辺の長さ [m]

    Returns
    -------
    list of (freq_hz, scene, mesh_radio_map, planar_radio_map)
        周波数ごとの結果リスト。scene は全周波数で共通 (frequency が更新済み)
    """
    from sionna.rt import RadioMapSolver

    # シーンロード・TX 配置は1回のみ
    scene, measurement_surface = _load_scene(xml_path, tx_positions, cfg)

    rm_solver = RadioMapSolver()
    results: list[tuple[float, object, object, object]] = []

    for freq_hz in cfg.frequency_hz:
        logger.info(
            "Computing radio maps: freq=%.4gGHz, cell_size=%.1fm, max_depth=%d, samples=%d",
            freq_hz / 1e9,
            cfg.cell_size_m,
            cfg.max_depth,
            cfg.num_samples,
        )

        # 周波数のみ差し替え (シーン・TX 配置は再利用)
        scene.frequency = freq_hz

        # MeshRadioMap: DEM メッシュ上 (3D 俯瞰レンダリング用)
        mesh_radio_map = rm_solver(
            scene,
            measurement_surface=measurement_surface,
            max_depth=cfg.max_depth,
            samples_per_tx=cfg.num_samples,
            diffraction=True,
        )
        logger.info("MeshRadioMap computed (%.4gGHz).", freq_hz / 1e9)

        # PlanarRadioMap: エリア [0, area_size_m] x [0, area_size_m] の水平面
        planar_radio_map = rm_solver(
            scene,
            max_depth=cfg.max_depth,
            samples_per_tx=cfg.num_samples,
            cell_size=[cfg.cell_size_m, cfg.cell_size_m],  # type: ignore
            center=[area_size_m / 2, area_size_m / 2, cfg.rx_height_m],  # type: ignore
            orientation=[0.0, 0.0, 0.0],  # type: ignore
            size=[area_size_m, area_size_m],  # type: ignore
            diffraction=True,
        )
        logger.info("PlanarRadioMap computed (%.4gGHz).", freq_hz / 1e9)

        results.append((freq_hz, scene, mesh_radio_map, planar_radio_map))

    return results
