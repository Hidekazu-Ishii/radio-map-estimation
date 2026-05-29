"""
rss_map.py
----------
Sionna RT を使って RSS マップを計算し、結果を保存する。

処理の流れ
----------
1. srt.load_scene() でシーン XML を読み込む
2. アンテナアレー・送信機を設定
3. RadioMapSolver で PlanarRadioMap を計算
4. rss [W] を dBm に変換して numpy 配列として保存

座標系の対応
------------
ローカル座標: x ∈ [0, size_m], y ∈ [0, size_m]（左下原点）
RadioMap center: シーン中心 = (size_m/2, size_m/2, rx_height_m)
RadioMap size  : [size_m, size_m]
セル (i, j) の左下座標: (i * cell_size, j * cell_size)
  → rss_map[j, i] がセル (i, j) の RSS に対応

出力
----
data/processed/<scene_name>/
    rss_map.npz
        rss_dbm  : ndarray shape (num_tx, num_cells_y, num_cells_x) [dBm]
        cell_size: float [m]
        size_m   : float [m]
        tx_positions: ndarray shape (num_tx, 3) [m]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
Vec3 = tuple[float, float, float]


def compute_rss_map(
    xml_path: Path,
    tx_position: list[float],
    size_m: float,
    cell_size_m: float,
    frequency_hz: float,
    tx_power_dbm: float,
    rx_height_m: float,
    num_samples: int,
    max_depth: int,
) -> dict:
    """
    Sionna RT でシーンを読み込み、RSS マップを計算する。

    Parameters
    ----------
    xml_path      : Mitsuba XML シーンファイルのパス
    tx_position   : 送信機位置 [x, y, z] m（ローカル座標）
    size_m        : エリアの一辺の長さ [m]
    cell_size_m   : セルの一辺の長さ [m]
    frequency_hz  : 搬送波周波数 [Hz]
    tx_power_dbm  : 送信電力 [dBm]
    rx_height_m   : 受信面の高さ [m]（地面からの高さ）
    num_samples   : レイトレーシングのサンプル数
    max_depth     : 最大反射回数

    Returns
    -------
    dict with keys:
        rss_dbm      : ndarray (num_tx, num_cells_y, num_cells_x) [dBm]
        cell_size_m  : float [m]
        size_m       : float [m]
        tx_positions : ndarray (num_tx, 3) [m]
    """
    # Sionna RT のインポートは実行時まで遅延（GPU 不要環境での import エラーを回避）
    import mitsuba as mi
    from sionna.rt import (
        PlanarArray,
        RadioMapSolver,
        Transmitter,
        load_scene,
    )

    # --- 1. シーン読み込み ---
    logger.info("Loading scene: %s", xml_path)
    scene = load_scene(str(xml_path))
    scene.frequency = frequency_hz
    logger.info("Scene loaded. Frequency: %.2e Hz", frequency_hz)

    # --- 2. アンテナアレー設定（等方性 dipole, 単一アンテナ）---
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="V",
    )

    # --- 3. 送信機設定 ---
    tx_power_w = 10 ** ((tx_power_dbm - 30) / 10)  # dBm → W
    tx = Transmitter(
        name="tx0",
        position=mi.Point3f(*tx_position),
        power_dbm=int(tx_power_dbm),
    )
    scene.add(tx)
    logger.info(
        "Transmitter: position=%s, power=%.1f dBm (%.4f W)",
        tx_position,
        tx_power_dbm,
        tx_power_w,
    )

    # --- 4. RadioMapSolver で PlanarRadioMap を計算 ---
    # RadioMapSolver の center はシーン座標系の中心
    # ローカル座標: 左下=(0,0), 右上=(size_m, size_m)
    # → center = (size_m/2, size_m/2, rx_height_m)
    rm_center = mi.Point3f(size_m / 2, size_m / 2, rx_height_m)
    rm_size = mi.Point2f(size_m, size_m)
    rm_cell = mi.Point2f(cell_size_m, cell_size_m)

    logger.info(
        "RadioMapSolver: center=%s, size=%s, cell_size=%s, num_samples=%d, max_depth=%d",
        rm_center,
        rm_size,
        rm_cell,
        num_samples,
        max_depth,
    )

    solver = RadioMapSolver()
    radio_map = solver(
        scene,
        max_depth=max_depth,
        samples_per_tx=num_samples,
        cell_size=mi.Point2f(*rm_cell),
        center=rm_center,
        size=rm_size,
        orientation=mi.Point3f(0, 0, 0),
    )

    # --- 5. RSS [W] → dBm に変換 ---
    # radio_map.rss shape: (num_tx, num_cells_y, num_cells_x)
    rss_w = np.array(radio_map.rss)
    # 0 W を -inf dBm にならないようにクリップ（数値安定性）
    rss_w_clipped = np.where(rss_w > 0, rss_w, np.finfo(float).tiny)
    rss_dbm = 10 * np.log10(rss_w_clipped) + 30  # W → dBm

    num_tx, num_cells_y, num_cells_x = rss_dbm.shape
    logger.info(
        "RSS map computed: shape=(%d, %d, %d) [num_tx, num_cells_y, num_cells_x]",
        num_tx,
        num_cells_y,
        num_cells_x,
    )
    logger.info("RSS range: %.1f ~ %.1f dBm", rss_dbm.min(), rss_dbm.max())

    tx_positions = np.array([tx_position], dtype=float)

    return {
        "rss_dbm": rss_dbm,
        "cell_size_m": cell_size_m,
        "size_m": size_m,
        "tx_positions": tx_positions,
    }


def save_rss_map(result: dict, output_dir: Path) -> Path:
    """
    RSS マップを npz ファイルとして保存する。

    Parameters
    ----------
    result     : compute_rss_map() の戻り値
    output_dir : 出力ディレクトリ

    Returns
    -------
    Path : 保存した npz ファイルのパス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rss_map.npz"

    np.savez(
        out_path,
        rss_dbm=result["rss_dbm"],
        cell_size_m=result["cell_size_m"],
        size_m=result["size_m"],
        tx_positions=result["tx_positions"],
    )
    logger.info(
        "Saved RSS map: %s  shape=%s",
        out_path,
        result["rss_dbm"].shape,
    )
    return out_path
