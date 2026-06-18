"""
radio map の PNG / npz 保存

役割
----
1. MeshRadioMap → scene.render_to_file() で radio_map_3d.png
2. PlanarRadioMap → show() / show_association() で 2D マップを PNG 保存
3. npz に数値データを保存 (radio_map.npz)

npz の配列定義:
    rss_dbm_raw        : Sionna RT 生出力 (ノイズなし、マスクなし)
    rss_dbm_noise      : ノイズ付加済み (建物上マスクなし)
    rss_dbm_gt         : 真値 (建物上 + 検出不可能を除外、ノイズ付加済み、それ以外は nan)
    mask_on_bldg       : 建物マスク (True = 建物上)
    mask_detectable    : 観測可能マスク (True = 観測可能)
    tx_association     : 接続 TX インデックス (未到達セルは -1)
    tx_positions       : TX 位置
    cell_size_m        : セルサイズ [m]
    noise_std_db       : 観測ノイズ標準偏差 [dB]

出力ファイル:
    radio_map_3d.png          3D シーン + RSS オーバーレイ
    radio_map_path_gain.png   PlanarRadioMap: path_gain
    radio_map_rss.png         PlanarRadioMap: RSS [dBm]
    radio_map_sinr.png        PlanarRadioMap: SINR [dB]
    radio_map_association.png TX ごとの接続エリア
    radio_map.npz             全配列 (rss_dbm_raw / rss_dbm_noise / rss_dbm_gt /
                              mask_on_bldg / mask_detectable / tx_association /
                              tx_positions / cell_size_m / noise_std_db)

設計方針
--------
- 保存のみを担う (計算は radiomap.py へ)
- matplotlib は GUI なし環境向けに Agg バックエンドを使用
- Sionna RT のインポートは関数内で行う (Mitsuba variant の自動設定のため)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from radio_map_estimation.sionna.bldg_mask import build_bldg_mask

logger = logging.getLogger(__name__)

# 検出可能な RSS の閾値 [dBm]
_UNDETECTABLE_THRESHOLD_DBM = -120.0


def save_radio_maps(
    scene,
    mesh_radio_map,
    planar_radio_map,
    tx_positions: list[tuple[float, float, float]],
    cell_size_m: float,
    noise_std_db: float,
    rng: np.random.Generator,
    area_size_m: float,
    bldg_footprint_ply_path: Path,
    output_dir: Path,
) -> None:
    """
    MeshRadioMap / PlanarRadioMap を PNG / npz として保存する

    Parameters
    ----------
    scene            : Sionna RT シーンオブジェクト (render_to_file に使用)
    mesh_radio_map   : MeshRadioMap (radiomap.build_radio_maps の出力)
    planar_radio_map : PlanarRadioMap (radiomap.build_radio_maps の出力)
    tx_positions     : TX 位置のリスト
    cell_size_m      : セルサイズ [m]
    noise_std_db     : 観測ノイズの標準偏差 [dB]
    rng              : NumPy 乱数ジェネレータ (再現性のため外部から受け渡す)
    area_size_m      : 対象エリアの一辺の長さ [m]
    bldg_footprint_ply_path : bldg_footprint.ply のパス (建物マスク生成に使用)
    output_dir       : 出力ディレクトリ
    """
    from sionna.rt import Camera

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 3D シーン + RSS オーバーレイ (MeshRadioMap)
    cam = Camera(position=[500.0, -1000.0, 1500.0])  # type: ignore
    cam.look_at(np.array([500.0, 500.0, 50.0]))  # type: ignore

    render_path = output_dir / "radio_map_3d.png"
    scene.render_to_file(
        camera=cam,
        radio_map=mesh_radio_map,
        filename=str(render_path),
        resolution=[1920, 1080],
        rm_metric="rss",
    )
    logger.info("Saved: %s", render_path)

    # 2. PlanarRadioMap の show() / show_association() で 2D マップを保存
    for metric in ("path_gain", "rss", "sinr"):
        fig = planar_radio_map.show(metric=metric, show_tx=True)
        out = output_dir / f"radio_map_{metric}.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", out)

    fig = planar_radio_map.show_association(metric="rss", show_tx=True)
    out = output_dir / "radio_map_association.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)

    # 3. npz 保存
    # 3a. 各配列の計算
    # MeshRadioMap: 3D レンダリング用 (不規則な DEM メッシュ上の点群)
    # PlanarRadioMap: グリッド上の RSS (mask / npz 保存はこちらを使用)
    rss_w_planar = np.array(planar_radio_map.rss)  # (num_tx, num_cells_y, num_cells_x)

    # num_tx > 1 の場合は全 TX の最大値を取る → (num_cells_y, num_cells_x)
    rss_w_max = rss_w_planar.max(axis=0)
    rss_dbm_raw: np.ndarray = 10.0 * np.log10(rss_w_max / 1e-3 + 1e-30)

    # 各セルに最も強い RSS を届けている TX インデックス (接続 TX)
    # RSS が全 TX で 0 のセル (未到達) は -1 とする
    tx_association: np.ndarray = np.where(
        rss_w_max > 0,
        np.argmax(rss_w_planar, axis=0),
        -1,
    ).astype(np.int32)

    noise: np.ndarray = rng.normal(0.0, noise_std_db, size=rss_dbm_raw.shape)
    rss_dbm_noise: np.ndarray = rss_dbm_raw + noise

    # 建物マスク (True = 建物上)
    mask_on_bldg: np.ndarray = build_bldg_mask(
        bldg_footprint_ply_path=bldg_footprint_ply_path,
        area_size_m=area_size_m,
        cell_size_m=cell_size_m,
    )

    # 検出可能マスク (True = 検出可能)
    mask_detectable: np.ndarray = rss_dbm_raw >= _UNDETECTABLE_THRESHOLD_DBM

    # rss_dbm_gt: 建物上 + 検出不可能を除外した真値 (ノイズ付加済み、それ以外は nan)
    rss_dbm_gt: np.ndarray = np.where(~mask_on_bldg & mask_detectable, rss_dbm_noise, np.nan)

    n_total = mask_on_bldg.size
    n_on_bldg = int(mask_on_bldg.sum())
    n_detectable = int(mask_detectable.sum())
    n_observable = int(np.sum(~np.isnan(rss_dbm_gt)))
    logger.info(
        "Masks: on_bldg=%d/%d (%.1f%%), detectable=%d/%d (%.1f%%)",
        n_on_bldg,
        n_total,
        100.0 * n_on_bldg / n_total,
        n_detectable,
        n_total,
        100.0 * n_detectable / n_total,
    )
    logger.info(
        "Observation rate (non_bldg and detectable): %d/%d (%.1f%%)",
        n_observable,
        n_total,
        100.0 * n_observable / n_total,
    )

    # 3b. radio_map.npz: 全配列を保存
    np.savez(
        output_dir / "radio_map.npz",
        rss_dbm_raw=rss_dbm_raw,
        rss_dbm_noise=rss_dbm_noise,
        rss_dbm_gt=rss_dbm_gt,
        mask_on_bldg=mask_on_bldg,
        mask_detectable=mask_detectable,
        tx_association=tx_association,
        tx_positions=np.array(tx_positions),
        cell_size_m=cell_size_m,
        noise_std_db=noise_std_db,
    )
    logger.info("Saved: %s", output_dir / "radio_map.npz")
