"""
radio map の PNG / npz 保存

役割
----
1. MeshRadioMap → scene.render_to_file() で radio_map_3d.png
2. PlanarRadioMap → show() / show_association() で 2D マップを PNG 保存
3. npz に数値データを保存 (rss_gt.npz / rss.npz)

npz の配列定義:
    rss_dbm_raw        : Sionna RT 生出力 (ノイズなし、マスクなし)
    rss_dbm            : ノイズ付加済み (建物上マスクなし)
    rss_dbm_gt         : 真値 (建物上マスク適用済み、ノイズ付加済み)
    rss_dbm_observable : 観測可能点のみ (建物上 + 検出不可能を除外、ノイズ付加済み)
    mask_observed      : 観測点候補マスク (True = 観測可能)
    mask_estimable     : 推定対象マスク (True = 建物上でない)
    tx_positions       : TX 位置
    cell_size_m        : セルサイズ [m]
    noise_std_db       : 観測ノイズ標準偏差 [dB]

出力ファイル:
    radio_map_3d.png          3D シーン + RSS オーバーレイ
    radio_map_path_gain.png   PlanarRadioMap: path_gain
    radio_map_rss.png         PlanarRadioMap: RSS [dBm]
    radio_map_sinr.png        PlanarRadioMap: SINR [dB]
    radio_map_association.png TX ごとの接続エリア
    rss_gt.npz                全配列 (rss_dbm_raw / rss_dbm / rss_dbm_gt /
                              rss_dbm_observable / mask_observed / mask_estimable /
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

    noise: np.ndarray = rng.normal(0.0, noise_std_db, size=rss_dbm_raw.shape)
    rss_dbm: np.ndarray = rss_dbm_raw + noise

    # 建物マスク (True = 建物上)
    mask_on_bldg: np.ndarray = build_bldg_mask(
        bldg_footprint_ply_path=bldg_footprint_ply_path,
        area_size_m=area_size_m,
        cell_size_m=cell_size_m,
    )

    # mask_estimable: 建物上でない = 推定対象 (検出不可能点も含む)
    mask_estimable: np.ndarray = ~mask_on_bldg

    # mask_observed: 推定対象 かつ 検出可能 = 観測点候補
    mask_observed: np.ndarray = mask_estimable & (rss_dbm_raw >= _UNDETECTABLE_THRESHOLD_DBM)

    # rss_dbm_gt: 真値 (建物上マスク適用済み、ノイズ付加済み)
    rss_dbm_gt: np.ndarray = np.where(mask_estimable, rss_dbm, np.nan)

    # rss_dbm_observable: 観測可能点のみ (建物上 + 検出不可能を除外、ノイズ付加済み)
    rss_dbm_observable: np.ndarray = np.where(mask_observed, rss_dbm, np.nan)

    logger.info(
        "Masks: estimable=%d/%d (%.1f%%), observed=%d/%d (%.1f%%)",
        mask_estimable.sum(),
        mask_estimable.size,
        100.0 * mask_estimable.sum() / mask_estimable.size,
        mask_observed.sum(),
        mask_observed.size,
        100.0 * mask_observed.sum() / mask_observed.size,
    )

    # 3b. radio_map.npz: 全配列を保存
    np.savez(
        output_dir / "radio_map.npz",
        rss_dbm_raw=rss_dbm_raw,
        rss_dbm=rss_dbm,
        rss_dbm_gt=rss_dbm_gt,
        rss_dbm_observable=rss_dbm_observable,
        mask_observed=mask_observed,
        mask_estimable=mask_estimable,
        tx_positions=np.array(tx_positions),
    )
    logger.info("Saved: %s", output_dir / "rss_gt.npz")
