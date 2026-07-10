"""
bldg.ply の頂点から TX 位置を自動決定する

設計方針
--------
- bldg.ply は全建物を単一メッシュに結合したものであり個別建物の識別情報を持たない
- mesh.split() で建物ごとに分離し、各建物の代表点 (屋根最高頂点の z + 建物重心の x,y) を候補とする
- 建物の重心 (x, y) に配置することで TX が建物端に偏ることを防ぎ、
  全方向に均等に電波が届くようにする (iso パターンとの整合性)
- 候補エリアは外周 10% を除いた内側に限定し、隅への配置を防ぐ
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# UMa 基地局アンテナ高オフセット [m] (屋上からの高さ)
_TX_ANTENNA_OFFSET_M = 3.0


def _build_candidates(
    mesh: trimesh.Trimesh,
) -> np.ndarray:
    """
    bldg.ply を建物ごとに分離し、各建物の TX 候補点を返す

    各建物の候補点:
        x, y : 建物全頂点の重心 (建物中心上空に TX を配置するため)
        z    : 建物の屋根最高頂点の z (最も高い点)

    建物端の頂点に配置すると建物自体が遮蔽物となり指向性が生まれるため、
    重心を使うことで iso パターンとの整合性を保つ

    Parameters
    ----------
    mesh : bldg.ply の trimesh

    Returns
    -------
    np.ndarray
        shape: (num_buildings, 3)、各行が (x, y, z) の候補点
    """
    components = mesh.split(only_watertight=False)
    candidates = []
    for comp in components:
        v = comp.vertices
        cx = float(v[:, 0].mean())
        cy = float(v[:, 1].mean())
        z_max = float(v[:, 2].max())
        candidates.append((cx, cy, z_max))
    return np.array(candidates)  # (num_buildings, 3)


def build_tx_positions(
    bldg_ply_path: Path,
    num_tx: int,
    center_search_radius_m,
    min_separation_m: float,
    area_size_m: float,
) -> list[tuple[float, float, float]]:
    """
    bldg.ply の建物ごとの重心から num_tx 個の TX 位置を順次決定する

    処理フロー:
        1. mesh.split() で建物ごとに分離し、各建物の候補点 (重心 x,y + 屋根最高 z) を生成
        2. 候補エリアを限定
           [area_size_m/5, area_size_m*4/5] x [area_size_m/5, area_size_m*4/5]
        3. 1局目: エリア中心 (area_size_m/2, area_size_m/2) から
           center_search_radius_m 以内の候補の中で z 最大の建物に配置する
           候補がない場合は available な全候補から argmax(z) にフォールバック
        4. 1局目以降: available な候補の中で z 最大の建物を TX として選択
        5. 配置済み全 TX から min_separation_m 未満の候補を除外
        6. num_tx 回繰り返す

    Parameters
    ----------
    bldg_ply_path    : bldg.ply のパス
    num_tx           : 配置する TX 数
    min_separation_m : TX 間の最小離間距離 [m]
    area_size_m      : 対象エリアの一辺の長さ [m]

    Returns
    -------
    list of (x, y, z) タプル (長さ num_tx)

    Raises
    ------
    FileNotFoundError
        bldg.ply が存在しない場合
    RuntimeError
        候補が不足して num_tx 個配置できない場合
    """
    if not bldg_ply_path.exists():
        raise FileNotFoundError(f"bldg.ply not found: {bldg_ply_path}")

    mesh = trimesh.load(str(bldg_ply_path))

    # 建物ごとの候補点 (重心 x,y + 屋根最高 z) を生成
    cands = _build_candidates(mesh)  # type: ignore
    logger.info("bldg: %d buildings detected via mesh.split()", len(cands))

    # 候補エリアを外周 20% を除いた内側に絞る
    margin = area_size_m / 5
    in_area = (
        (cands[:, 0] >= margin)
        & (cands[:, 0] <= area_size_m - margin)
        & (cands[:, 1] >= margin)
        & (cands[:, 1] <= area_size_m - margin)
    )
    available = in_area.copy()
    logger.info(
        "TX candidates: %d / %d buildings in [%.0f, %.0f] x [%.0f, %.0f]",
        available.sum(),
        len(cands),
        margin,
        area_size_m - margin,
        margin,
        area_size_m - margin,
    )

    tx_positions: list[tuple[float, float, float]] = []
    ecx, ecy = area_size_m / 2, area_size_m / 2  # エリア中心

    for i in range(num_tx):
        candidates = np.where(available)[0]
        if len(candidates) == 0:
            raise RuntimeError(
                f"Cannot place TX {i}/{num_tx}: no available candidates. "
                f"Try reducing num_tx or min_separation_m ({min_separation_m} m)."
            )
        # 1局目の配置ロジック
        elif i == 0:
            # エリア中心から center_search_radius_m 以内の候補の中で z 最大
            center_dists = np.hypot(cands[:, 0] - ecx, cands[:, 1] - ecy)
            near_center = available & (center_dists <= center_search_radius_m)
            if near_center.any():
                near_candidates = np.where(near_center)[0]
                idx = near_candidates[int(np.argmax(cands[near_candidates, 2]))]
                logger.info(
                    "TX%d: placed near center (%d candidates within %.0f m of center)",
                    i,
                    near_center.sum(),
                    center_search_radius_m,
                )
            else:
                # フォールバック: 中央付近に建物がない場合は全候補から argmax(z)
                logger.warning(
                    "TX%d: no candidates within %.0f m of center, falling back to available argmax(z).",
                    i,
                    center_search_radius_m,
                )
                idx = candidates[int(np.argmax(cands[candidates, 2]))]
        # 2局目の配置ロジック
        else:
            # available な候補の中で z 最大
            idx = candidates[int(np.argmax(cands[candidates, 2]))]

        tx_x, tx_y = float(cands[idx, 0]), float(cands[idx, 1])
        tx_z = float(cands[idx, 2]) + _TX_ANTENNA_OFFSET_M
        tx_positions.append((tx_x, tx_y, tx_z))

        logger.info(
            "TX%d: position=(%.1f, %.1f, %.1f) m  [bldg_z=%.1f m + offset=%.1f m]",
            i,
            tx_x,
            tx_y,
            tx_z,
            tx_z - _TX_ANTENNA_OFFSET_M,
            _TX_ANTENNA_OFFSET_M,
        )

        # 配置済み全 TX から min_separation_m 未満の候補を除外
        for prev_x, prev_y, _ in tx_positions:
            dists = np.hypot(cands[:, 0] - prev_x, cands[:, 1] - prev_y)
            available &= dists >= min_separation_m

    return tx_positions
