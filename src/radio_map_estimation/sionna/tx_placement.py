"""
bldg.ply の頂点から TX 位置を自動決定する

設計方針
--------
- bldg.ply は全建物を単一メッシュに結合したものであり個別建物の識別情報を持たない
- 頂点単位で候補を管理し、配置済み TX から min_separation_m 以上離れた頂点の中から
  z 最大の頂点を次の TX として選ぶ
- 候補エリアは外周 10% を除いた内側に限定し、隅への配置を防ぐ
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# UMa 基地局アンテナ高オフセット [m] (屋上からの高さ)
_TX_ANTENNA_OFFSET_M = 2.0


def build_tx_positions(
    bldg_ply_path: Path,
    num_tx: int,
    min_separation_m: float,
    area_size_m: float,
) -> list[tuple[float, float, float]]:
    """
    bldg.ply の全頂点から num_tx 個の TX 位置を順次決定する

    処理フロー:
        1. 候補エリアを外周 10% を除いた内側に限定
           [area_size_m/10, area_size_m*9/10] x [area_size_m/10, area_size_m*9/10]
        2. available な候補の中で z 最大の頂点を TX として選択
        3. 配置済み全 TX から min_separation_m 未満の頂点を候補から除外
        4. num_tx 回繰り返す

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
        候補頂点が不足して num_tx 個配置できない場合
    """
    if not bldg_ply_path.exists():
        raise FileNotFoundError(f"bldg.ply not found: {bldg_ply_path}")

    mesh = trimesh.load(str(bldg_ply_path))
    verts = mesh.vertices  # type: ignore

    # 候補エリアを外周 10% を除いた内側に絞る
    margin = area_size_m / 10
    in_area = (
        (verts[:, 0] >= margin)
        & (verts[:, 0] <= area_size_m - margin)
        & (verts[:, 1] >= margin)
        & (verts[:, 1] <= area_size_m - margin)
    )
    available = in_area.copy()
    logger.info(
        "TX candidates: %d / %d vertices in [%.0f, %.0f] x [%.0f, %.0f]",
        available.sum(),
        len(verts),
        margin,
        area_size_m - margin,
        margin,
        area_size_m - margin,
    )

    tx_positions: list[tuple[float, float, float]] = []

    for i in range(num_tx):
        candidates = np.where(available)[0]
        if len(candidates) == 0:
            raise RuntimeError(
                f"Cannot place TX {i + 1}/{num_tx}: no available candidates. "
                f"Try reducing num_tx or min_separation_m ({min_separation_m} m)."
            )

        idx = candidates[int(np.argmax(verts[candidates, 2]))]
        tx_x, tx_y = float(verts[idx, 0]), float(verts[idx, 1])
        tx_z = float(verts[idx, 2]) + _TX_ANTENNA_OFFSET_M
        tx_positions.append((tx_x, tx_y, tx_z))

        logger.info(
            "TX%d: position=(%.1f, %.1f, %.1f) m  [bldg_z=%.1f m + offset=%.1f m]",
            i + 1,
            tx_x,
            tx_y,
            tx_z,
            tx_z - _TX_ANTENNA_OFFSET_M,
            _TX_ANTENNA_OFFSET_M,
        )

        # 配置済み全 TX から min_separation_m 未満の頂点を候補から除外
        for prev_x, prev_y, _ in tx_positions:
            dists = np.hypot(verts[:, 0] - prev_x, verts[:, 1] - prev_y)
            available &= dists >= min_separation_m

    return tx_positions
