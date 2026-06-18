"""
bldg_footprint.ply から建物上セルのマスクを生成する

役割
----
bldg_footprint.ply (LOD1 底面フットプリント、z=0 のローカル座標メッシュ) を
読み込み、各セルに対して建物ポリゴンが占める面積割合が coverage_ratio 以上で
あれば「建物上」と判定する

設計方針
--------
- bldg.parquet の geometry (EPSG:6677 → ローカル座標変換済み) を使うため
  LOD1/LOD2 問わず確実なフットプリントが得られる
- bldg_extruder.build_bldg_footprint_mesh() が生成した PLY を入力とする
- 純粋関数のみ (I/O なし、副作用なし)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def build_bldg_mask(
    bldg_footprint_ply_path: Path,
    area_size_m: float,
    cell_size_m: float,
    coverage_ratio: float = 0.99,
) -> np.ndarray:
    """
    bldg_footprint.ply から建物上セルのマスクを生成する

    各セルについて、建物フットプリントポリゴンがセル面積の
    coverage_ratio 以上を占める場合に True (建物上) と判定する

    Parameters
    ----------
    bldg_footprint_ply_path : bldg_footprint.ply のパス
    area_size_m             : 対象エリアの一辺の長さ [m]
    cell_size_m             : セルサイズ [m]
    coverage_ratio          : 建物と判定するセルカバレッジ閾値 (デフォルト: 0.75)

    Returns
    -------
    np.ndarray
        shape: (num_cells_y, num_cells_x), dtype=bool
        True = 建物上のセル

    Raises
    ------
    FileNotFoundError
        bldg_footprint.ply が存在しない場合
    """
    if not bldg_footprint_ply_path.exists():
        raise FileNotFoundError(f"bldg_footprint.ply not found: {bldg_footprint_ply_path}")

    num_cells = int(area_size_m / cell_size_m)

    # bldg_footprint.ply を読み込み、各面を shapely Polygon に変換して unary_union
    mesh = trimesh.load(str(bldg_footprint_ply_path))
    verts_xy = mesh.vertices[:, :2]  # type: ignore

    polys = []
    for face in mesh.faces:  # type: ignore
        pts = verts_xy[face]
        poly = Polygon(pts)
        if poly.is_valid and not poly.is_empty:
            polys.append(poly)

    if not polys:
        logger.warning("No valid footprint polygons found in bldg_footprint.ply.")
        return np.zeros((num_cells, num_cells), dtype=bool)

    bldg_union = unary_union(polys)
    logger.info(
        "Building footprint union: area=%.1f m², num_polygons=%d",
        bldg_union.area,
        len(polys),
    )

    # セルグリッドを生成して各セルの建物カバレッジを計算
    cell_area = cell_size_m**2
    mask = np.zeros((num_cells, num_cells), dtype=bool)

    for j in range(num_cells):  # y 方向
        for i in range(num_cells):  # x 方向
            x_min = i * cell_size_m
            x_max = x_min + cell_size_m
            y_min = j * cell_size_m
            y_max = y_min + cell_size_m
            cell_box = box(x_min, y_min, x_max, y_max)
            intersection = bldg_union.intersection(cell_box)
            if intersection.area / cell_area >= coverage_ratio:
                mask[j, i] = True

    logger.info(
        "Building mask: %d / %d cells on-building (coverage_ratio=%.2f)",
        mask.sum(),
        mask.size,
        coverage_ratio,
    )
    return mask
