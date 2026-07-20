"""
bldg_footprint.ply から建物上セルのマスクを生成する

役割
----
bldg_footprint.ply (LOD1 底面フットプリント、z=0 のローカル座標メッシュ) を
読み込み、各セルの左下端点が建物フットプリント内にあれば「建物上」と判定する

設計方針
--------
- 観測点座標 = セル左下端という設計決定と完全に整合させる
- bldg.parquet の geometry (EPSG:6677 → ローカル座標変換済み) を使うため
  LOD1/LOD2 問わず確実なフットプリントが得られる
- bldg_extruder.build_bldg_footprint_mesh() が生成した PLY を入力とする
- margin = area_size_m / 5 は build_area_spec.py と同じ定義を用い、
  bbox の広がりとマスクの広がりを完全に一致させる
- 純粋関数のみ (I/O なし、副作用なし)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import shapely
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..utils.grid_transform import bldg_index_to_coord, margin_num_cells

logger = logging.getLogger(__name__)


def build_bldg_mask(
    bldg_footprint_ply_path: Path,
    cfg,
    output_dir: Path,
) -> np.ndarray:
    """
    bldg_footprint.ply から建物上セルのマスクを生成する

    各セルの左下端点が建物フットプリントポリゴン内にある場合に
    True (建物上) と判定する

    マージン (build_area_spec.py と同一定義: margin = area_size_m / 5) の分だけ
    有効エリア [0, area_size_m] の外側にもマスクを拡張する.

    Parameters
    ----------
    bldg_footprint_ply_path : bldg_footprint.ply のパス
    area_size_m             : 対象エリア (マージンを含まないコア部分) の一辺の長さ [m]
    bldg_cell_size_m        : bldg_mask のセルサイズ [m]
    margin_m                : 有効エリア外側に拡張するマージン [m]
                              (areaspec_builder.py の margin と同じ値を渡す)

    Returns
    -------
    np.ndarray
        shape: (num_cells_total, num_cells_total), dtype=bool
        num_cells_total = num_cells_core + 2 * num_margin_cells
        インデックス k (0-based) は実座標 x = (k - num_margin_cells) * cell_size に対応する.
        margin_m=0.0 の場合は従来通り k がそのまま x = k * cell_size に対応する.
    """
    if not bldg_footprint_ply_path.exists():
        raise FileNotFoundError(f"bldg_footprint.ply not found: {bldg_footprint_ply_path}")

    num_cells_core = int(cfg.area_size_m / cfg.bldg_cell_size_m)
    num_margin_cells = margin_num_cells(cfg.margin_m, cfg.bldg_cell_size_m)
    num_cells_total = num_cells_core + 2 * num_margin_cells

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
        return np.zeros((num_cells_total, num_cells_total), dtype=bool)

    bldg_union = unary_union(polys)
    logger.info(
        "Building footprint union: area=%.1f m², num_polygons=%d",
        bldg_union.area,
        len(polys),
    )

    # 全セルの (row, col) インデックスから、共通関数で座標を生成する
    idx = np.arange(num_cells_total)
    row_idx, col_idx = np.meshgrid(idx, idx, indexing="ij")  # row_idx: y方向, col_idx: x方向
    rows_flat = row_idx.ravel()
    cols_flat = col_idx.ravel()

    coords = bldg_index_to_coord(rows_flat, cols_flat, cfg.bldg_cell_size_m, cfg.margin_m)  # (N,2)

    # 左下端点が建物フットプリント内にあるセルを建物上と判定
    mask_flat = shapely.contains_xy(bldg_union, coords[:, 0], coords[:, 1])
    bldg_mask = mask_flat.reshape(num_cells_total, num_cells_total)

    logger.info(
        "Building mask: %d / %d cells on-building (margin=%.1f m, num_margin_cells=%d)",
        bldg_mask.sum(),
        bldg_mask.size,
        cfg.margin_m,
        num_margin_cells,
    )

    # bldg_map.npz: 配列を保存
    np.savez(
        output_dir / "bldg_map.npz",
        bldg_mask=bldg_mask,
        area_size_m=cfg.area_size_m,
        bldg_cell_size_m=cfg.bldg_cell_size_m,
        margin_m=cfg.margin_m,
    )
    logger.info("Saved: %s", output_dir / "bldg_map.npz")

    return bldg_mask
