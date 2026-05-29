# src/radio_map_estimation/scene/building_mask.py
"""
PLY に含まれる建物と AreaSpec から building_mask と building_heights を生成し,
BuildingData として返す

役割
----
1. 面積比判定で building_mask (建物セルの bool マップ) を生成
2. セルごとの最大建物高さ building_heights を生成

判定基準
--------
    intersection_area(建物, セル) / cell_area >= threshold

ローカル座標系での処理
----------------------
入力の gdf は投影座標系 (EPSG:6677)
BuildingData.to_local_gdf() でローカル座標 (bbox 左下原点) に変換してから
グリッドとの交差判定を行う
"""

import logging

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.strtree import STRtree

from radio_map_estimation.scene.schema import AreaSpec, BuildingData

logger = logging.getLogger(__name__)


def make_building_data(
    gdf_proj: gpd.GeoDataFrame,
    area_spec: AreaSpec,
    threshold: float = 0.5,  # 面積比閾値 (例: 0.5 → 50%以上で建物セルと判定)
) -> BuildingData:
    """
    面積比判定で building_mask と building_heights を生成し BuildingData を返す

    Parameters
    ----------
    gdf_proj  : bbox_filter.load_and_filter() の第1戻り値
                投影座標系 (EPSG:6677) の建物 GeoDataFrame
                カラム: geometry, height_m
                ※ PLY に含まれる建物 (押し出し成功) のみを渡すこと
    area_spec : bbox_filter.load_and_filter() の第2戻り値
    threshold : 建物セル判定の面積比閾値 [0, 1]
                intersection_area / cell_area >= threshold で建物セル

    Returns
    -------
    BuildingData
        building_mask[j, i]    = セル(i,j) が建物セルか (bool)
        building_heights[j, i] = セル(i,j) の最大建物高さ [m] (float)
        [j, i] = [y_idx, x_idx],[0, 0] が左下
        rss_dbm[tx, j, i] と同じインデックス
    """
    n = area_spec.grid_size
    cell_size = area_spec.cell_size_m
    cell_area = cell_size**2

    mask = np.zeros((n, n), dtype=bool)
    heights = np.zeros((n, n), dtype=float)

    if gdf_proj.empty:
        logger.warning("gdf_proj is empty. Returning zero mask and heights.")
        return BuildingData(
            gdf=gdf_proj,
            building_mask=mask,
            building_heights=heights,
            area_spec=area_spec,
        )

    # 投影座標 → ローカル座標 (bbox 左下原点) に平行移動
    gdf_local = gdf_proj.copy()
    gdf_local["geometry"] = gdf_local["geometry"].translate(
        xoff=-area_spec.bbox_xmin,
        yoff=-area_spec.bbox_ymin,
    )

    geoms = list(gdf_local.geometry)
    h_vals = gdf_local["height_m"].fillna(0.0).to_numpy()
    tree = STRtree(geoms)

    hit_count = 0
    for j in range(n):
        for i in range(n):
            cell = box(
                i * cell_size,
                j * cell_size,
                (i + 1) * cell_size,
                (j + 1) * cell_size,
            )
            candidates = tree.query(cell)
            if len(candidates) == 0:
                continue

            total_area = 0.0
            max_height = 0.0
            for k in candidates:
                if not geoms[k].intersects(cell):
                    continue
                total_area += geoms[k].intersection(cell).area
                max_height = max(max_height, h_vals[k])

            if total_area / cell_area >= threshold:
                mask[j, i] = True
                heights[j, i] = max_height
                hit_count += 1

    logger.info(
        "Building mask (threshold=%.0f%%): %d / %d cells occupied (%.1f%%)",
        threshold * 100,
        hit_count,
        n * n,
        100 * hit_count / (n * n),
    )

    return BuildingData(
        gdf=gdf_proj,
        building_mask=mask,
        building_heights=heights,
        area_spec=area_spec,
    )
