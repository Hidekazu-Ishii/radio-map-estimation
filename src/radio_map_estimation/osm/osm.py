# src/radio_map_estimation/scene/osm_buildings.py
"""
目的: OpenStreetMapから建物ポリゴンを取得して、指定エリアの
      BuildingData (グリッド + GeoDataFrame) を生成する.

ローカル座標系の定義:
    bbox_m の左下 (xmin, ymin) を原点 (0, 0) とする直交座標系.
    すなわち x ∈ [0, area_size_m], y ∈ [0, area_size_m].
    Sionna シーン・グリッドインデックスはすべてこの座標系を使う.
    - グリッド [row, col] = [y_idx, x_idx], [0,0] が左下 (x=0, y=0)
    - TX/RX 位置もこの座標系で指定する
"""

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import box as shapely_box

from .osm_schema import AreaSpec, BuildingData


def fetch_buildings_osm(
    center_lat: float,
    center_lon: float,
    area_size_m: float,
    cell_size_m: float,
    meters_per_level: float,
) -> BuildingData:
    """
    OSMnx で建物フットプリントと高さを取得し, BuildingData を返す.

    高さの優先順位:
        height タグ → building:levels x meters_per_level → default_building_height_m

    Parameters
    ----------
    center_lat, center_lon : float
        取得エリアの中心緯度・経度.
    area_size_m : float
        取得範囲の一辺 [m] (正方形).
    cell_size_m : float
        グリッドセル一辺 [m]. grid_size = area_size_m / cell_size_m.
    meters_per_level : float
        building:levels から高さへの変換係数 [m/階].

    Returns
    -------
    BuildingData
    """
    half = area_size_m / 2.0

    gdf_raw = ox.features_from_point(
        (center_lat, center_lon),
        tags={"building": True},
        dist=half * 1.1,
    )

    gdf_proj = ox.projection.project_gdf(gdf_raw)
    assert gdf_proj.crs is not None
    crs_str = gdf_proj.crs.to_string()

    center_gdf = gpd.GeoDataFrame(geometry=[Point(center_lon, center_lat)], crs="EPSG:4326")
    center_proj = center_gdf.to_crs(gdf_proj.crs)
    cx = float(center_proj.geometry.x.values[0])
    cy = float(center_proj.geometry.y.values[0])
    bbox_m = (cx - half, cy - half, cx + half, cy + half)

    height_m = _estimate_heights(gdf_proj, meters_per_level)

    gdf_clean = gdf_proj[["geometry"]].reset_index(drop=True).assign(height_m=height_m)
    gdf_clean = gpd.GeoDataFrame(gdf_clean, geometry="geometry", crs=gdf_proj.crs)

    building_mask, building_heights = _rasterize_buildings(
        gdf_proj=gdf_clean,
        bbox_m=bbox_m,
        cell_size_m=cell_size_m,
        area_size_m=area_size_m,
    )

    area_spec = AreaSpec(
        center_lat=center_lat,
        center_lon=center_lon,
        area_size_m=area_size_m,
        cell_size_m=cell_size_m,
        crs=crs_str,
        bbox_xmin=bbox_m[0],
        bbox_ymin=bbox_m[1],
        bbox_xmax=bbox_m[2],
        bbox_ymax=bbox_m[3],
    )

    return BuildingData(
        gdf=gdf_clean,
        building_mask=building_mask,
        building_heights=building_heights,
        area_spec=area_spec,
    )


def _estimate_heights(
    gdf: gpd.GeoDataFrame,
    meters_per_level: float,
) -> np.ndarray:
    """建物高さを推定して ndarray で返す."""
    height_m = np.full(len(gdf), np.nan)

    # 1. height タグ
    if "height" in gdf.columns:
        parsed = pd.to_numeric(gdf["height"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(parsed)
        height_m = np.where(mask, parsed, height_m)
        print(f"  height tag:        {mask.sum()} buildings")

    # 2. building:levels タグ
    if "building:levels" in gdf.columns:
        parsed = pd.to_numeric(gdf["building:levels"], errors="coerce").to_numpy(dtype=float)
        from_levels = parsed * meters_per_level
        mask = np.isnan(height_m) & np.isfinite(from_levels)
        height_m = np.where(mask, from_levels, height_m)
        print(f"  building:levels:   {mask.sum()} buildings")

    # 3. デフォルト値: height / building:levels タグから推定できた建物の平均高さ
    heights_from_tags = height_m[np.isfinite(height_m)]
    if len(heights_from_tags) == 0:
        raise ValueError(
            "height / building:levels タグが1件も取得できませんでした. "
            "これらのタグが整備されている都市・エリアを選んでください."
        )
    default_height_m = float(np.mean(heights_from_tags))

    num_default = int(np.isnan(height_m).sum())
    height_m = np.where(np.isnan(height_m), default_height_m, height_m)
    print(
        f"  default height:    {num_default} buildings "
        f"  ({num_default / len(height_m):.1%}, mean={default_height_m:.1f} m)"
    )
    return height_m


def _rasterize_buildings(
    gdf_proj: gpd.GeoDataFrame,
    bbox_m: tuple[float, float, float, float],
    cell_size_m: float,
    area_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    建物ポリゴンをグリッドにラスタライズする.

    Returns
    -------
    building_mask : (grid_size, grid_size) bool
    building_heights : (grid_size, grid_size) float
    """
    xmin, ymin, _, _ = bbox_m
    grid_size = int(area_size_m / cell_size_m)
    dx = cell_size_m
    clip_box = shapely_box(*bbox_m)

    gdf_clip = gdf_proj.clip(clip_box)
    valid = gdf_clip.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    gdf_valid = gdf_clip[valid].copy()

    building_mask = np.zeros((grid_size, grid_size), dtype=bool)
    building_heights = np.zeros((grid_size, grid_size), dtype=float)

    if len(gdf_valid) == 0:
        print("[warn] No valid building polygons were found.")
        return building_mask, building_heights

    for _, row in gdf_valid.iterrows():
        poly = row.geometry
        h = float(row["height_m"])
        pb = poly.bounds

        col_lo = max(0, int((pb[0] - xmin) / dx))
        col_hi = min(grid_size - 1, int((pb[2] - xmin) / dx))
        row_lo = max(0, int((pb[1] - ymin) / dx))
        row_hi = min(grid_size - 1, int((pb[3] - ymin) / dx))

        for r in range(row_lo, row_hi + 1):
            for c in range(col_lo, col_hi + 1):
                cell_xmin = xmin + c * dx
                cell_ymin = ymin + r * dx
                cell_box = shapely_box(cell_xmin, cell_ymin, cell_xmin + dx, cell_ymin + dx)
                cell_center = Point(cell_xmin + dx / 2, cell_ymin + dx / 2)
                # 面積がセルの 10% 以上を占めるか、中心を含んでいれば建物とする
                if poly.intersection(cell_box).area > (cell_size_m**2 * 0.1) or poly.contains(cell_center):
                    building_mask[r, c] = True
                    building_heights[r, c] = max(building_heights[r, c], h)

    return building_mask, building_heights
