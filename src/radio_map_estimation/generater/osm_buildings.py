"""
src/osm_buildings.py

目的: OpenStreetMapから建物ポリゴンを取得して、指定エリアのグリッド建物マスク (.npz)を生成して保存する.
"""

from pathlib import Path
from typing import cast

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import box as shapely_box


def fetch_buildings_osm(
    center_lat: float,
    center_lon: float,
    area_size_m: float,
    meters_per_level: float,
    default_building_height_m: float,
) -> tuple[gpd.GeoDataFrame, tuple[float, float, float, float]]:
    """
    osmnxで建物フットプリントと高さ情報を取得し、メートル座標系に投影する.

    取得するタグ:
        building: 建物の存在 (必須)
        height: 建物高さ [m] (任意、欠損あり)
        building:levels: 建物階数 (任意、欠損あり、heightの代替)

    高さの優先順位:
        height タグ → building:levels * meters_per_level → default_building_height_m

    Parameters
    ----------
    center_lat : float
        取得エリアの中心緯度
    center_lon : float
        取得エリアの中心経度
    area_size_m : float
        取得範囲の一辺 [m]
    meters_per_level : float
        building:levels タグから高さへの変換係数 [m/階]
    default_building_height_m : float
        height / levels タグが両方欠損している場合のデフォルト高さ [m]

    Returns
    -------
    gdf_proj : GeoDataFrame
        メートル座標系 (UTM) に投影済みの建物ポリゴン.
        'height_m' 列に建物高さ [m] が付与されている.
    bbox_m : (xmin, ymin, xmax, ymax) [m]
        投影座標系での取得エリアのbbox
    """
    half = area_size_m / 2.0

    # 中心点からの距離でエリア取得 (少し広めに取ってからクリップ)
    gdf_buildings = ox.features_from_point(
        (center_lat, center_lon),
        tags={"building": True},
        dist=half * 1.1,
    )

    # メートル座標系 (UTM) に投影
    gdf_proj = ox.projection.project_gdf(gdf_buildings)

    # 中心点をメートル座標に変換してbboxを定義
    center_gdf = gpd.GeoDataFrame(geometry=[Point(center_lon, center_lat)], crs="EPSG:4326")
    assert gdf_proj.crs is not None
    center_proj = center_gdf.to_crs(gdf_proj.crs)
    cx = float(center_proj.geometry.x.values[0])
    cy = float(center_proj.geometry.y.values[0])
    bbox_m = (cx - half, cy - half, cx + half, cy + half)

    # --- 建物高さの推定 ---

    height_m = np.full(len(gdf_proj), np.nan)

    # 1. height タグ [m]
    n_height = 0
    if "height" in gdf_proj.columns:
        parsed_h = pd.to_numeric(gdf_proj["height"], errors="coerce").to_numpy(dtype=float)
        mask_h = np.isfinite(parsed_h)
        height_m = np.where(mask_h, parsed_h, height_m)
        n_height = int(mask_h.sum())

    # 2. building:levels タグ (heightが欠損の箇所のみ補完)
    n_levels = 0
    if "building:levels" in gdf_proj.columns:
        parsed_l = pd.to_numeric(gdf_proj["building:levels"], errors="coerce").to_numpy(dtype=float)
        from_levels = parsed_l * meters_per_level
        mask_l = np.isnan(height_m) & np.isfinite(from_levels)
        height_m = np.where(mask_l, from_levels, height_m)
        n_levels = int(mask_l.sum())

    # 3. 残り欠損はデフォルト値
    n_default = int(np.isnan(height_m).sum())
    height_m = np.where(np.isnan(height_m), default_building_height_m, height_m)

    print(f"      height tag: {n_height}, levels tag: {n_levels}, default: {n_default} buildings")

    gdf_proj = gdf_proj.copy()
    gdf_proj["height_m"] = height_m

    return gdf_proj, bbox_m


def buildings_to_grid(
    gdf_proj: gpd.GeoDataFrame,
    bbox_m: tuple[float, float, float, float],
    grid_size: int,
    area_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    建物ポリゴンをグリッドに変換する.

    各セルに建物が存在するか (bool) と、その建物の高さ [m] を返す.
    複数の建物が重なるセルは最大高さを採用する.

    Parameters
    ----------
    gdf_proj : GeoDataFrame
        メートル座標系に投影済みの建物ポリゴン. 'height_m' 列を含む.
    bbox_m : (xmin, ymin, xmax, ymax) [m]
        投影座標系での取得エリアのbbox
    grid_size : int
        グリッド一辺のセル数
    area_size_m : float
        エリアサイズ [m]

    Returns
    -------
    building_mask : ndarray of shape (grid_size, grid_size), dtype bool
        建物が存在するセルが True.
        インデックス [row, col] は [y_idx, x_idx] に対応し、
        [0, 0] は領域の左下 (ymin, xmin) を指す.
    building_heights : ndarray of shape (grid_size, grid_size), dtype float
        セルの最大建物高さ [m]. 建物がない場合は 0.0.
    """
    xmin, ymin, xmax, ymax = bbox_m
    dx = area_size_m / grid_size

    # 指定範囲 (bbox) でクリップし、有効なポリゴンのみ抽出
    clip_box = shapely_box(xmin, ymin, xmax, ymax)
    gdf_clip = gdf_proj.clip(clip_box)

    # ポリゴン列を抽出 (MultiPolygonも含む)
    geometry_series = cast(gpd.GeoSeries, gdf_clip.geometry.dropna())
    valid_idx = geometry_series.index[geometry_series.geom_type.isin(["Polygon", "MultiPolygon"])]
    polys = cast(gpd.GeoSeries, geometry_series.loc[valid_idx])

    if len(polys) == 0:
        print("[warn] Building polygons not found. Returning empty grids.")
        return (
            np.zeros((grid_size, grid_size), dtype=bool),
            np.zeros((grid_size, grid_size), dtype=float),
        )

    heights = gdf_clip.loc[valid_idx, "height_m"].to_numpy(dtype=float)  # (N,)

    # 判定高速化のため、全建物を結合したジオメトリを用意
    merged = polys.union_all()

    building_mask = np.zeros((grid_size, grid_size), dtype=bool)
    building_heights = np.zeros((grid_size, grid_size), dtype=float)

    # 各セルに対して面積ベースの交差判定を行い、交差するポリゴンがあれば建物とみなす
    for row in range(grid_size):
        for col in range(grid_size):
            # セルの物理座標を計算
            cell_xmin = xmin + col * dx
            cell_ymin = ymin + row * dx
            cell_box = shapely_box(cell_xmin, cell_ymin, cell_xmin + dx, cell_ymin + dx)

            if merged.intersection(cell_box).area < 1e-6:
                continue

            building_mask[row, col] = True

            # 交差するポリゴンの最大高さを採用
            max_h = max(
                (
                    float(h)
                    for poly, h in zip(polys, heights, strict=True)
                    if poly.intersection(cell_box).area > 1e-6
                ),
                default=0.0,
            )
            building_heights[row, col] = max_h

    return building_mask, building_heights


def plot_building_map(
    building_mask: np.ndarray,
    building_heights: np.ndarray,
    area_size_m: float,
    center_lat: float,
    center_lon: float,
    save_path: Path,
) -> None:
    grid_size = building_mask.shape[0]

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 建物マスク
    ax = axes[0]
    img = np.ones((grid_size, grid_size, 3))  # 白: 道路・空地
    img[building_mask] = [0.2, 0.2, 0.6]  # 青: 建物
    ax.imshow(img, origin="lower", extent=(0, area_size_m, 0, area_size_m))
    ax.set_title(
        f"OSM Building Map\n"
        f"center=({center_lat:.4f}, {center_lon:.4f}), "
        f"{area_size_m:.0f}m x {area_size_m:.0f}m, "
        f"{grid_size}x{grid_size} grid"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # 建物高さ
    ax = axes[1]
    im = ax.imshow(
        building_heights,
        origin="lower",
        extent=(0, area_size_m, 0, area_size_m),
        cmap="YlOrRd",
    )
    plt.colorbar(im, ax=ax, label="Building height [m]")
    ax.set_title("Building Heights")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] saved → {save_path}")
