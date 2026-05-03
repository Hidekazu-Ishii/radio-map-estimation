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

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import box as shapely_box

# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AreaSpec:
    """シミュレーション対象エリアの仕様.

    ローカル座標系: bbox_m の左下 (xmin, ymin) を原点とする.
    area_size_m = xmax - xmin = ymax - ymin (正方形を前提とする).
    grid_size は area_size_m / cell_size_m から導出する (冗長なので保持しない).
    """

    center_lat: float
    center_lon: float
    area_size_m: float  # エリア一辺 [m]
    cell_size_m: float  # グリッドセル一辺 [m]
    crs: str  # 投影座標系 (例: "EPSG:32654")
    bbox_xmin: float
    bbox_ymin: float
    bbox_xmax: float
    bbox_ymax: float

    @property
    def bbox_m(self) -> tuple[float, float, float, float]:
        return (self.bbox_xmin, self.bbox_ymin, self.bbox_xmax, self.bbox_ymax)

    @property
    def grid_size(self) -> int:
        """グリッド一辺のセル数 (area_size_m / cell_size_m から導出)."""
        return int(self.area_size_m / self.cell_size_m)


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildingData:
    """建物情報をまとめたデータ構造.

    Attributes
    ----------
    gdf : GeoDataFrame
        メートル座標系 (投影座標系) の建物ポリゴン.
        カラムは geometry, height_m のみ.
        ローカル座標系への変換は to_local_gdf() で行う.
    building_mask : ndarray of shape (grid_size, grid_size), dtype bool
        建物が存在するセルが True.
        [row, col] = [y_idx, x_idx], [0,0] が左下.
    building_heights : ndarray of shape (grid_size, grid_size), dtype float
        セルの最大建物高さ [m]. 建物がない場合は 0.0.
    area_spec : AreaSpec
        エリア仕様 (座標系・グリッド定義を含む).
    """

    gdf: gpd.GeoDataFrame
    building_mask: np.ndarray
    building_heights: np.ndarray
    area_spec: AreaSpec

    def to_local_gdf(self) -> gpd.GeoDataFrame:
        """投影座標系 → ローカル座標系 (bbox左下を原点) に平行移動した GeoDataFrame を返す."""
        gdf_local = self.gdf.copy()
        gdf_local["geometry"] = gdf_local["geometry"].translate(
            xoff=-self.area_spec.bbox_xmin,
            yoff=-self.area_spec.bbox_ymin,
        )
        return gdf_local


# ---------------------------------------------------------------------------
# 純粋関数群
# ---------------------------------------------------------------------------


def fetch_buildings_osm(
    center_lat: float,
    center_lon: float,
    area_size_m: float,
    cell_size_m: float,
    meters_per_level: float,
    default_building_height_m: float,
    building_type_levels: dict[str, float],
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
    default_building_height_m : float
        タグが欠損している場合のデフォルト高さ [m].

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

    height_m = _estimate_heights(gdf_proj, meters_per_level, default_building_height_m, building_type_levels)

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
    default_height_m: float,
    building_type_levels: dict[str, float],
) -> np.ndarray:
    """建物高さを推定して ndarray で返す (内部関数)."""
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

    # 3. building タグ(用途)
    n_from_type = 0
    if "building" in gdf.columns:
        building_types = gdf["building"].astype(str).str.strip().str.lower()
        from_type = building_types.map(building_type_levels).to_numpy(dtype=float)
        from_type_m = from_type * meters_per_level
        mask = np.isnan(height_m) & np.isfinite(from_type_m)
        height_m = np.where(mask, from_type_m, height_m)
        n_from_type = int(mask.sum())
    print(f"  building type:     {n_from_type} buildings")

    # 4. デフォルト値
    n_default = int(np.isnan(height_m).sum())
    height_m = np.where(np.isnan(height_m), default_height_m, height_m)
    print(f"  default height:    {n_default} buildings")

    return height_m


def _rasterize_buildings(
    gdf_proj: gpd.GeoDataFrame,
    bbox_m: tuple[float, float, float, float],
    cell_size_m: float,
    area_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    建物ポリゴンをグリッドにラスタライズする (内部関数).

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
                if poly.intersection(cell_box).area > 1e-6:
                    building_mask[r, c] = True
                    building_heights[r, c] = max(building_heights[r, c], h)

    return building_mask, building_heights


def save_building_data(data: BuildingData, output_dir: Path) -> None:
    """BuildingData を npz ファイルに保存する."""
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = data.area_spec
    np.savez(
        output_dir / "building_data.npz",
        building_mask=data.building_mask,
        building_heights=data.building_heights,
        center_lat=spec.center_lat,
        center_lon=spec.center_lon,
        area_size_m=spec.area_size_m,
        cell_size_m=spec.cell_size_m,
        bbox_xmin=spec.bbox_xmin,
        bbox_ymin=spec.bbox_ymin,
        bbox_xmax=spec.bbox_xmax,
        bbox_ymax=spec.bbox_ymax,
    )
    print(f"[save] building_data.npz → {output_dir}")


def plot_building_data(data: BuildingData, save_path: Path) -> None:
    """BuildingData を可視化して保存する."""
    spec = data.area_spec
    grid_size = spec.grid_size
    s = spec.area_size_m

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    img = np.ones((grid_size, grid_size, 3))
    img[data.building_mask] = [0.2, 0.2, 0.6]
    ax.imshow(img, origin="lower", extent=(0, s, 0, s))
    ax.set_title(
        f"OSM Building Map\n"
        f"center=({spec.center_lat:.4f}, {spec.center_lon:.4f}), "
        f"{s:.0f}m x {s:.0f}m, cell={spec.cell_size_m:.1f}m"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax = axes[1]
    im = ax.imshow(
        data.building_heights,
        origin="lower",
        extent=(0, s, 0, s),
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
