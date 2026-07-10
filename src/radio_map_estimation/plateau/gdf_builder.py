"""
PLATEAU CityGML → GeoDataFrame 変換

各パーサーの出力 (イテレータ) を受け取り GeoDataFrame を構築する
保存・ダウンロードはここでは行わない (plateau_loader.py の責務)

出力 GeoDataFrame スキーマ
--------------------------
bldg / dem / tran 共通:
    geometry : Polygon  — 底面フットプリント投影 (EPSG:6668)
    surfaces : object   — 3D サーフェス list[list[(lon,lat,z)]]
                          bldg のみ None あり (measured_height 欠損の LOD1 建物)
"""

import itertools
import json
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from .bldg_parser import iter_buildings_from_zip
from .dem_parser import iter_dem_surfaces_from_zip
from .luse_parser import iter_water_surfaces_from_zip
from .tran_parser import iter_road_surfaces_from_zip

logger = logging.getLogger(__name__)

_INPUT_CRS = "EPSG:6668"

# surface ベースの GDF (DEM・道路) で使う共通ロジック
Surface3D = list[tuple[float, float, float]]


def _build_surface_geodataframe(
    surfaces: list[Surface3D],
    label: str,
) -> gpd.GeoDataFrame:
    """
    3D サーフェスリストから底面投影 Polygon を geometry とする GeoDataFrame を構築する

    Parameters
    ----------
    surfaces : 3D 座標リストのリスト (各要素が 1 ポリゴン)
    label    : ログ用ラベル ("DEM" / "tran" など)
    """
    records = [
        {
            "geometry": Polygon([(lon, lat) for lon, lat, _ in surface]),
            "surfaces": json.dumps(surface),
        }
        for surface in surfaces
        if len(surface) >= 3
    ]
    logger.info("%s: built %d polygons", label, len(records))

    if not records:
        return gpd.GeoDataFrame(columns=["geometry", "surfaces"], geometry="geometry", crs=_INPUT_CRS)

    return gpd.GeoDataFrame(pd.DataFrame(records), geometry="geometry", crs=_INPUT_CRS)


def _lod1_to_surfaces(
    footprint: "Polygon",
    z_base: float,
    height: float,
) -> list[list[tuple[float, float, float]]]:
    """
    LOD1 底面フットプリット + 底面 z + 高さ から直方体サーフェスを構築する

    底面・上面・側面の全ポリゴンを surfaces と同じ形式で返す

    Parameters
    ----------
    footprint : 底面 Polygon (2D, lon/lat)
    z_base    : 底面高さ [m]
    height    : 建物高さ [m]

    Returns
    -------
    list[list[(lon, lat, z)]] — 各面のポリゴン頂点リスト (閉じたリング)
    """
    z_top = z_base + height
    coords = list(footprint.exterior.coords)  # 閉じたリング (始点 == 終点)

    bottom = [(lon, lat, z_base) for lon, lat in reversed(coords)]
    top = [(lon, lat, z_top) for lon, lat in coords]
    sides = [
        [
            (lon0, lat0, z_base),
            (lon1, lat1, z_base),
            (lon1, lat1, z_top),
            (lon0, lat0, z_top),
            (lon0, lat0, z_base),  # 閉じる
        ]
        for (lon0, lat0), (lon1, lat1) in itertools.pairwise(coords)
    ]
    return [bottom, top, *sides]


def build_bldg_geodataframe(zip_path: Path) -> gpd.GeoDataFrame:
    """
    PLATEAU CityGML ZIP から建物 GeoDataFrame を構築する

    LOD2 あり         : surfaces にそのまま格納
    LOD2 なし・LOD1 あり: lod1_footprint の (z_base, z_top) から
                          直方体サーフェスを構築して surfaces に格納

    Returns
    -------
    GeoDataFrame
        columns  : surfaces, geometry
        geometry : LOD1 底面フットプリント (空間インデックス用), CRS: EPSG:6668
    """
    logger.info("Parsing buildings CityGML: %s", zip_path)

    records = [
        rec
        for rec in iter_buildings_from_zip(zip_path)
        if rec["lod2_surfaces"] is not None or rec["lod1_footprint"] is not None
    ]
    n_lod2 = sum(1 for r in records if r["lod2_surfaces"] is not None)
    logger.info(
        "Parsed %d buildings (LOD2: %d, LOD1 only: %d)",
        len(records),
        n_lod2,
        len(records) - n_lod2,
    )

    # LOD1 建物: footprint + z_base + z_top → 直方体サーフェスに変換
    for rec in records:
        if rec["lod2_surfaces"] is None and rec["lod1_footprint"] is not None:
            footprint, z_base, z_top = rec["lod1_footprint"]
            rec["surfaces"] = _lod1_to_surfaces(footprint, z_base, z_top - z_base)
        else:
            rec["surfaces"] = rec.pop("lod2_surfaces")

    # geometry 用に lod1_footprint を Polygon のみに戻す (z_base, z_top は不要)
    for rec in records:
        if rec["lod1_footprint"] is not None:
            rec["lod1_footprint"] = rec["lod1_footprint"][0]

    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(df, geometry="lod1_footprint", crs=_INPUT_CRS)
    gdf = gdf.rename_geometry("geometry")
    return gdf


def build_dem_geodataframe(zip_path: Path) -> gpd.GeoDataFrame:
    """PLATEAU CityGML ZIP から地形 (DEM) GeoDataFrame を構築する"""
    logger.info("Parsing DEM CityGML: %s", zip_path)
    surfaces = list(iter_dem_surfaces_from_zip(zip_path))
    return _build_surface_geodataframe(surfaces, "DEM")


def build_tran_geodataframe(zip_path: Path) -> gpd.GeoDataFrame:
    """PLATEAU CityGML ZIP から道路 (tran) GeoDataFrame を構築する"""
    logger.info("Parsing tran CityGML: %s", zip_path)
    surfaces = list(iter_road_surfaces_from_zip(zip_path))
    return _build_surface_geodataframe(surfaces, "tran")


def build_wtr_geodataframe(zip_path: Path) -> gpd.GeoDataFrame:
    """PLATEAU CityGML ZIP から水部エリア (luse orgLandUse=7000) GeoDataFrame を構築する"""
    logger.info("Parsing luse (water) CityGML: %s", zip_path)
    surfaces = list(iter_water_surfaces_from_zip(zip_path))
    return _build_surface_geodataframe(surfaces, "wtr")
