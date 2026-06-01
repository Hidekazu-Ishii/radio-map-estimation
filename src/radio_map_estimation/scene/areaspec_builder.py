"""
origin (地理座標) から AreaSpec を構築する

役割
----
origin_lon/lat + area_size_m → EPSG:6677 投影変換 → bbox 計算 → AreaSpec

ローカル座標系:
    origin (エリア左下隅) の投影座標 (origin_proj_x, origin_proj_y) を原点 (0, 0) とする
    有効エリア x, y ∈ [0, area_size_m]
    マージン部分 x, y ∈ [-margin, 0) および (area_size_m, area_size_m + margin]

bbox の定義:
    margin      = area_size_m / 5
    bbox_xmin   = origin_proj_x - margin
    bbox_xmax   = origin_proj_x + area_size_m + margin
    bbox_ymin   = origin_proj_y - margin
    bbox_ymax   = origin_proj_y + area_size_m + margin
"""

from __future__ import annotations

import logging

import geopandas as gpd
from shapely.geometry import Point

from radio_map_estimation.scene.schema import AreaSpec

logger = logging.getLogger(__name__)

_INPUT_CRS = "EPSG:6668"
_PROJ_CRS = "EPSG:6677"


def build_area_spec(
    origin_lon: float,
    origin_lat: float,
    area_size_m: float,
) -> AreaSpec:
    """
    エリア左下隅の地理座標と一辺の長さから AreaSpec を構築する

    Parameters
    ----------
    origin_lon  : エリア左下隅の経度 [deg] (EPSG:6668)
    origin_lat  : エリア左下隅の緯度 [deg] (EPSG:6668)
    area_size_m : エリアの一辺の長さ [m] (正方形)

    Returns
    -------
    AreaSpec
        origin_proj_x/y : ローカル座標の原点 (geo_to_local の ox, oy に使用)
        bbox_xmin/ymin  : origin より margin だけ負方向に広げた値
    """
    origin_proj = gpd.GeoSeries([Point(origin_lon, origin_lat)], crs=_INPUT_CRS).to_crs(_PROJ_CRS).iloc[0]
    ox, oy = origin_proj.x, origin_proj.y  # type: ignore
    margin = area_size_m / 5

    bbox_xmin = ox - margin
    bbox_ymin = oy - margin
    bbox_xmax = ox + area_size_m + margin
    bbox_ymax = oy + area_size_m + margin

    logger.info(
        "AreaSpec: origin=(%.6f, %.6f) → proj=(%.2f, %.2f), bbox=[%.2f, %.2f, %.2f, %.2f]",
        origin_lon,
        origin_lat,
        ox,
        oy,
        bbox_xmin,
        bbox_ymin,
        bbox_xmax,
        bbox_ymax,
    )

    return AreaSpec(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        area_size_m=area_size_m,
        crs=_PROJ_CRS,
        origin_proj_x=ox,
        origin_proj_y=oy,
        bbox_xmin=bbox_xmin,
        bbox_ymin=bbox_ymin,
        bbox_xmax=bbox_xmax,
        bbox_ymax=bbox_ymax,
    )
