"""
bldg / dem / tran / wtr の parquet を読み込み、AreaSpec の bbox でフィルタする

役割
----
parquet 読み込み + bbox フィルタ → 投影座標系 (EPSG:6677) の GeoDataFrame

設計方針
--------
- I/O のみを担う (メッシュ生成・座標変換は mesh_builder / building_extruder へ)
- bldg / dem / tran / wtr すべてに共通の単一インターフェースを提供する
- geometry は EPSG:6668 (地理座標) を前提とする

フィルタ戦略
------------
粗フィルタ (地理座標 bbox) → 投影変換 → 精フィルタ (投影座標 bbox) の 2 段階
粗フィルタは投影 bbox を地理座標に逆変換して使用する
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from radio_map_estimation.scene.schema import AreaSpec

logger = logging.getLogger(__name__)

_INPUT_CRS = "EPSG:6668"
_PROJ_CRS = "EPSG:6677"


def load_filtered(
    parquet_path: Path,
    area_spec: AreaSpec,
) -> gpd.GeoDataFrame:
    """
    parquet を読み込み、AreaSpec の bbox でフィルタした GeoDataFrame を返す

    bldg / dem / tran / wtr すべてに使用できる共通関数
    返却される GeoDataFrame の CRS は EPSG:6677 (投影座標系)

    Parameters
    ----------
    parquet_path : 対象 parquet のパス (bldg / dem / tran / wtr)
    area_spec    : AreaSpec (bbox_xmin/ymin/xmax/ymax を使用)

    Returns
    -------
    gpd.GeoDataFrame
        CRS: EPSG:6677、bbox 内のレコードのみ
        空の場合は空の GeoDataFrame を返す (例外は送出しない)
    """
    gdf = gpd.read_parquet(parquet_path)
    logger.info("Loaded %d records from %s", len(gdf), parquet_path.name)

    if gdf.empty:
        return gdf

    # 粗フィルタ: 投影 bbox を地理座標に逆変換して使用
    bbox_proj = box(
        area_spec.bbox_xmin,
        area_spec.bbox_ymin,
        area_spec.bbox_xmax,
        area_spec.bbox_ymax,
    )
    bbox_geo = gpd.GeoDataFrame(geometry=[bbox_proj], crs=_PROJ_CRS).to_crs(_INPUT_CRS).geometry.iloc[0]
    gdf_rough = gdf[gdf.geometry.intersects(bbox_geo)].copy()  # type: ignore
    logger.info("Rough filter: %d records", len(gdf_rough))

    if gdf_rough.empty:
        logger.warning("No records found in bbox for %s", parquet_path.name)
        return gdf_rough

    # 精フィルタ: 投影座標系に変換後、投影 bbox で確定フィルタ
    gdf_proj = gdf_rough.to_crs(_PROJ_CRS)
    gdf_proj = gdf_proj[gdf_proj.geometry.intersects(bbox_proj)].copy()
    logger.info("Final filter: %d records in projected bbox", len(gdf_proj))

    return gdf_proj.reset_index(drop=True)
