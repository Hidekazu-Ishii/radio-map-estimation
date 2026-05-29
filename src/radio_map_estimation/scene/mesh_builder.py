"""
dem / tran / wtr の parquet からローカル座標の trimesh を生成する

役割
----
1. dem.parquet  → DEM trimesh (build_dem_mesh)
2. tran.parquet → 道路 trimesh (build_tran_mesh) 、頂点 z を DEM 補間
3. wtr.parquet  → 水面 trimesh (build_wtr_mesh) 、頂点 z を DEM 補間

設計方針
--------
- parquet_loader で GeoDataFrame を取得し、mesh_utils で trimesh に変換する
- tran / wtr は PLATEAU 仕様上 surfaces.z=0 固定のため、
  dem_mesh による補間を必須引数とし、z=0 のまま残るケースを排除する
- surfaces=None の行は警告ログを出してスキップする
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import trimesh

from radio_map_estimation.scene.mesh_utils import interpolate_ground_z, surfaces_to_trimesh
from radio_map_estimation.scene.parquet_loader import load_filtered
from radio_map_estimation.scene.schema import AreaSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 共通: GeoDataFrame の surfaces 列から trimesh を生成
# ---------------------------------------------------------------------------


def _build_mesh_from_gdf(
    parquet_path: Path,
    area_spec: AreaSpec,
    label: str,
) -> trimesh.Trimesh | None:
    """
    parquet を読み込み、surfaces 列から trimesh を生成する共通処理

    surfaces は JSON 文字列 (dem / tran / wtr) または Python オブジェクト (bldg)
    各行が 1 サーフェス (dem) または複数サーフェス (tran / wtr) に対応する

    Parameters
    ----------
    parquet_path : 対象 parquet のパス
    area_spec    : AreaSpec (ox, oy として bbox_xmin/ymin を使用)
    label        : ログ出力用ラベル ("dem" / "tran" / "wtr")

    Returns
    -------
    trimesh.Trimesh | None
    """
    gdf = load_filtered(parquet_path, area_spec)
    if gdf.empty:
        logger.warning("%s: no records in bbox, mesh not generated.", label)
        return None

    ox, oy = area_spec.bbox_xmin, area_spec.bbox_ymin
    meshes: list[trimesh.Trimesh] = []

    for _, row in gdf.iterrows():
        raw = row.get("surfaces")
        if raw is None:
            logger.warning("%s: surfaces=None, skipping row.", label)
            continue

        # JSON 文字列の場合はデシリアライズ
        surfaces = json.loads(raw) if isinstance(raw, str) else raw

        # surfaces の構造を判定して surfaces_to_trimesh の入力形式に統一する
        # dem  : [[lon, lat, z], ...]          1行 = 1サーフェス → [[...]] でラップ
        # tran / wtr: [[[lon, lat, z], ...], ...] 1行 = 複数サーフェス → そのまま渡す
        if surfaces and isinstance(surfaces[0][0], (int, float)):
            surfaces = [surfaces]

        mesh = surfaces_to_trimesh(surfaces, ox, oy)
        if mesh is not None:
            meshes.append(mesh)

    if not meshes:
        logger.warning("%s: all rows failed to generate mesh.", label)
        return None

    result = trimesh.util.concatenate(meshes)
    logger.info(
        "%s mesh: %d vertices, %d faces, z=[%.1f, %.1f] m",
        label,
        len(result.vertices),
        len(result.faces),
        result.bounds[0, 2],
        result.bounds[1, 2],
    )
    return result


# ---------------------------------------------------------------------------
# DEM
# ---------------------------------------------------------------------------


def build_dem_mesh(
    dem_parquet: Path,
    area_spec: AreaSpec,
) -> trimesh.Trimesh | None:
    """
    dem.parquet からローカル座標の DEM trimesh を生成する

    dem の surfaces は JSON 文字列 [[lon, lat, z], ...]
    各行が TIN の 1 三角形 (通常 4 点、閉じたリング) に対応する

    Parameters
    ----------
    dem_parquet : dem.parquet のパス
    area_spec   : AreaSpec

    Returns
    -------
    trimesh.Trimesh | None
    """
    return _build_mesh_from_gdf(dem_parquet, area_spec, label="dem")


# ---------------------------------------------------------------------------
# 道路
# ---------------------------------------------------------------------------


def build_tran_mesh(
    tran_parquet: Path,
    area_spec: AreaSpec,
    dem_mesh: trimesh.Trimesh,
) -> trimesh.Trimesh | None:
    """
    tran.parquet からローカル座標の道路 trimesh を生成する

    PLATEAU 仕様上 surfaces.z=0 固定のため、dem_mesh による DEM 補間を必須とする
    全頂点の z を interpolate_ground_z で上書きすることで z=0 残留を排除する

    Parameters
    ----------
    tran_parquet : tran.parquet のパス
    area_spec    : AreaSpec
    dem_mesh     : DEM trimesh (必須、z 補間に使用)

    Returns
    -------
    trimesh.Trimesh | None
    """
    result = _build_mesh_from_gdf(tran_parquet, area_spec, label="tran")
    if result is None:
        return None

    for i, (x, y, _) in enumerate(result.vertices):
        result.vertices[i, 2] = interpolate_ground_z(x, y, dem_mesh)

    assert np.all(np.isfinite(result.vertices[:, 2])), (
        "build_tran_mesh: non-finite z detected after DEM interpolation"
    )

    logger.info(
        "tran mesh: z adjusted to DEM elevation, z=[%.1f, %.1f] m",
        result.vertices[:, 2].min(),
        result.vertices[:, 2].max(),
    )
    return result


# ---------------------------------------------------------------------------
# 水面
# ---------------------------------------------------------------------------


def build_wtr_mesh(
    wtr_parquet: Path,
    area_spec: AreaSpec,
    dem_mesh: trimesh.Trimesh,
) -> trimesh.Trimesh | None:
    """
    wtr.parquet からローカル座標の水面 trimesh を生成する

    PLATEAU 仕様上 surfaces.z=0 固定のため、dem_mesh による DEM 補間を必須とする
    全頂点の z を interpolate_ground_z で上書きすることで z=0 残留を排除する

    Parameters
    ----------
    wtr_parquet : wtr.parquet のパス
    area_spec   : AreaSpec
    dem_mesh    : DEM trimesh (必須、z 補間に使用)

    Returns
    -------
    trimesh.Trimesh | None
    """
    result = _build_mesh_from_gdf(wtr_parquet, area_spec, label="wtr")
    if result is None:
        return None

    for i, (x, y, _) in enumerate(result.vertices):
        result.vertices[i, 2] = interpolate_ground_z(x, y, dem_mesh)

    assert np.all(np.isfinite(result.vertices[:, 2])), (
        "build_wtr_mesh: non-finite z detected after DEM interpolation"
    )

    logger.info(
        "wtr mesh: z adjusted to DEM elevation, z=[%.1f, %.1f] m",
        result.vertices[:, 2].min(),
        result.vertices[:, 2].max(),
    )
    return result
