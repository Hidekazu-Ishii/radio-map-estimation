"""
投影座標系の建物 GeoDataFrame を 3D メッシュに変換する

役割
----
bldg.parquet → ローカル座標の建物 trimesh を生成する

変換戦略
--------
surfaces が有効値 → surfaces_to_trimesh で直接変換
surfaces が None  → 警告ログを出してスキップ (データは整備済みを前提とする)

bldg.parquet の surfaces スキーマ:
  LOD2 あり             : LOD2 全サーフェス (屋根形状を保持)
  LOD2 なし・LOD1 あり  : 底面・上面・側面の直方体サーフェス

z オフセット補正
----------------
bldg の surfaces.z は CityGML 実測値 (絶対標高) であり、DEM と同一測量由来だが
数 cm〜数十 cm のズレが生じうる建物が DEM に埋もれることを防ぐため、
建物ごとに底面 z を建物 centroid の DEM 補間値に合わせてオフセット補正する

    bldg_min_z = 建物メッシュ全頂点の z 最小値 (= 底面の絶対標高)
    dem_z      = 建物 centroid の DEM レイキャスト補間値
    offset     = dem_z - bldg_min_z
    全頂点の z += offset (建物全体を平行移動、形状は保持)

座標系
------
入力 : 投影座標系 (EPSG:6677) の geometry + EPSG:6668 の surfaces
出力 : ローカル座標系 (bbox 左下原点) の trimesh、z は DEM 基準に補正済み
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

from radio_map_estimation.scene.mesh_utils import interpolate_ground_z, surfaces_to_trimesh
from radio_map_estimation.scene.parquet_loader import load_filtered
from radio_map_estimation.scene.schema import AreaSpec

logger = logging.getLogger(__name__)


def _apply_dem_offset(
    mesh: trimesh.Trimesh,
    dem_mesh: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """
    建物メッシュの底面 z を DEM 補間値に合わせてオフセット補正する

    建物フットプリントの centroid における DEM 標高を取得し、
    建物全体を平行移動することで底面が DEM 地表面に接地するよう補正する
    建物の高さ (形状) は保持される

    Parameters
    ----------
    mesh     : ローカル座標の建物 trimesh (補正前)
    dem_mesh : DEM trimesh (z 補間に使用)

    Returns
    -------
    trimesh.Trimesh
        z オフセット補正済みの建物 trimesh
    """
    bldg_min_z = float(mesh.vertices[:, 2].min())
    centroid_x = float(mesh.vertices[:, 0].mean())
    centroid_y = float(mesh.vertices[:, 1].mean())
    dem_z = interpolate_ground_z(centroid_x, centroid_y, dem_mesh)

    offset = dem_z - bldg_min_z
    mesh.vertices[:, 2] += offset

    assert np.all(np.isfinite(mesh.vertices[:, 2])), (
        "_apply_dem_offset: non-finite z detected after offset correction"
    )

    logger.debug(
        "bldg offset: centroid=(%.2f, %.2f), bldg_min_z=%.3f, dem_z=%.3f, offset=%.3f",
        centroid_x,
        centroid_y,
        bldg_min_z,
        dem_z,
        offset,
    )
    return mesh


def build_bldg_footprint_mesh(
    bldg_parquet: Path,
    area_spec: AreaSpec,
) -> trimesh.Trimesh | None:
    """
    bldg.parquet の geometry (LOD1 底面フットプリント) から
    ローカル座標の 2D メッシュ (z=0 固定) を生成する

    建物マスク生成 (bldg_mask.py) での交差判定に使用する
    surfaces ではなく geometry (EPSG:6677 の Polygon) を使うため、
    LOD1/LOD2 問わず確実なフットプリントが得られる

    Parameters
    ----------
    bldg_parquet : bldg.parquet のパス
    area_spec    : AreaSpec

    Returns
    -------
    trimesh.Trimesh | None
        z=0 の平面メッシュ (全建物フットプリントの結合)
    """
    from shapely.affinity import translate as shapely_translate
    from trimesh.creation import triangulate_polygon

    gdf = load_filtered(bldg_parquet, area_spec)
    if gdf.empty:
        logger.warning("bldg_footprint: no records in bbox.")
        return None

    ox, oy = area_spec.origin_proj_x, area_spec.origin_proj_y
    meshes: list[trimesh.Trimesh] = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # 投影座標 → ローカル座標 (z=0 固定)
        geom_local = shapely_translate(geom, xoff=-ox, yoff=-oy)
        if not geom_local.is_valid:
            geom_local = geom_local.buffer(0)
        if geom_local.is_empty:
            continue

        try:
            verts_2d, faces = triangulate_polygon(geom_local)
            verts_3d = np.zeros((len(verts_2d), 3))
            verts_3d[:, :2] = verts_2d
            meshes.append(trimesh.Trimesh(vertices=verts_3d, faces=faces, process=False))
        except Exception as e:
            logger.debug("bldg_footprint: triangulation failed: %s", e)
            continue

    if not meshes:
        logger.warning("bldg_footprint: all rows failed.")
        return None

    result = trimesh.util.concatenate(meshes)
    logger.info(
        "bldg_footprint mesh: %d buildings, %d faces",
        len(meshes),
        len(result.faces),
    )
    return result


def build_bldg_mesh(
    bldg_parquet: Path,
    area_spec: AreaSpec,
    dem_mesh: trimesh.Trimesh | None = None,
) -> trimesh.Trimesh | None:
    """
    bldg.parquet からローカル座標の建物 trimesh を生成する

    dem_mesh が指定された場合、建物ごとに底面 z を DEM 補間値で補正する
    surfaces=None の行は警告ログを出してスキップする
    bldg の surfaces は Python オブジェクト (json.loads 不要)

    Parameters
    ----------
    bldg_parquet : bldg.parquet のパス
    area_spec    : AreaSpec
    dem_mesh     : DEM trimesh (指定時に z オフセット補正を適用)

    Returns
    -------
    trimesh.Trimesh | None
        全行が失敗した場合は None を返す
    """
    gdf = load_filtered(bldg_parquet, area_spec)
    if gdf.empty:
        logger.warning("bldg: no records in bbox, mesh not generated.")
        return None

    ox, oy = area_spec.origin_proj_x, area_spec.origin_proj_y
    meshes: list[trimesh.Trimesh] = []
    n_skipped = 0

    for _, row in gdf.iterrows():
        surfaces = row.get("surfaces")
        if surfaces is None:
            logger.warning("bldg: surfaces=None, skipping row.")
            n_skipped += 1
            continue

        mesh = surfaces_to_trimesh(surfaces, ox, oy)
        if mesh is None:
            n_skipped += 1
            continue

        if dem_mesh is not None:
            mesh = _apply_dem_offset(mesh, dem_mesh)

        meshes.append(mesh)

    if n_skipped > 0:
        logger.warning("bldg: %d rows skipped (surfaces=None or conversion failed).", n_skipped)

    if not meshes:
        logger.warning("bldg: all rows failed to generate mesh.")
        return None

    result = trimesh.util.concatenate(meshes)
    logger.info(
        "bldg mesh: %d buildings, %d vertices, %d faces, z=[%.1f, %.1f] m",
        len(meshes),
        len(result.vertices),
        len(result.faces),
        result.bounds[0, 2],
        result.bounds[1, 2],
    )
    return result
