"""
座標変換・三角形分割・標高補間の共通ユーティリティ

複数モジュールで重複していた以下のロジックを集約する:
  - EPSG:6668 → EPSG:6677 → ローカル座標への変換
  - 3D サーフェスリストから trimesh への変換
  - DEM メッシュからの標高補間

設計方針
--------
- 純粋関数のみ (I/O なし、副作用なし)
- 全関数は surfaces を list[list[float]] で受け取る
  bldg は Python オブジェクトそのまま、dem/tran/wtr は json.loads() 後に渡す
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

_INPUT_CRS = "EPSG:6668"
_PROJ_CRS = "EPSG:6677"

# surfaces の型エイリアス: [[lon, lat, z], ...]
Surfaces = list[list[float]]


# ---------------------------------------------------------------------------
# 座標変換
# ---------------------------------------------------------------------------


def _make_transformer():
    """EPSG:6668 → EPSG:6677 変換器を生成する"""
    import pyproj

    return pyproj.Transformer.from_crs(_INPUT_CRS, _PROJ_CRS, always_xy=True)


def geo_to_local(
    lons: list[float],
    lats: list[float],
    zs: list[float],
    ox: float,
    oy: float,
) -> np.ndarray:
    """
    地理座標 (lon, lat, z) → ローカル座標 (x, y, z) に変換する

    Parameters
    ----------
    lons, lats : EPSG:6668 の経緯度 [deg]
    zs         : 標高 [m] (そのまま保持)
    ox, oy     : ローカル座標原点 (area_spec.bbox_xmin/ymin)

    Returns
    -------
    (N, 3) ndarray: ローカル座標 [m]
    """
    transformer = _make_transformer()
    xs, ys = transformer.transform(lons, lats)
    return np.array(
        [(x - ox, y - oy, z) for x, y, z in zip(xs, ys, zs, strict=False)],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# 三角形分割
# ---------------------------------------------------------------------------


def _triangulate_flat(verts: np.ndarray) -> trimesh.Trimesh | None:
    """
    水平面 (z が一定) を xy 投影で earcut 分割する

    Parameters
    ----------
    verts : (N, 3) 頂点配列 (閉じ点除去済み)
    """
    from trimesh.creation import triangulate_polygon

    try:
        poly_2d = Polygon([(v[0], v[1]) for v in verts])
        if not poly_2d.is_valid:
            poly_2d = poly_2d.buffer(0)
        if poly_2d.is_empty:
            return None

        verts_2d, faces = triangulate_polygon(poly_2d)
        verts_3d = np.zeros((len(verts_2d), 3))
        verts_3d[:, :2] = verts_2d
        verts_3d[:, 2] = float(np.mean(verts[:, 2]))
        return trimesh.Trimesh(vertices=verts_3d, faces=faces, process=False)
    except Exception:
        return None


def _triangulate_vertical(verts: np.ndarray) -> trimesh.Trimesh | None:
    """
    側面・屋根面 (z が変化) を fan triangulation で分割する

    N 頂点のポリゴン → N-2 枚の三角形 (頂点 0 を基点に展開)
    """
    try:
        n = len(verts)
        faces = np.array(
            [[0, i, i + 1] for i in range(1, n - 1)],
            dtype=np.int32,
        )
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    except Exception:
        return None


def triangulate_surface(verts: np.ndarray) -> trimesh.Trimesh | None:
    """
    3D ポリゴン頂点を面の向き (水平 or 傾斜) に応じて三角形分割する

    z 範囲が 1mm 未満 → 水平面 (earcut)
    それ以外          → 傾斜面 (fan triangulation)

    Parameters
    ----------
    verts : (N, 3) 頂点配列 (閉じ点除去済み)
    """
    if len(verts) < 3:
        return None
    z_range = float(np.max(verts[:, 2]) - np.min(verts[:, 2]))
    if z_range < 1e-3:
        return _triangulate_flat(verts)
    return _triangulate_vertical(verts)


# ---------------------------------------------------------------------------
# surfaces → trimesh
# ---------------------------------------------------------------------------


def surfaces_to_trimesh(
    surfaces: list[Surfaces],
    ox: float,
    oy: float,
) -> trimesh.Trimesh | None:
    """
    3D サーフェスリストをローカル座標の trimesh に変換する

    bldg・dem・tran・wtr のいずれにも使用できる共通変換関数
    dem/tran/wtr は json.loads() 後に渡すこと

    Parameters
    ----------
    surfaces : [[lon, lat, z], ...] のリスト (複数サーフェス)
    ox, oy   : ローカル座標原点 (area_spec.bbox_xmin/ymin)

    Returns
    -------
    trimesh.Trimesh | None (全サーフェスが変換失敗の場合 None)
    """
    meshes: list[trimesh.Trimesh] = []

    for surface in surfaces:
        if len(surface) < 3:
            continue

        lons = [pt[0] for pt in surface]
        lats = [pt[1] for pt in surface]
        zs = [pt[2] for pt in surface]

        verts = geo_to_local(lons, lats, zs, ox, oy)

        # 閉じ点を除去 (GML リングは始点 == 終点)
        if len(verts) > 1 and np.allclose(verts[0], verts[-1]):
            verts = verts[:-1]
        if len(verts) < 3:
            continue

        mesh = triangulate_surface(verts)
        if mesh is not None:
            meshes.append(mesh)

    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


# ---------------------------------------------------------------------------
# DEM 標高補間
# ---------------------------------------------------------------------------


def interpolate_ground_z(
    x: float,
    y: float,
    dem_mesh: trimesh.Trimesh,
) -> float:
    """
    DEM メッシュから (x, y) の標高をレイキャストで取得する

    真上 (z=1000m) から下向きにレイを飛ばし、DEM との交点の z を返す
    境界付近のレイミスを避けるため、x/y を DEM 範囲に 0.1m のマージンでクランプする

    レイキャストが失敗した場合は DEM 頂点の最近傍 z を返す
    これにより z が未補間・0 のまま残るケースを排除する

    Parameters
    ----------
    x, y     : ローカル座標 [m]
    dem_mesh : DEM trimesh (必須、None 非許容)

    Returns
    -------
    float
        補間した標高 [m]必ず有限値を返す (最近傍フォールバック保証)
    """
    margin = 0.1
    x_clamped = float(np.clip(x, dem_mesh.bounds[0, 0] + margin, dem_mesh.bounds[1, 0] - margin))
    y_clamped = float(np.clip(y, dem_mesh.bounds[0, 1] + margin, dem_mesh.bounds[1, 1] - margin))

    ray_origins = np.array([[x_clamped, y_clamped, 1000.0]])
    ray_directions = np.array([[0.0, 0.0, -1.0]])

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(dem_mesh)
    locations, _, _ = intersector.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False,
    )

    if len(locations) > 0:
        z = float(locations[0, 2])
        assert np.isfinite(z), f"interpolate_ground_z: ray cast returned non-finite z={z}"
        return z

    # フォールバック: 最近傍 DEM 頂点の z
    verts = dem_mesh.vertices
    dists = np.hypot(verts[:, 0] - x_clamped, verts[:, 1] - y_clamped)
    z = float(verts[np.argmin(dists), 2])
    assert np.isfinite(z), f"interpolate_ground_z: nearest vertex z is non-finite z={z}"
    logger.debug(
        "interpolate_ground_z: ray miss at (%.2f, %.2f), fallback to nearest vertex z=%.3f",
        x,
        y,
        z,
    )
    return z
