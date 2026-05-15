# src/radio_map_estimation/scene/mesh_builder.py
"""
BuildingData → 3D メッシュ (trimesh) → PLY ファイル群

建物: OBJ フォーマット (側面を quad で構成し縮退三角形を回避)
地面: PLY フォーマット (trimesh で生成)
"""

from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import triangulate as shapely_triangulate

from .schema import BuildingData


def extrude_polygon_to_obj(
    poly: Polygon,
    height: float,
    simplify_tolerance: float = 1.0,
) -> str:
    """
    2D ポリゴンを押し出した建物メッシュを OBJ 形式の文字列で返す.

    側面は quad (四角形) で構成する.
    三角分割を避けることで縮退三角形による法線の不安定化を防ぐ.
    """
    if simplify_tolerance > 0.0:
        simplified = poly.simplify(tolerance=simplify_tolerance, preserve_topology=True)
        if not isinstance(simplified, Polygon):
            raise ValueError(f"単純化後にPolygon以外: {type(simplified)}")
        poly = simplified

    coords = np.array(poly.exterior.coords)
    n = len(coords) - 1  # 始点=終点を除く
    coords = coords[:n]

    if n < 3:
        raise ValueError(f"縮退ポリゴン: 頂点数={n}")

    lines: list[str] = []

    # 底面頂点 (z=0): index 1..n
    for x, y in coords:
        lines.append(f"v {x:.6f} {y:.6f} 0.000000")
    # 天面頂点 (z=height): index n+1..2n
    for x, y in coords:
        lines.append(f"v {x:.6f} {y:.6f} {height:.6f}")

    # 側面: quad (四角形) で構成 (OBJ は 1-indexed)
    for i in range(n):
        j = (i + 1) % n
        # 反時計回り → 外向き法線
        lines.append(f"f {i + 1} {j + 1} {j + n + 1} {i + n + 1}")

    # 天面: Shapely で三角分割 (天面は三角形でも法線が安定)
    poly2d = Polygon(coords)
    for tri in shapely_triangulate(poly2d):
        tri_coords = np.array(tri.exterior.coords)[:3]
        idxs: list[int] = []
        for tc in tri_coords:
            dists = np.linalg.norm(coords - tc, axis=1)
            idxs.append(int(np.argmin(dists)) + n + 1)  # 天面: 1-indexed
        if len(set(idxs)) == 3:
            lines.append(f"f {idxs[0]} {idxs[1]} {idxs[2]}")

    return "\n".join(lines)


def build_building_meshes(
    building_data: BuildingData,
    min_height_m: float = 3.0,
    simplify_tolerance: float = 2.0,
) -> list[str]:
    """
    BuildingData から OBJ 文字列リストを生成する.

    to_local_gdf() が bbox 左下を原点とする座標変換を担う.
    建物ごとに個別 OBJ 文字列を返す (結合しない).

    Parameters
    ----------
    building_data : BuildingData
    min_height_m : float
        この高さ未満の建物はスキップする.
    simplify_tolerance : float
        ポリゴン単純化の許容誤差 [m]. 0.0 で単純化なし.

    Returns
    -------
    list[str]
        建物ごとの OBJ 文字列リスト.
    """
    gdf_local = building_data.to_local_gdf()

    obj_strings: list[str] = []
    for _, row in gdf_local.iterrows():
        geom = row.geometry
        height = float(row["height_m"])
        if height < min_height_m:
            continue
        polys: list[Polygon] = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        for poly in polys:
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
            try:
                obj_strings.append(extrude_polygon_to_obj(poly, height, simplify_tolerance))
            except Exception as e:
                print(f"[warn] skipped polygon: {e}")
                continue

    if not obj_strings:
        raise RuntimeError("有効な建物メッシュが1つも生成されませんでした.")

    print(f"[mesh] {len(obj_strings)} buildings generated.")
    return obj_strings


def build_ground_mesh(building_data: BuildingData) -> trimesh.Trimesh:
    """BuildingData から地面メッシュ (z=0 平面) を生成する."""
    s = building_data.area_spec.area_size_m
    vertices: np.ndarray = np.array(
        [[0.0, 0.0, 0.0], [s, 0.0, 0.0], [s, s, 0.0], [0.0, s, 0.0]],
        dtype=np.float64,
    )
    faces: np.ndarray = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def save_meshes_to_obj(
    building_obj_strings: list[str],
    ground_mesh: trimesh.Trimesh,
    mesh_dir: Path,
) -> tuple[list[Path], Path]:
    """
    建物 OBJ・地面 PLY ファイルを保存する.

    建物は building_00000.obj, building_00001.obj, ... として個別保存する.
    地面は ground.ply として保存する.

    Returns
    -------
    (building_objs, ground_ply) : (list[Path], Path)
    """
    mesh_dir.mkdir(parents=True, exist_ok=True)

    building_objs: list[Path] = []
    for i, obj_str in enumerate(building_obj_strings):
        obj_path = mesh_dir / f"building_{i:05d}.obj"
        obj_path.write_text(obj_str, encoding="utf-8")
        building_objs.append(obj_path)

    ground_ply = mesh_dir / "ground.ply"
    ground_mesh.export(str(ground_ply))

    print(f"[mesh] {len(building_objs)} buildings (OBJ) + ground (PLY) → {mesh_dir}")
    return building_objs, ground_ply
