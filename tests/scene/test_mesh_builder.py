"""
mesh_builder モジュールのテスト.
- extrude_polygon_to_obj  : OBJ 文字列生成
- build_building_meshes   : BuildingData → OBJ リスト
- build_ground_mesh       : 地面メッシュ生成
- save_meshes_to_obj      : ファイル保存
"""

import re

import geopandas as gpd
import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

from src.radio_map_estimation.scene.mesh_builder import (
    build_building_meshes,
    build_ground_mesh,
    extrude_polygon_to_obj,
    save_meshes_to_obj,
)
from src.radio_map_estimation.scene.schema import AreaSpec, BuildingData

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unit_square() -> Polygon:
    """1m x 1m の正方形ポリゴン."""
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


@pytest.fixture
def area_spec() -> AreaSpec:
    """100m x 100m エリア仕様."""
    return AreaSpec(
        center_lat=35.0,
        center_lon=139.0,
        area_size_m=100.0,
        cell_size_m=10.0,
        crs="EPSG:3857",
        bbox_xmin=0.0,
        bbox_ymin=0.0,
        bbox_xmax=100.0,
        bbox_ymax=100.0,
    )


@pytest.fixture
def building_data(area_spec: AreaSpec) -> BuildingData:
    """1 棟だけ含む最小 BuildingData."""
    poly = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
    gdf = gpd.GeoDataFrame({"geometry": [poly], "height_m": [15.0]}, crs="EPSG:3857")
    mask = np.zeros((10, 10), dtype=bool)
    heights = np.zeros((10, 10), dtype=float)
    return BuildingData(
        gdf=gdf,
        building_mask=mask,
        building_heights=heights,
        area_spec=area_spec,
    )


# ---------------------------------------------------------------------------
# extrude_polygon_to_obj
# ---------------------------------------------------------------------------


class TestExtrudePolygonToObj:
    def _count(self, obj: str, prefix: str) -> int:
        return sum(1 for line in obj.splitlines() if line.startswith(prefix))

    def test_vertex_count(self, unit_square):
        """底面 n 頂点 + 天面 n 頂点 = 2n 個の v 行が出力されること."""
        n = len(unit_square.exterior.coords) - 1  # 始点=終点を除く
        obj = extrude_polygon_to_obj(unit_square, height=10.0, simplify_tolerance=0.0)
        assert self._count(obj, "v ") == 2 * n

    def test_top_face_height(self, unit_square):
        """天面頂点の z 座標が height と一致すること."""
        height = 7.5
        obj = extrude_polygon_to_obj(unit_square, height=height, simplify_tolerance=0.0)
        z_values = [float(line.split()[3]) for line in obj.splitlines() if line.startswith("v ")]
        assert max(z_values) == pytest.approx(height)

    def test_face_lines_exist(self, unit_square):
        """側面・天面の f 行が最低 1 行以上あること."""
        obj = extrude_polygon_to_obj(unit_square, height=5.0, simplify_tolerance=0.0)
        assert self._count(obj, "f ") >= 1

    @pytest.mark.parametrize("height", [1.0, 10.0, 100.0])
    def test_various_heights(self, unit_square, height):
        """任意の高さで OBJ 文字列が生成されること."""
        obj = extrude_polygon_to_obj(unit_square, height=height, simplify_tolerance=0.0)
        assert len(obj) > 0


# ---------------------------------------------------------------------------
# build_building_meshes
# ---------------------------------------------------------------------------


class TestBuildBuildingMeshes:
    def test_returns_nonempty_list(self, building_data):
        """有効な建物があれば非空リストが返ること."""
        result = build_building_meshes(building_data, min_height_m=3.0)
        assert len(result) >= 1

    def test_min_height_filter(self, area_spec):
        """min_height_m 未満の建物がスキップされること."""
        poly = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
        gdf = gpd.GeoDataFrame({"geometry": [poly], "height_m": [1.0]}, crs="EPSG:3857")
        data = BuildingData(
            gdf=gdf,
            building_mask=np.zeros((10, 10), dtype=bool),
            building_heights=np.zeros((10, 10)),
            area_spec=area_spec,
        )
        with pytest.raises(RuntimeError, match="有効な建物メッシュ"):
            build_building_meshes(data, min_height_m=3.0)

    def test_each_element_is_obj_string(self, building_data):
        """各要素が 'v ' または 'f ' を含む OBJ 文字列であること."""
        for obj_str in build_building_meshes(building_data):
            assert "v " in obj_str or "f " in obj_str


# ---------------------------------------------------------------------------
# build_ground_mesh
# ---------------------------------------------------------------------------


class TestBuildGroundMesh:
    def test_returns_trimesh(self, building_data):
        """trimesh.Trimesh が返ること."""
        mesh = build_ground_mesh(building_data)
        assert isinstance(mesh, trimesh.Trimesh)

    def test_all_vertices_at_z0(self, building_data):
        """地面メッシュの全頂点が z=0 にあること."""
        mesh = build_ground_mesh(building_data)
        assert (mesh.vertices[:, 2] == 0.0).all()

    def test_ground_size_matches_area(self, building_data):
        """地面メッシュの xy 範囲が area_size_m と一致すること."""
        mesh = build_ground_mesh(building_data)
        s = building_data.area_spec.area_size_m
        assert mesh.vertices[:, 0].max() == pytest.approx(s)
        assert mesh.vertices[:, 1].max() == pytest.approx(s)


# ---------------------------------------------------------------------------
# save_meshes_to_obj
# ---------------------------------------------------------------------------


class TestSaveMeshesToObj:
    def test_obj_files_created(self, building_data, tmp_path):
        """building_XXXXX.obj ファイルが生成されること."""
        obj_strs = build_building_meshes(building_data)
        ground = build_ground_mesh(building_data)
        paths, _ = save_meshes_to_obj(obj_strs, ground, tmp_path)

        assert len(paths) == len(obj_strs)
        for p in paths:
            assert p.exists()

    def test_ground_ply_created(self, building_data, tmp_path):
        """ground.ply ファイルが生成されること."""
        obj_strs = build_building_meshes(building_data)
        ground = build_ground_mesh(building_data)
        _, ground_ply = save_meshes_to_obj(obj_strs, ground, tmp_path)

        assert ground_ply.exists()
        assert ground_ply.suffix == ".ply"

    def test_obj_filename_pattern(self, building_data, tmp_path):
        """ファイル名が building_NNNNN.obj 形式であること."""
        obj_strs = build_building_meshes(building_data)
        ground = build_ground_mesh(building_data)
        paths, _ = save_meshes_to_obj(obj_strs, ground, tmp_path)

        pattern = re.compile(r"building_\d{5}\.obj")
        for p in paths:
            assert pattern.match(p.name)
