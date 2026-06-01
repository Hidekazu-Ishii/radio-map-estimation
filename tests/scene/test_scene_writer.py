"""
scene_writer モジュールのテスト.
- write_mitsuba_xml : XML ファイル生成
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from src.radio_map_estimation.scene.scene_writer import write_mitsuba_xml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scene_dir(tmp_path: Path) -> Path:
    return tmp_path / "scene"


@pytest.fixture
def mesh_files(scene_dir: Path) -> tuple[list[Path], Path]:
    """ダミーの建物 OBJ・地面 PLY ファイルを用意する."""
    scene_dir.mkdir(parents=True)
    building_objs = [scene_dir / f"building_{i:05d}.obj" for i in range(3)]
    for p in building_objs:
        p.touch()
    ground_ply = scene_dir / "ground.ply"
    ground_ply.touch()
    return building_objs, ground_ply


# ---------------------------------------------------------------------------
# write_mitsuba_xml
# ---------------------------------------------------------------------------


class TestWriteMitsubaXml:
    def test_file_created(self, scene_dir, mesh_files):
        """scene.xml が生成されること."""
        building_objs, ground_ply = mesh_files
        xml_path = write_mitsuba_xml(scene_dir, building_objs, ground_ply)
        assert xml_path.exists()
        assert xml_path.name == "scene.xml"

    def test_valid_xml(self, scene_dir, mesh_files):
        """生成ファイルが valid な XML であること."""
        building_objs, ground_ply = mesh_files
        xml_path = write_mitsuba_xml(scene_dir, building_objs, ground_ply)
        ET.parse(xml_path)  # パース失敗なら ParseError を送出

    def test_building_shape_count(self, scene_dir, mesh_files):
        """建物 shape の数が building_objs の数と一致すること."""
        building_objs, ground_ply = mesh_files
        xml_path = write_mitsuba_xml(scene_dir, building_objs, ground_ply)
        root = ET.parse(xml_path).getroot()

        building_shapes = [s for s in root.findall("shape") if (s.get("id") or "").startswith("building_")]
        assert len(building_shapes) == len(building_objs)

    def test_ground_shape_exists(self, scene_dir, mesh_files):
        """地面 shape (id="ground") が存在すること."""
        building_objs, ground_ply = mesh_files
        xml_path = write_mitsuba_xml(scene_dir, building_objs, ground_ply)
        root = ET.parse(xml_path).getroot()

        ground = root.find("shape[@id='ground']")
        assert ground is not None

    def test_material_values(self, scene_dir, mesh_files):
        """指定した材質名が XML に反映されること."""
        building_objs, ground_ply = mesh_files
        xml_path = write_mitsuba_xml(
            scene_dir,
            building_objs,
            ground_ply,
            building_material="brick",
            ground_material="wet_ground",
        )
        content = xml_path.read_text()
        assert "brick" in content
        assert "wet_ground" in content

    def test_empty_buildings(self, scene_dir, mesh_files):
        """建物が 0 件でも scene.xml が生成されること."""
        _, ground_ply = mesh_files
        xml_path = write_mitsuba_xml(scene_dir, [], ground_ply)
        assert xml_path.exists()
        ET.parse(xml_path)

    @pytest.mark.parametrize(
        "building_material,ground_material",
        [
            ("concrete", "very_dry_ground"),
            ("glass", "wet_ground"),
            ("metal", "medium_dry_ground"),
        ],
    )
    def test_parametrized_materials(self, scene_dir, mesh_files, building_material, ground_material):
        """各材質の組み合わせで valid な XML が生成されること."""
        building_objs, ground_ply = mesh_files
        xml_path = write_mitsuba_xml(
            scene_dir,
            building_objs,
            ground_ply,
            building_material=building_material,
            ground_material=ground_material,
        )
        ET.parse(xml_path)  # valid XML であることを確認
