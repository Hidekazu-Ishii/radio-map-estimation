# src/radio_map_estimation/scene/scene_writer.py
"""
PLY ファイル群 → Mitsuba 3 XML シーンファイル

Sionna RT が読み込める Mitsuba 3 XML を生成する.
材質には Sionna 組み込みの ITU Radio Material を使用する.

使用可能な ITU material type:
    "concrete", "brick", "plywood", "wood", "glass", "metal",
    "very_dry_ground", "medium_dry_ground", "wet_ground"
"""

from pathlib import Path


def write_mitsuba_xml(
    scene_dir: Path,
    building_objs: list[Path],
    ground_ply: Path,
    building_material: str = "concrete",
    ground_material: str = "very_dry_ground",
) -> Path:
    """
    Sionna RT が読み込める Mitsuba 3 XML シーンファイルを書き出す.

    建物を個別 shape として登録することで, 隣接建物間の頂点 merge による法線破壊を防ぐ.

    Parameters
    ----------
    scene_dir : Path
        XML の保存ディレクトリ. PLY は相対パスで参照される.
    building_plys : list[Path]
        建物 PLY ファイルの絶対パスリスト.
    ground_ply : Path
        地面 PLY ファイルの絶対パス.
    building_material : str
        建物の ITU 材質名.
    ground_material : str
        地面の ITU 材質名.

    Returns
    -------
    Path
        書き出した scene.xml のパス.
    """
    # 建物 shape の XML ブロックを生成
    shapes_xml = "\n".join(
        f'    <shape type="obj" id="building_{i:05d}">\n'
        f'        <string name="filename" value="{ply.relative_to(scene_dir)}"/>\n'
        f'        <ref id="mat-itu-buildings"/>\n'
        f"    </shape>"
        for i, ply in enumerate(building_objs)
    )

    ground_rel = ground_ply.relative_to(scene_dir)

    xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<scene version="3.0.0">

    <!-- 建物マテリアル -->
    <bsdf type="itu-radio-material" id="mat-itu-buildings">
        <string name="type" value="{building_material}"/>
    </bsdf>

    <!-- 地面マテリアル -->
    <bsdf type="itu-radio-material" id="mat-itu-ground">
        <string name="type" value="{ground_material}"/>
    </bsdf>

{shapes_xml}

    <!-- 地面 -->
    <shape type="ply" id="ground">
        <string name="filename" value="{ground_rel}"/>
        <ref id="mat-itu-ground"/>
    </shape>

</scene>
"""
    scene_xml = scene_dir / "scene.xml"
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_xml.write_text(xml_content, encoding="utf-8")
    print(f"[xml] {len(building_objs)} buildings → {scene_xml}")
    return scene_xml
