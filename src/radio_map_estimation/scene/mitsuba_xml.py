"""
PLY ファイル群から Mitsuba3 / Sionna RT 用シーン XML を生成する

役割
----
材質別 PLY パスの辞書を受け取り、各材質に対して
- <bsdf type="itu-radio-material"> ノード (ITU-R P.2040-3 材質)
- <shape type="ply"> ノード (BSDF を ref で参照)
を持つ XML ファイルを生成する

材質マッピング (ITU-R P.2040-3) :
    itu_bldg → concrete         (bldg: 建物)
    itu_tran → concrete         (tran: 道路、アスファルト専用材質なし)
    itu_dem  → very_dry_ground  (dem: 地形)
    itu_wtr  → wet_ground       (wtr: 水面)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# sionna_material → ITU-R P.2040-3 材質名
# キー: "itu_" 除去後のレイヤ名 (bldg / tran / dem / wtr)
_MATERIAL_TO_ITU: dict[str, str] = {
    "bldg": "concrete",
    "tran": "concrete",
    "dem": "very_dry_ground",
    "wtr": "wet_ground",
}


def _itu_type(sionna_material: str) -> str:
    """
    sionna_material 名を ITU-R P.2040-3 の材質名に変換する

    Examples
    --------
    "itu_bldg" → "concrete"
    "itu_tran" → "concrete"
    "itu_dem"  → "very_dry_ground"
    "itu_wtr"  → "wet_ground"

    Raises
    ------
    ValueError
        未知の sionna_material が渡された場合
    """
    key = sionna_material.removeprefix("itu_")
    if key not in _MATERIAL_TO_ITU:
        raise ValueError(
            f"Unknown sionna_material: '{sionna_material}'. "
            f"Expected one of: {sorted('itu_' + k for k in _MATERIAL_TO_ITU)}"
        )
    return _MATERIAL_TO_ITU[key]


def save_mitsuba_xml(
    ply_paths: dict[str, Path],
    xml_path: Path,
) -> None:
    """
    材質別 PLY を参照する Mitsuba3 / Sionna RT 用シーン XML を生成する

    Parameters
    ----------
    ply_paths  : dict[sionna_material, Path]
                 キーは "itu_bldg" / "itu_tran" / "itu_dem" / "itu_wtr" のいずれか
    xml_path   : 出力する XML のパス
    """
    xml_path.parent.mkdir(parents=True, exist_ok=True)

    bsdf_nodes = ""
    for material in sorted(ply_paths):
        bsdf_nodes += (
            f'    <bsdf type="itu-radio-material" id="{material}">\n'
            f'        <string name="type" value="{_itu_type(material)}"/>\n'
            f"    </bsdf>\n"
        )

    shape_nodes = ""
    for material, ply_path in sorted(ply_paths.items()):
        try:
            ply_rel = ply_path.relative_to(xml_path.parent)
        except ValueError:
            ply_rel = ply_path.resolve().relative_to(xml_path.parent.resolve())

        shape_id = material.removeprefix("itu_")  # "itu_dem" → "dem"
        shape_nodes += (
            f'    <shape type="ply" id="{shape_id}">\n'
            f'        <string name="filename" value="{ply_rel}"/>\n'
            f'        <ref id="{material}"/>\n'
            f"    </shape>\n"
        )

    xml_content = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<!--\n"
        "  Mitsuba3 / Sionna RT scene XML\n"
        "  Generated from PLATEAU LOD2/LOD1 building models + DEM + road + water\n"
        "  Coordinate system: local meters, origin = bbox bottom-left, z-up\n"
        "  Materials: ITU-R P.2040-3\n"
        "-->\n"
        '<scene version="3.0.0">\n'
        "\n"
        '    <integrator type="path"/>\n'
        "\n"
        "    <!-- ===== Radio materials (ITU-R P.2040-3) ===== -->\n"
        f"{bsdf_nodes}"
        "\n"
        "    <!-- ===== Geometry (bldg / dem / tran / wtr) ===== -->\n"
        f"{shape_nodes}"
        "</scene>\n"
    )

    xml_path.write_text(xml_content, encoding="utf-8")
    logger.info("Saved Mitsuba XML: %s (%d material(s))", xml_path, len(ply_paths))
