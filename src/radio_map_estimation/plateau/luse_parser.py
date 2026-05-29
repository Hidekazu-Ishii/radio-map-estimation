"""
PLATEAU 土地利用モデル (LandUse) CityGML のパーサー

LandUse の lod1MultiSurface から土地利用ポリゴンを抽出し、
用途コード (uro:orgLandUse) で水部エリア (7000: 水面・河川・水路) のみを yield する

水部モデル (udx/wtr) は整備されていないエリアが多いため、
土地利用モデル (udx/luse) の水部区分を水面エリアとして代替する

CityGML 構造:
  LandUse > uro:landUseDetailAttribute > uro:LandUseDetailAttribute
          > uro:orgLandUse             — 土地利用コード (7000 = 水面・河川・水路)
  LandUse > lod1MultiSurface > MultiSurface > surfaceMember
          > Polygon > exterior > LinearRing > posList

z 座標について:
  LOD1 のため PLATEAU 仕様上 z=0 固定
  実際の水面標高が必要な場合は DEM から補間すること

CityGML 座標順: 緯度(lat) 経度(lon) 高さ(z)  →  (lon, lat, z) に変換する
CRS: EPSG:6668 (JGD2011 地理座標系)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Generator
from pathlib import Path

from .gml_utils import local_tag, parse_poslist_3d

_NS = {
    "core": "http://www.opengis.net/citygml/2.0",
    "luse": "http://www.opengis.net/citygml/landuse/2.0",
    "uro": "https://www.geospatial.jp/iur/uro/3.1",
    "gml": "http://www.opengis.net/gml",
}

# 水面・河川・水路の土地利用コード
_WATER_CODE = "7000"


def _is_water(luse_elem: ET.Element) -> bool:
    """LandUse 要素が水部エリア (orgLandUse == 7000) かどうかを返す"""
    elem = luse_elem.find("uro:landUseDetailAttribute/uro:LandUseDetailAttribute/uro:orgLandUse", _NS)
    return elem is not None and (elem.text or "").strip() == _WATER_CODE


def _extract_luse_surfaces(
    luse_elem: ET.Element,
) -> list[list[tuple[float, float, float]]]:
    """LandUse 要素から LOD1 ポリゴンの 3D 座標リストを返す"""
    surfaces: list[list[tuple[float, float, float]]] = []

    lod1 = luse_elem.find("luse:lod1MultiSurface", _NS)
    if lod1 is None:
        return surfaces

    for poly in lod1.findall(".//gml:Polygon", _NS):
        poslist_elem = poly.find(".//gml:posList", _NS)
        if poslist_elem is None or not poslist_elem.text:
            continue
        coords = parse_poslist_3d(poslist_elem.text.strip())
        if len(coords) >= 3:
            surfaces.append(coords)

    return surfaces


def iter_water_surfaces_from_gml_file(
    gml_file,
) -> Generator[list[tuple[float, float, float]], None, None]:
    """
    GML ファイルオブジェクト (または bytes) を解析し,
    水部エリア (orgLandUse=7000) の LOD1 ポリゴンを 1 枚ずつ yield する

    Yields
    ------
    list[(lon, lat, z)] — 水面ポリゴンの頂点リスト (z=0 固定)
    """
    if isinstance(gml_file, (bytes, bytearray)):
        root = ET.fromstring(gml_file)
        luses = [e for e in root.iter() if local_tag(e.tag) == "LandUse"]
    else:
        luses = []
        for _, elem in ET.iterparse(gml_file, events=("end",)):
            if local_tag(elem.tag) == "LandUse":
                luses.append(elem)

    for luse in luses:
        if _is_water(luse):
            yield from _extract_luse_surfaces(luse)


def iter_water_surfaces_from_zip(
    zip_path: Path,
) -> Generator[list[tuple[float, float, float]], None, None]:
    """
    PLATEAU CityGML ZIP をストリーム処理し,
    udx/luse/*.gml 内の水部エリアポリゴンを yield する
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        gml_names = sorted(name for name in zf.namelist() if "udx/luse" in name and name.endswith(".gml"))
        for gml_name in gml_names:
            with zf.open(gml_name) as f:
                yield from iter_water_surfaces_from_gml_file(f)
