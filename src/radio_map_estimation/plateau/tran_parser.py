"""
PLATEAU 交通モデル (道路) CityGML のパーサー

Road の lod2MultiSurface (なければ lod1MultiSurface) から
道路ポリゴンを 3D サーフェスリストとして抽出する

CityGML 構造:
  Road > lod2MultiSurface > MultiSurface > surfaceMember
       > Polygon > exterior > LinearRing > posList

LOD 優先順位: lod2MultiSurface → lod1MultiSurface

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
    "tran": "http://www.opengis.net/citygml/transportation/2.0",
    "gml": "http://www.opengis.net/gml",
}


def _extract_road_surfaces(
    road_elem: ET.Element,
) -> list[list[tuple[float, float, float]]]:
    """
    Road 要素から道路ポリゴンの 3D 座標リストを返す

    lod2MultiSurface を優先し, なければ lod1MultiSurface を使用する
    """
    surfaces: list[list[tuple[float, float, float]]] = []

    for lod_tag in ("tran:lod2MultiSurface", "tran:lod1MultiSurface"):
        lod = road_elem.find(lod_tag, _NS)
        if lod is None:
            continue
        for poly in lod.findall(".//gml:Polygon", _NS):
            poslist_elem = poly.find(".//gml:posList", _NS)
            if poslist_elem is None or not poslist_elem.text:
                continue
            coords = parse_poslist_3d(poslist_elem.text.strip())
            if len(coords) >= 3:
                surfaces.append(coords)
        if surfaces:
            break  # lod2 で取得できたら lod1 は不要

    return surfaces


def iter_road_surfaces_from_gml_file(
    gml_file,
) -> Generator[list[tuple[float, float, float]], None, None]:
    """
    GML ファイルオブジェクト (または bytes) を解析し,
    道路ポリゴンを 1 枚ずつ yield する

    Yields
    ------
    list[(lon, lat, z)] — 道路ポリゴンの頂点リスト
    """
    if isinstance(gml_file, (bytes, bytearray)):
        root = ET.fromstring(gml_file)
        roads = [e for e in root.iter() if local_tag(e.tag) == "Road"]
    else:
        roads = []
        for _, elem in ET.iterparse(gml_file, events=("end",)):
            if local_tag(elem.tag) == "Road":
                roads.append(elem)

    for road in roads:
        yield from _extract_road_surfaces(road)


def iter_road_surfaces_from_zip(
    zip_path: Path,
) -> Generator[list[tuple[float, float, float]], None, None]:
    """
    PLATEAU CityGML ZIP をストリーム処理し, udx/tran/*.gml 内の全道路ポリゴンを yield する
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        gml_names = sorted(name for name in zf.namelist() if "udx/tran" in name and name.endswith(".gml"))
        for gml_name in gml_names:
            with zf.open(gml_name) as f:
                yield from iter_road_surfaces_from_gml_file(f)
