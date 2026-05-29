"""
PLATEAU CityGML (建築物モデル) のパーサー

ZIP ファイルをディスクに解凍せずストリーム処理し、各建物の
- LOD2 3D サーフェス群 (list[list[tuple[float,float,float]]])
  屋根形状を含む全ポリゴン、なければ None
- LOD1 底面フットプリント (Polygon)
  LOD2 がない建物の押し出し用
- measuredHeight (計測高さ [m])
を抽出して辞書のジェネレータとして返す

LOD2 サーフェスの抽出方針
--------------------------
lod2MultiSurface 内の全 Polygon をそのまま 3D 座標リストとして返す
各 Polygon は [(lon, lat, z), ...] のリスト (閉じたリング)

LOD1 底面フットプリントの抽出方針
----------------------------------
lod1Solid が持つ複数の Polygon のうち
「全頂点の z 座標が一定 (水平面) かつ z が最小」のものを底面とする

CityGML 座標順: 緯度(lat) 経度(lon) 高さ(z)  →  3D 座標は (lon, lat, z)
CRS: EPSG:6668 (JGD2011 地理座標系)
"""

import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Generator
from pathlib import Path

from shapely.geometry import Polygon

from .gml_utils import local_tag, parse_poslist_2d, parse_poslist_3d

# ---------------------------------------------------------------------------
# XML 名前空間
# ---------------------------------------------------------------------------
_NS = {
    "core": "http://www.opengis.net/citygml/2.0",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "gml": "http://www.opengis.net/gml",
    "uro": "https://www.geospatial.jp/iur/uro/3.1",
}

# 底面判定の z 方向許容誤差 [m]
_Z_TOL = 1e-3


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------


def _extract_lod2_surfaces(
    building_elem: ET.Element,
    polygon_index: dict[str, ET.Element],
) -> list[list[tuple[float, float, float]]] | None:
    """
    LOD2 タグ配下の全サーフェスを収集して 3D 座標リストを返す

    2つのケースに対応する:
    1. surfaceMember が直接 Polygon を含む場合
    2. surfaceMember が xlink:href で外部参照する場合 (PLATEAU 2023 千代田区)

    lod2Solid を優先し、なければ lod2MultiSurface を試みる
    """
    surfaces: list[list[tuple[float, float, float]]] = []

    for lod_tag in ("bldg:lod2Solid", "bldg:lod2MultiSurface"):
        lod2 = building_elem.find(lod_tag, _NS)
        if lod2 is None:
            continue

        for sm in lod2.findall(".//gml:surfaceMember", _NS):
            poly = sm.find(".//gml:Polygon", _NS)
            if poly is None:
                href = sm.get("{http://www.w3.org/1999/xlink}href")
                if href:
                    poly = polygon_index.get(href.lstrip("#"))

            if poly is None:
                continue

            poslist_elem = poly.find(".//gml:posList", _NS)
            if poslist_elem is None or not poslist_elem.text:
                continue
            coords_3d = parse_poslist_3d(poslist_elem.text.strip())
            if len(coords_3d) >= 3:
                surfaces.append(coords_3d)

        if surfaces:
            break  # lod2Solid で取得できたら lod2MultiSurface は不要

    return surfaces if surfaces else None


def _extract_lod1_footprint(
    building_elem: ET.Element,
) -> tuple[Polygon, float, float] | None:
    """
    <bldg:lod1Solid> から底面フットプリント・底面 z・上面 z を返す

    水平面の条件: 全頂点 z が一定 (_Z_TOL 以内)
    底面: z が最小の水平面、上面: z が最大の水平面

    Returns
    -------
    (footprint, z_base, z_top) — 底面 Polygon (2D)、底面高さ [m]、上面高さ [m]
    None                       — lod1Solid が存在しない、または水平面が 1 面以下
    """
    lod1 = building_elem.find("bldg:lod1Solid", _NS)
    if lod1 is None:
        return None

    horizontal: list[tuple[float, list[tuple[float, float]]]] = []
    for poly in lod1.findall(".//gml:Polygon", _NS):
        poslist_elem = poly.find(".//gml:posList", _NS)
        if poslist_elem is None or not poslist_elem.text:
            continue
        coords_2d, zvals = parse_poslist_2d(poslist_elem.text.strip())
        if len(coords_2d) < 3:
            continue
        if (max(zvals) - min(zvals)) <= _Z_TOL:
            horizontal.append((min(zvals), coords_2d))

    if len(horizontal) < 2:
        return None

    horizontal.sort(key=lambda x: x[0])
    z_base, bottom_coords = horizontal[0]
    z_top = horizontal[-1][0]
    return Polygon(bottom_coords), z_base, z_top


def _extract_measured_height(building_elem: ET.Element) -> float | None:
    """<bldg:measuredHeight> の値 [m] を返す、存在しない場合は None"""
    elem = building_elem.find("bldg:measuredHeight", _NS)
    if elem is not None and elem.text:
        try:
            return float(elem.text)
        except ValueError:
            return None
    return None


def _extract_building_id(building_elem: ET.Element) -> str | None:
    """
    <uro:buildingIDAttribute> から建物 ID を取得する
    存在しない場合は gml:id を返す
    """
    id_attr = building_elem.find("uro:buildingIDAttribute/uro:BuildingIDAttribute/uro:buildingID", _NS)
    if id_attr is not None and id_attr.text:
        return id_attr.text.strip()
    return building_elem.get("{http://www.opengis.net/gml}id")


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def iter_buildings_from_gml_file(gml_file) -> Generator[dict, None, None]:
    """
    GML ファイルオブジェクト (または bytes) を解析し、建物ごとの辞書を yield する

    Yields
    ------
    dict with keys:
        lod2_surfaces  : list[list[(lon,lat,z)]] | None
        lod1_footprint : tuple[Polygon, float, float] | None  — (底面 Polygon, z_base [m], z_top [m])
    """
    if isinstance(gml_file, (bytes, bytearray)):
        root = ET.fromstring(gml_file)
    else:
        root = ET.parse(gml_file).getroot()

    gml_id_attr = "{http://www.opengis.net/gml}id"
    polygon_index: dict[str, ET.Element] = {
        elem.get(gml_id_attr): elem
        for elem in root.iter()
        if local_tag(elem.tag) == "Polygon" and elem.get(gml_id_attr) is not None
    }  # type: ignore

    for b in (e for e in root.iter() if local_tag(e.tag) == "Building"):
        yield {
            "lod2_surfaces": _extract_lod2_surfaces(b, polygon_index),
            "lod1_footprint": _extract_lod1_footprint(b),
        }


def iter_buildings_from_zip(zip_path: Path) -> Generator[dict, None, None]:
    """
    PLATEAU CityGML ZIP をストリーム処理し、udx/bldg/*.gml 内の全建物を yield する
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        gml_names = sorted(name for name in zf.namelist() if "udx/bldg" in name and name.endswith(".gml"))
        for gml_name in gml_names:
            with zf.open(gml_name) as f:
                yield from iter_buildings_from_gml_file(f)
