"""
PLATEAU 地形モデル (DEM) CityGML のパーサー

TINRelief の三角形パッチを 3D サーフェスリストとして抽出する

CityGML 構造:
  ReliefFeature > reliefComponent > TINRelief
    > tin > TriangulatedSurface > trianglePatches
      > Triangle > exterior > LinearRing > posList

CityGML 座標順: 緯度(lat) 経度(lon) 高さ(z)
→ (lon, lat, z) に変換する
CRS: EPSG:6668 (JGD2011 地理座標系)
"""

import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Generator
from pathlib import Path

# ---------------------------------------------------------------------------
# XML 名前空間
# ---------------------------------------------------------------------------
_NS = {
    "core": "http://www.opengis.net/citygml/2.0",
    "dem": "http://www.opengis.net/citygml/relief/2.0",
    "gml": "http://www.opengis.net/gml",
}


def _local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _parse_poslist_3d(text: str) -> list[tuple[float, float, float]]:
    """
    CityGML の posList を (lon, lat, z) のリストに変換する

    CityGML 座標順 (lat, lon, z) → (lon, lat, z) に変換する
    """
    vals = list(map(float, text.split()))
    return [(vals[i + 1], vals[i], vals[i + 2]) for i in range(0, len(vals), 3)]


def iter_dem_surfaces_from_gml_file(
    gml_file,
) -> Generator[list[tuple[float, float, float]], None, None]:
    """
    GML ファイルオブジェクト (または bytes) を解析し,
    TIN の三角形ポリゴンを 1 枚ずつ yield する

    Yields
    ------
    list[(lon, lat, z)] — 三角形の 3 頂点 (+ 閉じ点で計 4 点)
    """
    if isinstance(gml_file, (bytes, bytearray)):
        root = ET.fromstring(gml_file)
        triangles = root.findall(".//gml:Triangle", _NS)
    else:
        triangles = []
        for _, elem in ET.iterparse(gml_file, events=("end",)):
            if _local(elem.tag) == "Triangle":
                triangles.append(elem)

    for tri in triangles:
        poslist_elem = tri.find(".//gml:posList", _NS)
        if poslist_elem is None or not poslist_elem.text:
            continue
        coords = _parse_poslist_3d(poslist_elem.text.strip())
        if len(coords) >= 3:
            yield coords


def iter_dem_surfaces_from_zip(
    zip_path: Path,
) -> Generator[list[tuple[float, float, float]], None, None]:
    """
    PLATEAU CityGML ZIP をストリーム処理し,
    udx/dem/*.gml 内の全 TIN 三角形を yield する

    Parameters
    ----------
    zip_path : PLATEAU CityGML ZIP のパス
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        gml_names = sorted(name for name in zf.namelist() if "udx/dem" in name and name.endswith(".gml"))
        for gml_name in gml_names:
            with zf.open(gml_name) as f:
                yield from iter_dem_surfaces_from_gml_file(f)
