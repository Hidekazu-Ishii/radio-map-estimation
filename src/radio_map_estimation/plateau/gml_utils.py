"""
CityGML パーサー共通ユーティリティ
"""


def local_tag(tag: str) -> str:
    """Clark 記法 {ns}localname → localname を返す"""
    return tag.split("}")[-1] if "}" in tag else tag


def parse_poslist_3d(text: str) -> list[tuple[float, float, float]]:
    """
    CityGML の posList テキストを 3D 座標リストに変換する

    CityGML 座標順 (lat, lon, z) → (lon, lat, z) に変換する

    Returns
    -------
    list[(lon, lat, z)]
    """
    vals = list(map(float, text.split()))
    return [(vals[i + 1], vals[i], vals[i + 2]) for i in range(0, len(vals), 3)]


def parse_poslist_2d(text: str) -> tuple[list[tuple[float, float]], list[float]]:
    """
    CityGML の posList テキストを 2D 座標リストと z リストに変換する

    Returns
    -------
    coords_2d : list[(lon, lat)]
    zvals     : list[float]
    """
    vals = list(map(float, text.split()))
    coords_2d = [(vals[i + 1], vals[i]) for i in range(0, len(vals), 3)]
    zvals = [vals[i + 2] for i in range(0, len(vals), 3)]
    return coords_2d, zvals
