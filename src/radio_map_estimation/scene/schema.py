"""
scene モジュール全体で共有するデータ構造を定義する

設計方針
--------
- データ構造の定義のみを担う (ロジックは各モジュールに委譲)
- frozen=True による不変性でデータ整合性を保証
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AreaSpec:
    """シミュレーション対象エリアの仕様

    ローカル座標系: origin (エリア左下隅) を (0, 0) とし、
    有効エリア x, y ∈ [0, area_size_m]
    マージン部分は x, y ∈ [-margin, 0) および (area_size_m, area_size_m + margin]

    bbox は origin を基点に、負方向・正方向ともに margin (area_size_m / 5) を
    加えた広い範囲を取得する:
        bbox_xmin = origin_proj_x - margin
        bbox_xmax = origin_proj_x + area_size_m + margin
        bbox_ymin = origin_proj_y - margin
        bbox_ymax = origin_proj_y + area_size_m + margin

    Attributes
    ----------
    origin_lat    : エリア左下隅の緯度 [deg]
    origin_lon    : エリア左下隅の経度 [deg]
    area_size_m   : エリアの一辺の長さ [m]
    crs           : 投影座標系の EPSG 文字列 (例: "EPSG:6677")
    origin_proj_x : origin の投影座標 x [m] (ローカル座標の原点)
    origin_proj_y : origin の投影座標 y [m] (ローカル座標の原点)
    bbox_xmin     : 投影座標の x 最小値 [m] (ローカル座標で -margin に対応)
    bbox_ymin     : 投影座標の y 最小値 [m] (ローカル座標で -margin に対応)
    bbox_xmax     : 投影座標の x 最大値 [m] (ローカル座標で area_size_m + margin に対応)
    bbox_ymax     : 投影座標の y 最大値 [m] (ローカル座標で area_size_m + margin に対応)
    """

    origin_lat: float
    origin_lon: float
    area_size_m: float
    crs: str
    origin_proj_x: float
    origin_proj_y: float
    bbox_xmin: float
    bbox_ymin: float
    bbox_xmax: float
    bbox_ymax: float

    @property
    def bbox_m(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) [m] in projected CRS."""
        return (self.bbox_xmin, self.bbox_ymin, self.bbox_xmax, self.bbox_ymax)
