# src/radio_map_estimation/scene/schema.py
"""
scene パッケージのデータ構造.
"""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from omegaconf import OmegaConf


@dataclass(frozen=True)
class CityConfig:
    """都市固有の設定 (configs/cities/{city}.yaml)."""

    center_lat: float
    center_lon: float
    area_size_m: float
    cell_size_m: float


@dataclass(frozen=True)
class BuildingHeightConfig:
    """建物高さ推定パラメータ (configs/building_height.yaml)."""

    meters_per_level: float


@dataclass(frozen=True)
class AreaSpec:
    """シミュレーション対象エリアの仕様.

    ローカル座標系: bbox_m の左下 (xmin, ymin) を原点とする.
    area_size_m = xmax - xmin = ymax - ymin (正方形を前提とする).
    grid_size は area_size_m / cell_size_m から導出する.
    """

    center_lat: float
    center_lon: float
    area_size_m: float
    cell_size_m: float
    crs: str
    bbox_xmin: float
    bbox_ymin: float
    bbox_xmax: float
    bbox_ymax: float

    @property
    def bbox_m(self) -> tuple[float, float, float, float]:
        return (self.bbox_xmin, self.bbox_ymin, self.bbox_xmax, self.bbox_ymax)

    @property
    def grid_size(self) -> int:
        return int(self.area_size_m / self.cell_size_m)


@dataclass(frozen=True)
class BuildingData:
    """建物情報をまとめたデータ構造.

    Attributes
    ----------
    gdf : GeoDataFrame
        投影座標系の建物ポリゴン. カラムは geometry, height_m のみ.
    building_mask : ndarray of shape (grid_size, grid_size), dtype bool
        建物セルが True. [row, col] = [y_idx, x_idx], [0,0] が左下.
    building_heights : ndarray of shape (grid_size, grid_size), dtype float
        セルの最大建物高さ [m]. 建物がない場合は 0.0.
    area_spec : AreaSpec
    """

    gdf: gpd.GeoDataFrame
    building_mask: np.ndarray
    building_heights: np.ndarray
    area_spec: AreaSpec

    def to_local_gdf(self) -> gpd.GeoDataFrame:
        """投影座標系 → ローカル座標系 (bbox左下を原点) に平行移動."""
        gdf_local = self.gdf.copy()
        gdf_local["geometry"] = gdf_local["geometry"].translate(
            xoff=-self.area_spec.bbox_xmin,
            yoff=-self.area_spec.bbox_ymin,
        )
        return gdf_local


def load_city_config(config_path: Path) -> CityConfig:
    cfg = OmegaConf.load(config_path)
    return CityConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]


def load_building_height_config(config_path: Path) -> BuildingHeightConfig:
    cfg = OmegaConf.load(config_path)
    return BuildingHeightConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]


def save_building_data(data: BuildingData, output_dir: Path) -> None:
    """BuildingData を npz ファイルに保存する."""
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = data.area_spec
    np.savez(
        output_dir / "building_data.npz",
        building_mask=data.building_mask,
        building_heights=data.building_heights,
        center_lat=spec.center_lat,
        center_lon=spec.center_lon,
        area_size_m=spec.area_size_m,
        cell_size_m=spec.cell_size_m,
        bbox_xmin=spec.bbox_xmin,
        bbox_ymin=spec.bbox_ymin,
        bbox_xmax=spec.bbox_xmax,
        bbox_ymax=spec.bbox_ymax,
    )
    print(f"[save] building_data.npz → {output_dir}")


def load_building_data(npz_path: Path, gdf: gpd.GeoDataFrame) -> BuildingData:
    """npz ファイルから BuildingData を復元する."""
    d = np.load(npz_path)
    area_spec = AreaSpec(
        center_lat=float(d["center_lat"]),
        center_lon=float(d["center_lon"]),
        area_size_m=float(d["area_size_m"]),
        cell_size_m=float(d["cell_size_m"]),
        crs="",  # npz には保存されないため別途渡す必要あり
        bbox_xmin=float(d["bbox_xmin"]),
        bbox_ymin=float(d["bbox_ymin"]),
        bbox_xmax=float(d["bbox_xmax"]),
        bbox_ymax=float(d["bbox_ymax"]),
    )
    return BuildingData(
        gdf=gdf,
        building_mask=d["building_mask"],
        building_heights=d["building_heights"],
        area_spec=area_spec,
    )
