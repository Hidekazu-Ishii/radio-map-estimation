"""
scripts/fetch_buildings.py

目的: OSM建物マップを取得するエントリポイント.
      configs/osm_buildings.yaml を読み込んで src/osm_buildings.py の処理を呼び出す.

Usage:
    uv run scripts/sim_data/fetch_buildings.py
    uv run scripts/sim_data/fetch_buildings.py --config configs/osm_buildings.yaml
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from radio_map_estimation.generater.osm_buildings import (
    buildings_to_grid,
    fetch_buildings_osm,
    plot_building_map,
)


@dataclass(frozen=True)
class OSMBuildingsConfig:
    locations: list[tuple[float, float]]
    area_size_m: float
    grid_size: int
    meters_per_level: float
    default_building_height_m: float


def load_config(config_path: Path) -> OSMBuildingsConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return OSMBuildingsConfig(**raw)


# OSMから建物マップを取得してグリッドに変換・保存する.


def main(
    center_lat: float,
    center_lon: float,
    save_dir: Path,
    area_size_m: float,
    grid_size: int,
    meters_per_level: float,
    default_building_height_m: float,
) -> None:

    stem = f"{center_lat:.4f}_{center_lon:.4f}"
    output_npz = save_dir / f"{stem}.npz"
    output_png = save_dir / f"{stem}.png"

    print("=== OSM Building Map Fetcher ===")
    print(f"  center : ({center_lat}, {center_lon})")
    print(f"  area   : {area_size_m} m x {area_size_m} m")
    print(f"  grid   : {grid_size} x {grid_size}")

    # 1. OSM取得
    print("[1/3] Fetching buildings from OpenStreetMap ...")
    gdf_proj, bbox_m = fetch_buildings_osm(
        center_lat,
        center_lon,
        area_size_m,
        meters_per_level,
        default_building_height_m,
    )
    print(f"      Number of polygons fetched: {len(gdf_proj)}")

    # 2. グリッド変換
    print("[2/3] Converting to grid ...")
    building_mask, building_heights = buildings_to_grid(gdf_proj, bbox_m, grid_size, area_size_m)
    print(f"      Building coverage: {building_mask.mean() * 100:.1f}%")
    print(
        f"      Height range (buildings only): "
        f"{building_heights[building_mask].min():.1f} ~ "
        f"{building_heights[building_mask].max():.1f} m"
    )

    # 3. 保存
    print("[3/3] Saving ...")
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        building_mask=building_mask,  # (grid_size, grid_size) bool
        building_heights=building_heights,  # (grid_size, grid_size) float [m]
        area_size_m=np.array(area_size_m),
        grid_size=np.array(grid_size),
        center_lat=np.array(center_lat),
        center_lon=np.array(center_lon),
        bbox_m=np.array(bbox_m),  # (xmin, ymin, xmax, ymax) [m]
    )
    print(f"      saved → {output_npz}")

    plot_building_map(
        building_mask,
        building_heights,
        area_size_m,
        center_lat,
        center_lon,
        output_png,
    )


def run(config_path: Path) -> None:
    cfg = load_config(config_path)

    root = Path(__file__).resolve().parents[2]
    save_dir = root / "data" / "buildings"

    for lat, lon in cfg.locations:
        print(f"\n[Location] ({lat}, {lon})")
        try:
            main(
                center_lat=lat,
                center_lon=lon,
                save_dir=save_dir,
                area_size_m=cfg.area_size_m,
                grid_size=cfg.grid_size,
                meters_per_level=cfg.meters_per_level,
                default_building_height_m=cfg.default_building_height_m,
            )
        except Exception as e:
            print(f"[error] Failed ({lat}, {lon}): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/osm_buildings.yaml"),
    )
    args = parser.parse_args()
    run(args.config)
