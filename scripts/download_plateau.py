"""
PLATEAU データダウンロードのエントリポイント

使い方:
    uv run scripts/download_plateau.py configs/plateau_download.yaml > outputs/logs/0_download_plateau.log 2>&1

処理の流れ:
    CityGML ZIP (LOD0/1/2 を含む単一ファイル)
        ↓ download_zip (1 回)
        ↓ build_bldg / dem / tran_geodataframe
        ↓ save_geodataframe
    data/raw/<city_code>_<city_name>_<year>/
        citygml.zip   — CityGML ZIP (ダウンロード済みはスキップ)
        bldg.parquet  — 建物 (LOD2 サーフェス + LOD1 直方体)
        dem.parquet   — 地形 TIN 三角形
        tran.parquet  — 道路ポリゴン
        wtr.parquet   — 水部エリア (luse orgLandUse=7000)
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from radio_map_estimation.plateau.gdf_builder import (
    build_bldg_geodataframe,
    build_dem_geodataframe,
    build_tran_geodataframe,
    build_wtr_geodataframe,
)
from radio_map_estimation.plateau.io import download_zip, save_geodataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 設定 dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AreaConfig:
    city_code: str
    city_name: str
    year: int
    citygml_url: str


@dataclass(frozen=True)
class PlateauDownloadConfig:
    areas: tuple[AreaConfig, ...]
    raw_dir: Path
    output_format: str

    @classmethod
    def from_omega(cls, cfg: DictConfig) -> "PlateauDownloadConfig":
        fmt = cfg.get("output_format", "geoparquet")
        if fmt not in ("geoparquet", "geojson"):
            raise ValueError(f"output_format must be 'geoparquet' or 'geojson', got {fmt!r}")
        return cls(
            areas=tuple(
                AreaConfig(
                    city_code=a.city_code,
                    city_name=a.city_name,
                    year=int(a.year),
                    citygml_url=a.citygml_url,
                )
                for a in cfg.areas
            ),
            raw_dir=Path(cfg.raw_dir),
            output_format=fmt,
        )


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------


def _suffix(fmt: str) -> str:
    return ".parquet" if fmt == "geoparquet" else ".geojson"


def main(config_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = PlateauDownloadConfig.from_omega(OmegaConf.load(config_path))  # type: ignore
    suffix = _suffix(cfg.output_format)

    for area in cfg.areas:
        logger.info("=== Processing: %s ===", area.city_name)

        area_dir = root / cfg.raw_dir / f"{area.city_code}_{area.city_name}_{area.year}"
        zip_path = area_dir / "citygml.zip"

        download_zip(area.citygml_url, zip_path)

        gdf_bldg = build_bldg_geodataframe(zip_path)
        save_geodataframe(gdf_bldg, area_dir / f"bldg{suffix}")
        logger.info(
            "  bldg: %d rows (surfaces: %d, none: %d)",
            len(gdf_bldg),
            gdf_bldg["surfaces"].notna().sum(),
            gdf_bldg["surfaces"].isna().sum(),
        )

        gdf_dem = build_dem_geodataframe(zip_path)
        save_geodataframe(gdf_dem, area_dir / f"dem{suffix}")
        logger.info("  dem : %d triangles", len(gdf_dem))

        gdf_tran = build_tran_geodataframe(zip_path)
        save_geodataframe(gdf_tran, area_dir / f"tran{suffix}")
        logger.info("  tran: %d road polygons", len(gdf_tran))

        gdf_wtr = build_wtr_geodataframe(zip_path)
        save_geodataframe(gdf_wtr, area_dir / f"wtr{suffix}")
        logger.info("  wtr : %d water polygons (from luse orgLandUse=7000)", len(gdf_wtr))

        logger.info("%s → %s", area.city_name, area_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run scripts/download_plateau.py <config.yaml>")
        sys.exit(1)
    main(Path(sys.argv[1]))
