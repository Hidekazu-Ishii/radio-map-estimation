"""
PLATEAU CityGML のダウンロード・保存
"""

import logging
from pathlib import Path

import geopandas as gpd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_zip(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """
    ZIP をストリームダウンロードして dest に保存する

    既にファイルが存在する場合はスキップ (再ダウンロードなし)
    """
    if dest.exists():
        logger.info("ZIP already exists, skipping download: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)

    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            dest.open("wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                bar.update(len(chunk))

    logger.info("Download complete: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def save_geodataframe(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """GeoDataFrame を GeoParquet または GeoJSON として保存する"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".parquet":
        gdf.to_parquet(output_path, index=False)
    elif suffix == ".geojson":
        gdf.to_file(output_path, driver="GeoJSON")
    else:
        raise ValueError(f"Unsupported output format: {suffix!r}")

    logger.info("Saved GeoDataFrame (%d rows) → %s", len(gdf), output_path)
