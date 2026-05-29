"""
PLATEAU GeoParquet から Sionna RT 用シーンを構築するエントリポイント

使い方:
    uv run scripts/build_scene.py configs/build_scene.yaml > outputs/logs/1_build_scene.log 2>&1

処理の流れ:
    configs/build_scene.yaml
        → city x mesh_code の全組み合わせをループ
        → area_spec_builder.build_area_spec()  : AreaSpec 構築
        → 各メッシュ生成 → PLY 保存 → Mitsuba XML 生成
    data/processed/<city_dir>/<mesh_code>/
        ├── bldg.ply
        ├── dem.ply
        ├── tran.ply
        ├── wtr.ply
        ├── scene.xml
        └── combined.ply

材質マッピング (ITU-R P.2040-3) :
    itu_bldg → concrete         (bldg: 建物)
    itu_tran → concrete         (tran: 道路、アスファルト専用材質なし)
    itu_dem  → very_dry_ground  (dem: 地形)
    itu_wtr  → wet_ground       (wtr: 水面)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import trimesh
from omegaconf import DictConfig, OmegaConf

from radio_map_estimation.scene.areaspec_builder import build_area_spec
from radio_map_estimation.scene.bldg_extruder import build_bldg_mesh
from radio_map_estimation.scene.mesh_builder import (
    build_dem_mesh,
    build_tran_mesh,
    build_wtr_mesh,
)
from radio_map_estimation.scene.mitsuba_xml import save_mitsuba_xml
from radio_map_estimation.scene.scene_preview import save_scene_preview

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 材質定数 (ITU-R P.2040-3)
_MATERIAL_BLDG = "itu_bldg"
_MATERIAL_DEM = "itu_dem"
_MATERIAL_TRAN = "itu_tran"
_MATERIAL_WTR = "itu_wtr"


# ---------------------------------------------------------------------------
# 設定 dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CityConfig:
    """1つの city ディレクトリに対する設定"""

    city_dir: str
    mesh_codes: tuple[str, ...]


@dataclass(frozen=True)
class BuildSceneConfig:
    """build_scene.yaml 全体の設定"""

    cities: tuple[CityConfig, ...]
    area_size_m: float

    @classmethod
    def from_omega(cls, cfg: DictConfig) -> BuildSceneConfig:
        area_size_m = float(cfg.area_size_m)
        if area_size_m <= 0:
            raise ValueError(f"area_size_m must be positive, got {area_size_m}")

        cities = tuple(
            CityConfig(
                city_dir=str(city.city_dir),
                mesh_codes=tuple(str(code) for code in city.mesh_codes),
            )
            for city in cfg.cities
        )
        if not cities:
            raise ValueError("cities must not be empty")

        return cls(cities=cities, area_size_m=area_size_m)


# ---------------------------------------------------------------------------
# パス導出 (純粋関数)
# ---------------------------------------------------------------------------


def _parquet_paths(data_dir: Path, city_dir: str) -> tuple[Path, Path, Path, Path]:
    """bldg / dem / tran / wtr の parquet パスを返す

    data/raw/<city_dir>/bldg.parquet を想定
    """
    base = data_dir / "raw" / city_dir
    return (
        base / "bldg.parquet",
        base / "dem.parquet",
        base / "tran.parquet",
        base / "wtr.parquet",
    )


# ---------------------------------------------------------------------------
# mesh_code → origin
# ---------------------------------------------------------------------------


def meshcode_to_origin(mesh_code: str) -> tuple[float, float]:
    """8桁の3次メッシュコードから左下隅の (lon, lat) を返す

    3次メッシュ (標準地域メッシュ) の仕様:
      1次: 緯度 40分 x 経度 1度
      2次: 1次を 8x8 分割 → 緯度 5分 x 経度 7.5分
      3次: 2次を 10x10 分割 → 緯度 30秒 x 経度 45秒 (≈ 1km²)
    """
    if len(mesh_code) != 8 or not mesh_code.isdigit():
        raise ValueError(f"mesh_code must be 8-digit string, got {mesh_code!r}")

    p = int(mesh_code[0:2])  # 1次: 緯度インデックス
    u = int(mesh_code[2:4])  # 1次: 経度インデックス
    q = int(mesh_code[4])  # 2次: 緯度インデックス
    v = int(mesh_code[5])  # 2次: 経度インデックス
    r = int(mesh_code[6])  # 3次: 緯度インデックス
    s = int(mesh_code[7])  # 3次: 経度インデックス

    origin_lat = (p * 40 + q * 5) / 60.0 + (r * 30) / 3600.0
    origin_lon = (u + 100) + (v * 7.5) / 60.0 + (s * 45) / 3600.0

    return origin_lon, origin_lat


# ---------------------------------------------------------------------------
# PLY 保存
# ---------------------------------------------------------------------------


def _save_ply(mesh: trimesh.Trimesh, path: Path) -> None:
    """メッシュを PLY ファイルとして保存する"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(mesh.export(file_type="ply"))  # type: ignore
    logger.info("Saved PLY: %s", path.name)


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------


def main(config_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    processed_dir = data_dir / "processed"

    cfg = BuildSceneConfig.from_omega(OmegaConf.load(config_path))  # type: ignore

    for city in cfg.cities:
        bldg_parquet, dem_parquet, tran_parquet, wtr_parquet = _parquet_paths(data_dir, city.city_dir)

        for mesh_code in city.mesh_codes:
            logger.info("Processing: city=%s, mesh_code=%s", city.city_dir, mesh_code)
            output_dir = processed_dir / city.city_dir / mesh_code
            output_dir.mkdir(parents=True, exist_ok=True)
            ply_paths: dict[str, Path] = {}

            origin_lon, origin_lat = meshcode_to_origin(mesh_code)

            # 1. AreaSpec 構築
            area_spec = build_area_spec(origin_lon, origin_lat, cfg.area_size_m)

            # 2. DEM (他メッシュの z 補間に使用するため最初に生成)
            dem_mesh = build_dem_mesh(dem_parquet, area_spec)
            if dem_mesh is not None:
                path = output_dir / "dem.ply"
                _save_ply(dem_mesh, path)
                ply_paths[_MATERIAL_DEM] = path
            else:
                logger.warning("mesh_code=%s: dem mesh not generated.", mesh_code)

            # 3. 道路 (dem_mesh による z 補間が必須)
            if dem_mesh is not None:
                tran_mesh = build_tran_mesh(tran_parquet, area_spec, dem_mesh)
                if tran_mesh is not None:
                    path = output_dir / "tran.ply"
                    _save_ply(tran_mesh, path)
                    ply_paths[_MATERIAL_TRAN] = path

            # 4. 水面 (dem_mesh による z 補間が必須)
            if dem_mesh is not None:
                wtr_mesh = build_wtr_mesh(wtr_parquet, area_spec, dem_mesh)
                if wtr_mesh is not None:
                    path = output_dir / "wtr.ply"
                    _save_ply(wtr_mesh, path)
                    ply_paths[_MATERIAL_WTR] = path

            # 5. 建物 (dem_mesh で底面 z をオフセット補正)
            bldg_mesh = build_bldg_mesh(bldg_parquet, area_spec, dem_mesh=dem_mesh)
            if bldg_mesh is not None:
                path = output_dir / "bldg.ply"
                _save_ply(bldg_mesh, path)
                ply_paths[_MATERIAL_BLDG] = path
            else:
                logger.warning("mesh_code=%s: bldg mesh not generated.", mesh_code)

            if not ply_paths:
                logger.warning("mesh_code=%s: no meshes generated, skipping.", mesh_code)
                continue

            # 6. Mitsuba XML
            xml_path = output_dir / "scene.xml"
            save_mitsuba_xml(ply_paths, xml_path, mesh_code)

            # 7. 3D プレビュー HTML
            save_scene_preview(
                scene_dir=output_dir,
                html_path=output_dir / "scene.html",
                title=f"PLATEAU 3D Scene: {mesh_code}",
            )

            logger.info("Done: %s", output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)

    main(Path(sys.argv[1]))
