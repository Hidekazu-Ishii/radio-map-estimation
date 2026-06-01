"""
PLATEAU シーン構築 + Sionna RT シミュレーションの統合エントリポイント

使い方:
    uv run scripts/simulate.py configs/scene.yaml configs/sionna.yaml

処理の流れ:
    configs/scene.yaml + configs/sionna.yaml
        → city x mesh_code の全組み合わせをループ
        → build_area_spec()        : AreaSpec 構築
        → build_*_mesh()           : PLY 生成・保存
        → save_mitsuba_xml()       : scene.xml 生成
        → save_scene_preview()     : scene.html 生成
        → build_tx_positions()     : TX 位置を bldg.ply から自動決定
        → build_radio_maps()       : Sionna RT シミュレーション
        → save_radio_maps()        : PNG / npz 保存

出力先: data/processed/<city_dir>/<mesh_code>/
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from omegaconf import DictConfig, OmegaConf

from radio_map_estimation.scene.areaspec_builder import build_area_spec
from radio_map_estimation.scene.bldg_extruder import build_bldg_footprint_mesh, build_bldg_mesh
from radio_map_estimation.scene.mesh_builder import (
    build_dem_mesh,
    build_tran_mesh,
    build_wtr_mesh,
)
from radio_map_estimation.scene.mitsuba_xml import save_mitsuba_xml
from radio_map_estimation.scene.scene_preview import save_scene_preview
from radio_map_estimation.sionna.radiomap import build_radio_maps
from radio_map_estimation.sionna.tx_placement import build_tx_positions
from radio_map_estimation.sionna.visualize import save_radio_maps

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
class SionnaConfig:
    """sionna.yaml 全体の設定"""

    num_tx: int
    min_separation_m: float
    tx_power_dbm: float
    tx_pattern: str
    tx_tilt_deg: float
    rx_height_m: float
    frequency_hz: float
    cell_size_m: float
    num_samples: int
    max_depth: int
    noise_std_db: float

    @classmethod
    def from_omega(cls, cfg: DictConfig) -> SionnaConfig:
        num_tx = int(cfg.num_tx)
        if num_tx < 1:
            raise ValueError(f"num_tx must be >= 1, got {num_tx}")
        tx_pattern = str(cfg.tx_pattern)
        if tx_pattern not in ("iso", "tr38901"):
            raise ValueError(f"tx_pattern must be 'iso' or 'tr38901', got {tx_pattern!r}")
        return cls(
            num_tx=num_tx,
            min_separation_m=float(cfg.min_separation_m),
            tx_power_dbm=float(cfg.tx_power_dbm),
            tx_pattern=tx_pattern,
            tx_tilt_deg=float(cfg.tx_tilt_deg),
            rx_height_m=float(cfg.rx_height_m),
            frequency_hz=float(cfg.frequency_hz),
            cell_size_m=float(cfg.cell_size_m),
            num_samples=int(cfg.num_samples),
            max_depth=int(cfg.max_depth),
            noise_std_db=float(cfg.noise_std_db),
        )


@dataclass(frozen=True)
class CityConfig:
    """1つの city ディレクトリに対する設定"""

    city_dir: str
    mesh_codes: tuple[str, ...]


@dataclass(frozen=True)
class SceneConfig:
    """scene.yaml 全体の設定"""

    cities: tuple[CityConfig, ...]
    area_size_m: float

    @classmethod
    def from_omega(cls, cfg: DictConfig) -> SceneConfig:
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
    """bldg / dem / tran / wtr の parquet パスを返す"""
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
    """8桁の3次メッシュコードから左下隅の (lon, lat) を返す"""
    if len(mesh_code) != 8 or not mesh_code.isdigit():
        raise ValueError(f"mesh_code must be 8-digit string, got {mesh_code!r}")

    p = int(mesh_code[0:2])
    u = int(mesh_code[2:4])
    q = int(mesh_code[4])
    v = int(mesh_code[5])
    r = int(mesh_code[6])
    s = int(mesh_code[7])

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


def main(scene_config_path: Path, sionna_config_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    processed_dir = data_dir / "processed"

    scene_cfg = SceneConfig.from_omega(OmegaConf.load(scene_config_path))  # type: ignore
    sionna_cfg = SionnaConfig.from_omega(OmegaConf.load(sionna_config_path))  # type: ignore

    # 乱数シードはエントリポイントで1回だけ固定する
    rng = np.random.default_rng(seed=42)

    for city in scene_cfg.cities:
        bldg_parquet, dem_parquet, tran_parquet, wtr_parquet = _parquet_paths(data_dir, city.city_dir)

        for mesh_code in city.mesh_codes:
            logger.info("Processing: city=%s, mesh_code=%s", city.city_dir, mesh_code)
            output_dir = processed_dir / city.city_dir / mesh_code
            output_dir.mkdir(parents=True, exist_ok=True)

            # 既存の PLY / XML / HTML を削除して再生成 (残留データの混入を防ぐ)
            for old_file in output_dir.glob("*.ply"):
                old_file.unlink()
            for suffix in ("*.xml", "*.html"):
                for old_file in output_dir.glob(suffix):
                    old_file.unlink()

            ply_paths: dict[str, Path] = {}

            origin_lon, origin_lat = meshcode_to_origin(mesh_code)

            # 1. AreaSpec 構築
            area_spec = build_area_spec(origin_lon, origin_lat, scene_cfg.area_size_m)

            # 2. DEM
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

            # 5b. 建物底面フットプリント (建物マスク生成用)
            bldg_footprint_mesh = build_bldg_footprint_mesh(bldg_parquet, area_spec)
            if bldg_footprint_mesh is not None:
                _save_ply(bldg_footprint_mesh, output_dir / "bldg_footprint.ply")
            else:
                logger.warning("mesh_code=%s: bldg_footprint mesh not generated.", mesh_code)

            if not ply_paths:
                logger.warning("mesh_code=%s: no meshes generated, skipping.", mesh_code)
                continue

            # 6. Mitsuba XML
            xml_path = output_dir / "scene.xml"
            save_mitsuba_xml(ply_paths, xml_path)

            # 7. 3D プレビュー HTML
            save_scene_preview(
                scene_dir=output_dir,
                html_path=output_dir / "scene.html",
                title=f"PLATEAU 3D Scene: {mesh_code}",
            )

            # 8. TX 位置を bldg.ply から自動決定
            bldg_ply_path = output_dir / "bldg.ply"
            tx_positions = build_tx_positions(
                bldg_ply_path=bldg_ply_path,
                num_tx=sionna_cfg.num_tx,
                min_separation_m=sionna_cfg.min_separation_m,
                area_size_m=scene_cfg.area_size_m,
            )

            # 9. Sionna RT シミュレーション
            scene, mesh_radio_map, planar_radio_map = build_radio_maps(
                xml_path=xml_path,
                tx_positions=tx_positions,
                cfg=sionna_cfg,
                area_size_m=scene_cfg.area_size_m,
            )

            # 10. 結果を PNG / npz として保存
            save_radio_maps(
                scene=scene,
                mesh_radio_map=mesh_radio_map,
                planar_radio_map=planar_radio_map,
                tx_positions=tx_positions,
                cell_size_m=sionna_cfg.cell_size_m,
                noise_std_db=sionna_cfg.noise_std_db,
                rng=rng,
                area_size_m=scene_cfg.area_size_m,
                bldg_footprint_ply_path=output_dir / "bldg_footprint.ply",
                output_dir=output_dir,
            )

            logger.info("Done: %s", output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <scene.yaml> <sionna.yaml>")
        sys.exit(1)

    main(Path(sys.argv[1]), Path(sys.argv[2]))
