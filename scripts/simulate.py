"""
PLATEAU シーン構築 + Sionna RT シミュレーションの統合エントリポイント

使い方:
    uv run scripts/simulate.py configs/plateau.yaml configs/sionna.yaml

処理の流れ:
    configs/scene.yaml + configs/sionna.yaml
        → city x mesh_code の全組み合わせをループ
        → build_area_spec()        : AreaSpec 構築
        → build_*_mesh()           : PLY 生成・保存
        → save_mitsuba_xml()       : scene.xml 生成
        → save_scene_preview()     : scene.html 生成
        → build_tx_positions()     : TX 位置を bldg.ply から自動決定
        → build_radio_maps()       : Sionna RT シミュレーション (周波数リスト)
        → save_radio_maps()        : PNG / npz 保存 (周波数ごとのサブディレクトリ)

出力先: data/processed/<city_dir>/<mesh_code>/<freq_GHz>/
    例:  data/processed/tokyo/53393599/2.0GHz/
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
from radio_map_estimation.sionna.bldg_mask import build_bldg_mask
from radio_map_estimation.sionna.radiomap import build_radio_maps
from radio_map_estimation.sionna.save_radiomap import save_radio_maps
from radio_map_estimation.sionna.tx_placement import build_tx_positions
from radio_map_estimation.utils.visualize import save_rss_png

# from radio_map_estimation.scene.scene_preview import save_scene_preview

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
    center_search_radius_m: float
    min_separation_m: float
    tx_power_dbm: float
    tx_pattern: str
    tx_tilt_deg: float
    rx_height_m: float
    frequency_hz: tuple[float, ...]  # 複数周波数に対応
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
        # frequency_hz はスカラーでもリストでも tuple[float, ...] に統一する
        raw_freq = cfg.frequency_hz
        if isinstance(raw_freq, (int, float)):
            frequency_hz: tuple[float, ...] = (float(raw_freq),)
        else:
            frequency_hz = tuple(float(f) for f in raw_freq)
        if not frequency_hz:
            raise ValueError("frequency_hz must not be empty")
        return cls(
            num_tx=num_tx,
            center_search_radius_m=float(cfg.center_search_radius_m),
            min_separation_m=float(cfg.min_separation_m),
            tx_power_dbm=float(cfg.tx_power_dbm),
            tx_pattern=tx_pattern,
            tx_tilt_deg=float(cfg.tx_tilt_deg),
            rx_height_m=float(cfg.rx_height_m),
            frequency_hz=frequency_hz,
            cell_size_m=float(cfg.cell_size_m),
            num_samples=int(cfg.num_samples),
            max_depth=int(cfg.max_depth),
            noise_std_db=float(cfg.noise_std_db),
        )


@dataclass(frozen=True)
class AreaConfig:
    city_code: str
    city_name: str
    year: int
    city_dir: str
    mesh_codes: tuple[str, ...]
    citygml_url: str


@dataclass(frozen=True)
class PlateauConfig:
    areas: tuple[AreaConfig, ...]
    output_format: str
    area_size_m: float
    margin_m: float
    bldg_cell_size_m: float

    @classmethod
    def from_omega(cls, cfg: DictConfig) -> PlateauConfig:
        fmt = cfg.get("output_format", "geoparquet")
        if fmt not in ("geoparquet", "geojson"):
            raise ValueError(f"output_format must be 'geoparquet' or 'geojson', got {fmt!r}")
        return cls(
            areas=tuple(
                AreaConfig(
                    city_code=str(a.city_code),
                    city_name=str(a.city_name),
                    year=int(a.year),
                    city_dir=str(a.city_dir),
                    mesh_codes=tuple(str(c) for c in a.mesh_codes),
                    citygml_url=str(a.citygml_url),
                )
                for a in cfg.areas
            ),
            output_format=fmt,
            area_size_m=float(cfg.area_size_m),
            margin_m=float(cfg.margin_m),
            bldg_cell_size_m=float(cfg.bldg_cell_size_m),
        )


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


def _freq_dir_name(freq_hz: float) -> str:
    """周波数 [Hz] からディレクトリ名を生成する。例: 2.0e9 → '2.0GHz'"""
    ghz = freq_hz / 1e9
    # 小数点以下の不要なゼロを除去しつつ最低1桁は残す
    # 例: 2.0e9 → "2.0GHz", 3.5e9 → "3.5GHz", 10.0e9 → "10.0GHz"
    return f"{ghz:.10g}GHz" if ghz != int(ghz) else f"{ghz:.1f}GHz"


# ---------------------------------------------------------------------------
# mesh_code → origin
# ---------------------------------------------------------------------------


def meshcode_to_origin(mesh_code: str) -> tuple[float, float]:
    """8桁の3次メッシュコードから南西端の (lon, lat) を返す"""
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


def main(plateau_config_path: Path, sionna_config_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    processed_dir = data_dir / "processed"

    plateau_cfg = PlateauConfig.from_omega(OmegaConf.load(plateau_config_path))  # type: ignore
    sionna_cfg = SionnaConfig.from_omega(OmegaConf.load(sionna_config_path))  # type: ignore

    # 乱数シードはエントリポイントで1回だけ固定する
    rng = np.random.default_rng(seed=42)

    for area in plateau_cfg.areas:
        bldg_parquet, dem_parquet, tran_parquet, wtr_parquet = _parquet_paths(data_dir, area.city_dir)

        for mesh_code in area.mesh_codes:
            logger.info("Processing: city=%s, mesh_code=%s", area.city_dir, mesh_code)
            output_dir = processed_dir / area.city_dir / mesh_code
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
            area_spec = build_area_spec(origin_lon, origin_lat, plateau_cfg.area_size_m, plateau_cfg.margin_m)

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

            # 7. TX 位置を bldg.ply から自動決定
            bldg_ply_path = output_dir / "bldg.ply"
            tx_positions = build_tx_positions(
                bldg_ply_path=bldg_ply_path,
                num_tx=sionna_cfg.num_tx,
                center_search_radius_m=sionna_cfg.center_search_radius_m,
                min_separation_m=sionna_cfg.min_separation_m,
                area_size_m=plateau_cfg.area_size_m,
            )

            # 建物マスク (True = 建物上) + 保存
            bldg_mask = build_bldg_mask(
                bldg_footprint_ply_path=output_dir / "bldg_footprint.ply",
                cfg=plateau_cfg,
                output_dir=output_dir,
            )

            # 建物マスクのみ可視化 (高解像度、bldg_cell_size_m 基準)
            save_rss_png(
                tx_coords=np.array(tx_positions),
                area_size_m=plateau_cfg.area_size_m,
                output_path=output_dir / "bldg_mask.png",
                title="Building mask",
                rss_dbm=None,
                bldg_mask=bldg_mask,
            )

            # 3D プレビュー HTML
            # save_scene_preview(
            # scene_dir=output_dir,
            # html_path=output_dir / "scene.html",
            # title=f"PLATEAU 3D Scene: {mesh_code}",
            # )

            # 8. Sionna RT シミュレーション (全周波数)
            # シーンロード・TX 配置は build_radio_maps 内で1回のみ実行される
            # 周波数ごとに scene.frequency を差し替えて RadioMapSolver を再実行する
            radio_map_results = build_radio_maps(
                xml_path=xml_path,
                tx_positions=tx_positions,
                cfg=sionna_cfg,
                area_size_m=plateau_cfg.area_size_m,
            )

            # 9. 周波数ごとに結果を PNG / npz として保存
            for freq_hz, scene, mesh_radio_map, planar_radio_map, rss_dbm in radio_map_results:
                freq_dir = output_dir / _freq_dir_name(freq_hz)
                freq_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Saving radio maps: %s", freq_dir)

                save_radio_maps(
                    scene=scene,
                    mesh_radio_map=mesh_radio_map,
                    planar_radio_map=planar_radio_map,
                    rss_dbm=rss_dbm,
                    tx_positions=tx_positions,
                    freq_hz=freq_hz,
                    cfg=sionna_cfg,
                    area_size_m=plateau_cfg.area_size_m,
                    bldg_mask=bldg_mask,
                    rng=rng,
                    output_dir=freq_dir,
                )
                logger.info("Done: %s", freq_dir)

            logger.info("All frequencies done: %s", output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <scene.yaml> <sionna.yaml>")
        sys.exit(1)

    main(Path(sys.argv[1]), Path(sys.argv[2]))
