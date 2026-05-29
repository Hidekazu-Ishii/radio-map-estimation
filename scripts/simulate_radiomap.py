# scripts/simulate_radiomap.py
"""
エントリポイント: 建物取得 + 補完処理 → Sionna RT でマルチ TX 電波マップ生成.

建物ソース:
    --source osm      OSMnx 経由 (デフォルト, ネット接続必要)
    --source citygml  CityGML ファイル経由 (PLATEAU / LoD2-DE)

Usage:
    uv run scripts/simulate_radiomap.py --city berlin
    uv run scripts/simulate_radiomap.py --city tokyo \\
        --source citygml --citygml-dir /data/plateau/tokyo/udx/bldg
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator, default_rng
from radio_map_estimation.scene.citygml_buildings import fetch_buildings_citygml
from radio_map_estimation.scene.datasource.osm_buildings import fetch_buildings_osm

from radio_map_estimation.osm.mesh_builder import (
    build_building_meshes,
    build_ground_mesh,
    save_meshes_to_obj,
)
from radio_map_estimation.osm.osm_schema import (
    BuildingData,
    BuildingHeightConfig,
    CityConfig,
    load_building_height_config,
    load_city_config,
    save_building_data,
)
from radio_map_estimation.osm.scene_writer import write_mitsuba_xml
from radio_map_estimation.simulate.observation_noise import (
    add_observation_noise,
)
from radio_map_estimation.simulate.radiomap_solver import (
    radio_map_to_rss_dbm,
    run_radio_map,
)
from radio_map_estimation.simulate.schema import (
    SimulationConfig,
    load_simulation_config,
)
from radio_map_estimation.simulate.tx_placement import place_tx_ppp
from radio_map_estimation.simulate.visualize import (
    plot_radio_map,
    plot_tx_association,
)
from radio_map_estimation.simulate.visualize_sionna import (
    save_figure,
    show_building_data,
    show_radio_map,
    show_tx_association,
)

# Sionna が型スタブを提供しないため Any で受ける
# PlanarRadioMap の実体は sionna.rt.PlanarRadioMap
PlanarRadioMap = Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        help="都市名 (configs/cities/{city}.yaml に対応)",
    )
    parser.add_argument(
        "--source",
        choices=["osm", "citygml"],
        default="osm",
        help="建物データソース (osm: OSMnx, citygml: CityGML ファイル)",
    )
    parser.add_argument(
        "--citygml-dir",
        type=Path,
        default=None,
        help="CityGML ファイルのディレクトリ (--source citygml 時に必須)",
    )
    parser.add_argument(
        "--citygml-source",
        choices=["plateau", "lod2de"],
        default="plateau",
        help="CityGML データ種別 (plateau=東京PLATEAU, lod2de=独LoD2)",
    )
    parser.add_argument(
        "--citygml-glob",
        default="**/*.gml",
        help="CityGML ファイルの glob パターン (default: **/*.gml)",
    )
    return parser.parse_args()


def _run_trial(
    trial_idx: int,
    rng: Generator,
    city_cfg: CityConfig,
    sim_cfg: SimulationConfig,
    building_data: BuildingData,
    scene_xml: Path,
    npz_dir: Path,
    rss_dir: Path,
    assoc_dir: Path,
    radiomap_dir: Path,
    tx_dir: Path,
) -> None:
    """
    1 trial 分の処理: TX 配置 → ray tracing → 保存.
    """
    npz_dir.mkdir(parents=True, exist_ok=True)
    rss_dir.mkdir(parents=True, exist_ok=True)
    assoc_dir.mkdir(parents=True, exist_ok=True)
    radiomap_dir.mkdir(parents=True, exist_ok=True)
    tx_dir.mkdir(parents=True, exist_ok=True)

    # 制約付き PPP で TX 配置
    tx_positions = place_tx_ppp(
        rng=rng,
        area_size_m=city_cfg.area_size_m,
        intensity=sim_cfg.tx_intensity,
        building_heights=building_data.building_heights,
        cell_size_m=city_cfg.cell_size_m,
        building_mask=building_data.building_mask,
        min_open_neighbors=sim_cfg.min_open_neighbors,
        min_separation_m=sim_cfg.min_separation_m,
        tx_height_above_building_m=sim_cfg.tx_height_above_building_m,
        min_building_height_m=sim_cfg.min_building_height_m,
        max_building_height_m=sim_cfg.max_building_height_m,
        height_weight_power=sim_cfg.height_weight_power,
        inner_margin_m=sim_cfg.inner_margin_m,
    )

    tx_positions_list = [(float(x), float(y), float(z)) for x, y, z in tx_positions]
    # ray tracing
    radio_map = run_radio_map(
        scene_xml=scene_xml,
        tx_positions=tx_positions_list,
        frequency_hz=sim_cfg.frequency_hz,
        tx_power_dbm=sim_cfg.tx_power_dbm,
        building_data=building_data,
        rx_height_m=sim_cfg.rx_height_m,
        max_depth=sim_cfg.max_depth,
        cell_size_m=city_cfg.cell_size_m,
        samples_per_tx=sim_cfg.samples_per_tx,
        seed=sim_cfg.master_seed + trial_idx,
    )

    # numpy 変換
    rss_dbm_per_tx = radio_map_to_rss_dbm(radio_map)
    # best-server: 全 TX の最大値
    rss_gt = np.max(rss_dbm_per_tx, axis=0)  # (H, W)

    # 観測ノイズ付加
    rss_observed = add_observation_noise(
        rss_dbm=rss_gt,
        noise_std_db=sim_cfg.noise_std_db,
        rng=rng,
    )

    # 保存
    np.savez(
        npz_dir / f"{trial_idx:03d}.npz",
        rss_gt=rss_gt,
        rss_observed=rss_observed,
        rss_dbm_per_tx=rss_dbm_per_tx,
        tx_positions=tx_positions,
    )

    # 可視化 (Sionna 公式 API)
    save_figure(
        show_radio_map(radio_map, metric="rss", tx=None),
        rss_dir / f"{trial_idx:03d}.png",
    )
    save_figure(
        show_tx_association(radio_map, metric="rss"),
        assoc_dir / f"{trial_idx:03d}.png",
    )

    # 可視化 (自作)
    plot_radio_map(
        rss_dbm=rss_gt,
        building_mask=building_data.building_mask,
        area_size_m=building_data.area_spec.area_size_m,
        tx_positions=tx_positions,
        frequency_hz=sim_cfg.frequency_hz,
        save_path=radiomap_dir / f"{trial_idx:03d}.png",
    )
    plot_tx_association(
        rss_dbm_per_tx=rss_dbm_per_tx,
        building_mask=building_data.building_mask,
        area_size_m=building_data.area_spec.area_size_m,
        tx_positions=tx_positions,
        save_path=tx_dir / f"{trial_idx:03d}.png",
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    # --- 設定読み込み ---
    city_cfg: CityConfig = load_city_config(root / "configs" / "cities" / f"{args.city}.yaml")
    bh_cfg: BuildingHeightConfig = load_building_height_config(root / "configs" / "building_height.yaml")
    sim_cfg: SimulationConfig = load_simulation_config(root / "configs" / "simulation.yaml")

    # 出力ディレクトリ
    output_dir = root / "data" / "simulation" / f"{city_cfg.center_lat:.4f}_{city_cfg.center_lon:.4f}"
    scene_dir = output_dir / "scene"
    mesh_dir = scene_dir / "mesh"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- [共通] 建物取得 ---
    if args.source == "citygml":
        if args.citygml_dir is None:
            raise ValueError("--source citygml を指定する場合は --citygml-dir が必要です.")
        citygml_paths = sorted(args.citygml_dir.glob(args.citygml_glob))
        if not citygml_paths:
            raise FileNotFoundError(
                f"CityGML ファイルが見つかりません: {args.citygml_dir / args.citygml_glob}"
            )
        print(f"[1/3] Fetching buildings from CityGML ({len(citygml_paths)} files)...")
        building_data: BuildingData = fetch_buildings_citygml(
            citygml_paths=citygml_paths,
            center_lat=city_cfg.center_lat,
            center_lon=city_cfg.center_lon,
            area_size_m=city_cfg.area_size_m,
            cell_size_m=city_cfg.cell_size_m,
            source=args.citygml_source,
        )
    else:
        print("[1/3] Fetching buildings from OSM...")
        building_data = fetch_buildings_osm(
            center_lat=city_cfg.center_lat,
            center_lon=city_cfg.center_lon,
            area_size_m=city_cfg.area_size_m,
            cell_size_m=city_cfg.cell_size_m,
            meters_per_level=bh_cfg.meters_per_level,
        )
    save_figure(show_building_data(building_data), output_dir / "building_map.png")

    # --- [共通] 3D シーン構築 ---
    print("[2/3] Building 3D scene...")
    building_obj_strings = build_building_meshes(building_data=building_data)
    ground_mesh = build_ground_mesh(building_data=building_data)
    building_objs, ground_ply = save_meshes_to_obj(
        building_obj_strings=building_obj_strings,
        ground_mesh=ground_mesh,
        mesh_dir=mesh_dir,
    )
    scene_xml = write_mitsuba_xml(
        scene_dir=scene_dir,
        building_objs=building_objs,
        ground_ply=ground_ply,
        building_material=sim_cfg.building_material,  # str 一律
        ground_material=sim_cfg.ground_material,
    )

    # --- [trial ループ] ---
    rng: Generator = default_rng(sim_cfg.master_seed)

    npz_dir = output_dir / "npz"
    rss_dir = output_dir / "rss"
    assoc_dir = output_dir / "association"
    radiomap_dir = output_dir / "radiomap"
    tx_dir = output_dir / "tx"
    print(f"[3/3] Running {sim_cfg.n_trials} trials...")
    for trial_idx in range(sim_cfg.n_trials):
        print(f"\n--- Trial {trial_idx:03d} / {sim_cfg.n_trials} ---")
        _run_trial(
            trial_idx=trial_idx,
            rng=rng,
            city_cfg=city_cfg,
            sim_cfg=sim_cfg,
            building_data=building_data,
            scene_xml=scene_xml,
            npz_dir=npz_dir,
            rss_dir=rss_dir,
            assoc_dir=assoc_dir,
            radiomap_dir=radiomap_dir,
            tx_dir=tx_dir,
        )

    print(f"\n[done] All trials saved to {output_dir}")


if __name__ == "__main__":
    main()
