"""
エントリポイント: OSMnx で建物取得 → Sionna RT で電波マップ生成.
configs/simulation/{city_name}.yaml を読み込んで実行する.
建物マップ・電波マップは data/simulation/{lat}_{lon}/ に保存される.

使い方:
    uv run scripts/simulate_radiomap.py --config configs/cities/{city_name}.yaml
"""

import argparse
from pathlib import Path

import numpy as np
from numpy.random import Generator, default_rng

from radio_map_estimation.schema import AreaConfig, SimulationConfig, load_area_config, load_simulation_config
from radio_map_estimation.simulater.osm_buildings import (
    BuildingData,
    fetch_buildings_osm,
    plot_building_data,
    save_building_data,
)
from radio_map_estimation.simulater.radiomap import (
    build_building_meshes,
    build_ground_mesh,
    compute_best_server_map,
    place_tx_ppp,
    plot_radio_map,
    run_radio_map,
    save_meshes_to_ply,
    write_mitsuba_xml,
)


def _run_trial(
    trial_idx: int,
    rng: Generator,
    area_cfg: AreaConfig,
    sim_cfg: SimulationConfig,
    building_data: BuildingData,
    scene_xml: Path,
    output_dir: Path,
) -> None:
    """
    1 trial 分の処理: TX 配置 → TX ループ → best-server 合成 → 保存.

    rng は呼び出し元から受け渡す (trial 間で状態が連続する).
    """

    # PPP で TX 配置
    tx_locations: np.ndarray = place_tx_ppp(
        rng=rng,
        area_size_m=area_cfg.area_size_m,
        intensity=sim_cfg.tx_intensity,
    )
    n_tx = len(tx_locations)
    print(f"  TX count: {n_tx}")

    # TX ごとに ray tracing
    rss_dbm_list: list[np.ndarray] = []
    for tx_idx, (tx_x, tx_y) in enumerate(tx_locations):
        print(f"  [TX {tx_idx}] position=({tx_x:.1f}, {tx_y:.1f}, {sim_cfg.tx_height_m:.1f})")
        tx_position_local = (float(tx_x), float(tx_y), float(sim_cfg.tx_height_m))
        rss_dbm = run_radio_map(
            scene_xml=scene_xml,
            tx_position_local=tx_position_local,
            frequency_hz=sim_cfg.frequency_hz,
            tx_power_dbm=sim_cfg.tx_power_dbm,
            building_data=building_data,
            rx_height_m=sim_cfg.rx_height_m,
            max_depth=sim_cfg.max_depth,
            cell_size_m=area_cfg.cell_size_m,
            samples_per_tx=sim_cfg.samples_per_tx,
        )
        rss_dbm_list.append(rss_dbm)

    # rss_best / rss_observed は nan (全TX未到達セル) を許容. 後の正規化処理で rss_lower_dbm に置換.
    # best-server マップを合成
    rss_best: np.ndarray = compute_best_server_map(rss_dbm_list)

    # 観測ノイズの付加 (真値 rss_best に N(0, noise_std_db^2) を加える)
    rss_observed = rss_best + rng.normal(0.0, sim_cfg.noise_std_db, size=rss_best.shape)

    # 保存: rss_observed, 全 TX の rss, TX 座標
    np.savez(
        output_dir / f"{trial_idx:03d}.npz",
        rss_best=rss_best,
        rss_observed=rss_observed,
        rss_dbm_per_tx=np.stack(rss_dbm_list, axis=0),  # (T, H, W)
        tx_locations=tx_locations,  # (T, 2)
    )
    plot_radio_map(
        rss_dbm=rss_best,
        building_mask=building_data.building_mask,
        area_size_m=building_data.area_spec.area_size_m,
        tx_locations=tx_locations,
        frequency_hz=sim_cfg.frequency_hz,
        save_path=output_dir / f"{trial_idx:03d}.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    area_cfg = load_area_config(args.config)
    sim_cfg = load_simulation_config(root / "configs" / "simulation.yaml")

    output_dir = root / "data" / "simulation" / f"{area_cfg.center_lat:.4f}_{area_cfg.center_lon:.4f}"
    scene_dir = output_dir / "scene"
    mesh_dir = scene_dir / "mesh"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- OSMnx で建物取得 ---
    print("[1/3] Fetching buildings from OSM...")
    building_data: BuildingData = fetch_buildings_osm(
        center_lat=area_cfg.center_lat,
        center_lon=area_cfg.center_lon,
        area_size_m=area_cfg.area_size_m,
        cell_size_m=area_cfg.cell_size_m,
        meters_per_level=area_cfg.meters_per_level,
        default_building_height_m=area_cfg.default_building_height_m,
        building_type_levels=area_cfg.building_type_levels,
    )
    save_building_data(data=building_data, output_dir=output_dir)
    plot_building_data(data=building_data, save_path=output_dir / "building_map.png")

    # --- 3D シーン構築 ---
    print("[2/3] Building 3D scene...")
    building_meshes = build_building_meshes(building_data=building_data)
    ground_mesh = build_ground_mesh(building_data=building_data)
    building_plys, ground_ply = save_meshes_to_ply(
        building_meshes=building_meshes,
        ground_mesh=ground_mesh,
        mesh_dir=mesh_dir,
    )
    scene_xml = write_mitsuba_xml(
        scene_dir=scene_dir,
        building_plys=building_plys,
        ground_ply=ground_ply,
        building_material=sim_cfg.building_material,
        ground_material=sim_cfg.ground_material,
    )

    # --- trial ループ ---
    rng: Generator = default_rng(area_cfg.master_seed)

    print(f"[3/3] Running {area_cfg.n_trials} trials...")
    for trial_idx in range(area_cfg.n_trials):
        print(f"\n--- Trial {trial_idx} / {area_cfg.n_trials} ---")
        _run_trial(
            trial_idx=trial_idx,
            rng=rng,
            area_cfg=area_cfg,
            sim_cfg=sim_cfg,
            building_data=building_data,
            scene_xml=scene_xml,
            output_dir=output_dir,
        )

    print(f"\n[done] All trials saved to {output_dir}")


if __name__ == "__main__":
    main()
