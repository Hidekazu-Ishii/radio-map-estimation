"""
エントリポイント: OSMnx で建物取得 → Sionna RT で電波マップ生成.
configs/simulation/{city_name}.yaml を読み込んで実行する.
建物マップ・電波マップは data/simulation/{lat}_{lon}/ に保存される.

使い方:
    uv run scripts/simulate_radiomap.py --config configs/simulation/{city_name}.yaml
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.random import Generator, default_rng
from omegaconf import OmegaConf

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


@dataclass
class RadioMapConfig:
    # エリア設定
    center_lat: float
    center_lon: float
    area_size_m: float
    cell_size_m: float
    # 建物高さ推定
    meters_per_level: float
    default_building_height_m: float
    building_type_levels: dict[str, float]
    # 送信機 (ローカル座標系: bbox左下が原点)
    tx_height_m: float
    tx_power_dbm: float
    tx_intensity: float
    # 電波設定
    frequency_hz: float
    rx_height_m: float
    # Ray Tracing
    max_depth: int
    samples_per_tx: int
    # 材質
    building_material: str
    ground_material: str
    # 観測ノイズ
    noise_std_db: float
    # 試行
    n_trials: int
    master_seed: int


def load_config(config_path: Path) -> RadioMapConfig:
    cfg = OmegaConf.load(config_path)
    return RadioMapConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]


def _run_trial(
    trial_idx: int,
    rng: Generator,
    cfg: RadioMapConfig,
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
        area_size_m=cfg.area_size_m,
        intensity=cfg.tx_intensity,
    )
    n_tx = len(tx_locations)
    print(f"  TX count: {n_tx}")

    # TX ごとに ray tracing
    rss_dbm_list: list[np.ndarray] = []
    for tx_idx, (tx_x, tx_y) in enumerate(tx_locations):
        print(f"  [TX {tx_idx}] position=({tx_x:.1f}, {tx_y:.1f}, {cfg.tx_height_m:.1f})")
        tx_position_local = (float(tx_x), float(tx_y), float(cfg.tx_height_m))
        rss_dbm = run_radio_map(
            scene_xml=scene_xml,
            tx_position_local=tx_position_local,
            frequency_hz=cfg.frequency_hz,
            tx_power_dbm=cfg.tx_power_dbm,
            building_data=building_data,
            rx_height_m=cfg.rx_height_m,
            max_depth=cfg.max_depth,
            cell_size_m=cfg.cell_size_m,
            samples_per_tx=cfg.samples_per_tx,
        )
        rss_dbm_list.append(rss_dbm)

    # best-server マップを合成
    rss_best: np.ndarray = compute_best_server_map(rss_dbm_list)

    # 観測ノイズの付加 (真値 rss_best に N(0, noise_std_db^2) を加える)
    rss_observed = rss_best + rng.normal(0.0, cfg.noise_std_db, size=rss_best.shape)

    # 保存: rss_observed, 全 TX の rss, TX 座標
    np.savez(
        output_dir / f"{trial_idx:03d}.npz",
        rss_gt=rss_best,  # Ground Truth
        rss_observed=rss_observed,  # 観測値
        rss_dbm_per_tx=np.stack(rss_dbm_list, axis=0),  # (T, H, W)
        tx_locations=tx_locations,  # (T, 2)
    )
    plot_radio_map(
        rss_dbm=rss_best,
        building_mask=building_data.building_mask,
        area_size_m=building_data.area_spec.area_size_m,
        tx_locations=tx_locations,
        frequency_hz=cfg.frequency_hz,
        save_path=output_dir / f"{trial_idx:03d}.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, required=True, help="simulation yaml (例: configs/simulation/berlin.yaml)"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)

    output_dir = root / "data" / "simulation" / f"{cfg.center_lat:.4f}_{cfg.center_lon:.4f}"
    scene_dir = output_dir / "scene"
    mesh_dir = scene_dir / "mesh"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- OSMnx で建物取得 ---
    print("[1/3] Fetching buildings from OSM...")
    building_data: BuildingData = fetch_buildings_osm(
        center_lat=cfg.center_lat,
        center_lon=cfg.center_lon,
        area_size_m=cfg.area_size_m,
        cell_size_m=cfg.cell_size_m,
        meters_per_level=cfg.meters_per_level,
        default_building_height_m=cfg.default_building_height_m,
        building_type_levels=cfg.building_type_levels,
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
        building_material=cfg.building_material,
        ground_material=cfg.ground_material,
    )

    # --- trial ループ ---
    rng: Generator = default_rng(cfg.master_seed)

    print(f"[3/3] Running {cfg.n_trials} trials...")
    for trial_idx in range(cfg.n_trials):
        print(f"\n--- Trial {trial_idx} / {cfg.n_trials} ---")
        _run_trial(
            trial_idx=trial_idx,
            rng=rng,
            cfg=cfg,
            building_data=building_data,
            scene_xml=scene_xml,
            output_dir=output_dir,
        )

    print(f"\n[done] All trials saved to {output_dir}")


if __name__ == "__main__":
    main()
