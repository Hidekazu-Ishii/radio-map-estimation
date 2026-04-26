"""
scripts/generate_radiomap.py

目的: ラジオマップを生成するエントリポイント.
      configs/radiomap.yaml を読み込んで src/radiomap.py の処理を呼び出す.

Usage:
    uv run scripts/sim_data/generate_radiomap.py
    uv run scripts/sim_data/generate_radiomap.py --config configs/radiomap.yaml
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from radio_map_estimation.generater.radiomap import (
    generate_rss_map,
    generate_shadowing_field,
    load_building_map,
    place_bs_ppp,
    plot_results,
)


@dataclass(frozen=True)
class ConditionConfig:
    frequency_hz: float
    bs_intensity: float
    sigma_shadow_db: float
    corr_distance: float
    noise_std_db: float


@dataclass(frozen=True)
class RadiomapConfig:
    n_trials: int
    master_seed: int
    tx_power_dbm: float
    bs_height_m: float
    ue_height_m: float
    n_sample: int
    locations: list[tuple[float, float]]
    conditions: list[ConditionConfig]


def load_config(config_path: Path) -> RadiomapConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw["conditions"] = [ConditionConfig(**c) for c in raw["conditions"]]
    return RadiomapConfig(**raw)


# OSM建物マップ + 簡易電波伝播モデルで、指定エリアのラジオマップを生成して保存する.


def main(
    seed: int,
    load_dir: Path,
    save_dir: Path,
    # --- よく変えるパラメータ (実験条件) ---
    frequency_hz: float,
    bs_intensity: float,
    sigma_shadow_db: float,
    corr_distance: float,
    noise_std_db: float,
    # --- 基本固定パラメータ ---
    tx_power_dbm: float,
    bs_height_m: float,
    ue_height_m: float,
    n_sample: int,
) -> None:
    rng = np.random.default_rng(seed)

    print("=== Radio Map Data Generation ===")

    # 建物マップの読み込み
    print(f"[1/4] Loading building map from '{load_dir}' ...")
    building_mask, building_heights, area_size, grid_size = load_building_map(load_dir)
    print(
        f"      area={area_size:.0f}m, grid={grid_size}x{grid_size}, "
        f"building coverage={building_mask.mean() * 100:.1f}%"
    )

    # 基地局配置
    print("[2/4] Placing BSs (PPP)...")
    bs_locations = place_bs_ppp(rng, area_size, bs_intensity)
    print(f"      Number of BSs: {len(bs_locations)}")

    # グリッド座標の生成
    dx = area_size / grid_size
    xs = np.arange(grid_size) * dx + dx / 2
    ys = np.arange(grid_size) * dx + dx / 2
    xx, yy = np.meshgrid(xs, ys)
    grid_coords = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (P, 2)

    # RSSマップの生成
    print("[3/4] Computing RSS maps (LOS/NLOS path loss + shadowing)...")
    rss_all_bs: list[np.ndarray] = []
    is_los_all_bs: list[np.ndarray] = []

    for b_idx, bs_xy in enumerate(bs_locations):
        print(f"      BS {b_idx + 1}/{len(bs_locations)} ...")
        shadowing = generate_shadowing_field(rng, grid_size, area_size, sigma_shadow_db, corr_distance)
        rss, is_los = generate_rss_map(
            rng,
            bs_xy,
            bs_height_m,
            grid_coords,
            ue_height_m,
            shadowing,
            building_heights,
            area_size,
            tx_power_dbm,
            frequency_hz,
            n_sample,
        )
        rss_all_bs.append(rss)
        is_los_all_bs.append(is_los)

    rss_all_bs_arr = np.stack(rss_all_bs, axis=0)  # (B, P)
    is_los_all_bs_arr = np.stack(is_los_all_bs, axis=0)  # (B, P)

    # best-server: RSSが最大のBSを選択
    best_bs_idx = rss_all_bs_arr.argmax(axis=0)  # (P,)
    rss_best = rss_all_bs_arr.max(axis=0)  # (P,)
    is_los_best = is_los_all_bs_arr[best_bs_idx, np.arange(len(best_bs_idx))]  # (P,)

    # 観測ノイズの付加 (真値 rss_best に N(0, noise_std_db^2) を加える)
    rss_observed = rss_best + rng.normal(0.0, noise_std_db, size=rss_best.shape)

    # データの保存
    print("[4/4] Saving results...")
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_dir,
        building_mask=building_mask,  # (grid_size, grid_size) bool
        bs_locations=bs_locations,  # (B, 2)   float [m]
        grid_coords=grid_coords,  # (P, 2)   float [m]
        rss_all_bs=rss_all_bs_arr,  # (B, P)   float [dBm]  真値
        is_los_all_bs=is_los_all_bs_arr,  # (B, P)   bool
        rss_best=rss_best,  # (P,)     float [dBm]  真値
        is_los_best=is_los_best,  # (P,)     bool
        rss_observed=rss_observed,  # (P,)     float [dBm]  観測値
        area_size=np.array(area_size),
        grid_size=np.array(grid_size),
        seed=np.array(seed),
        noise_std_db=np.array(noise_std_db),
        building_map_path=np.array(str(load_dir)),
    )
    print(f"      Data saved → {save_dir}")

    # 結果のプロット
    plot_path = save_dir.with_suffix(".png")
    plot_results(building_mask, bs_locations, rss_best, is_los_best, area_size, plot_path)

    # サマリーの表示
    los_ratio = is_los_best.mean() * 100
    print("\n--- Summary ---")
    print(f"  Building map : {load_dir}")
    print(f"  Grid         : {grid_size}x{grid_size} ({area_size:.0f}m x {area_size:.0f}m)")
    print(f"  BSs          : {len(bs_locations)}")
    print(f"  RSS range    : {rss_best.min():.1f} ~ {rss_best.max():.1f} dBm")
    print(f"  LOS ratio    : {los_ratio:.1f}%")
    print(f"  Building coverage: {building_mask.mean() * 100:.1f}%")


def run(config_path: Path) -> None:
    cfg = load_config(config_path)

    root = Path(__file__).resolve().parents[2]
    load_dir = root / "data" / "buildings"
    simdata_dir = root / "data" / "simulation"

    master_rng = np.random.default_rng(cfg.master_seed)
    trial_seeds = master_rng.integers(int(1e9), size=cfg.n_trials)

    for lat, lon in cfg.locations:
        building_map_path = load_dir / f"{lat:.4f}_{lon:.4f}.npz"

        for cond in cfg.conditions:
            freq_ghz = cond.frequency_hz / 1e9
            cond_dir = simdata_dir / f"{lat:.4f}_{lon:.4f}" / f"{freq_ghz:.1f}GHz"
            print(f"\n[Condition] ({lat}, {lon})  {freq_ghz:.1f}GHz")

            for trial_idx, seed in enumerate(trial_seeds):
                main(
                    seed=int(seed),
                    load_dir=building_map_path,
                    save_dir=cond_dir / f"{trial_idx}.npz",
                    frequency_hz=cond.frequency_hz,
                    bs_intensity=cond.bs_intensity,
                    sigma_shadow_db=cond.sigma_shadow_db,
                    corr_distance=cond.corr_distance,
                    noise_std_db=cond.noise_std_db,
                    tx_power_dbm=cfg.tx_power_dbm,
                    bs_height_m=cfg.bs_height_m,
                    ue_height_m=cfg.ue_height_m,
                    n_sample=cfg.n_sample,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/radiomap.yaml"),
    )
    args = parser.parse_args()
    run(args.config)
