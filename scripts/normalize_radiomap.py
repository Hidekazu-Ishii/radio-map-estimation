"""
エントリポイント: data/cities/{center_lat}_{center_lon}/ 下にある npz ファイルの
rss_gt, rss_observed を正規化して data/processed/ に保存する.
configs/cities/{city}.yaml と configs/normalize.yaml を読み込んで実行する.

使い方:
    uv run scripts/normalize_radiomap.py --config configs/cities/{city_name}.yaml
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from radio_map_estimation.schema import load_area_config
from radio_map_estimation.simulater.normalize import (
    compute_rss_bounds,
    process_trial,
    save_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    area_cfg = load_area_config(args.config)
    norm_cfg = OmegaConf.load(root / "configs" / "normalize.yaml")

    area_id = f"{area_cfg.center_lat:.4f}_{area_cfg.center_lon:.4f}"
    sim_dir = root / "data" / "simulation" / area_id
    out_dir = root / "data" / "processed" / area_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: global 統計量の計算 ---
    print("[1/2] Computing global rss_bounds...")
    rss_lower_dbm, rss_upper_dbm = compute_rss_bounds(
        sim_dir=sim_dir,
        n_trials=area_cfg.n_trials,
        lower_percentile=norm_cfg.lower_percentile,
        upper_percentile=norm_cfg.upper_percentile,
    )
    save_stats(out_dir, rss_lower_dbm, rss_upper_dbm, norm_cfg.lower_percentile, norm_cfg.upper_percentile)

    # --- Step 2: 各 trial を正規化して保存 ---
    print(f"[2/2] Normalizing {area_cfg.n_trials} trials...")
    for trial_idx in range(area_cfg.n_trials):
        process_trial(
            trial_idx=trial_idx,
            sim_dir=sim_dir,
            out_dir=out_dir,
            rss_lower_dbm=rss_lower_dbm,
            rss_upper_dbm=rss_upper_dbm,
        )
        print(f"  [{trial_idx + 1}/{area_cfg.n_trials}] {trial_idx:03d}.npz → done")

    print(f"\n[done] Processed data saved to {out_dir}")


if __name__ == "__main__":
    main()
