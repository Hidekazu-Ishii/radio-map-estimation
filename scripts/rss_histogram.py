"""
正規化済み rss_gt_norm のヒストグラムを可視化する.
0.0 と 1.0 への集中しすぎていないかを確認する.

使い方:
    uv run scripts/plot_rss_histogram.py --config configs/cities/{city_name}.yaml
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from radio_map_estimation.schema import load_area_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    area_cfg = load_area_config(args.config)

    area_id = f"{area_cfg.center_lat:.4f}_{area_cfg.center_lon:.4f}"
    processed_dir = root / "data" / "processed" / area_id

    # stats の読み込み
    stats = np.load(processed_dir / "stats.npz")
    rss_lower = float(stats["rss_lower_dbm"])
    rss_upper = float(stats["rss_upper_dbm"])
    lower_pct = float(stats["lower_percentile"])
    upper_pct = float(stats["upper_percentile"])

    # 全 trial の rss_gt_norm を収集
    values: list[np.ndarray] = []
    for trial_idx in range(area_cfg.n_trials):
        npz = np.load(processed_dir / f"{trial_idx:03d}.npz")
        values.append(npz["rss_gt_norm"].ravel())
    all_values = np.concatenate(values)

    # クリップの割合を計算
    lower_clip_ratio = float((all_values == 0.0).mean()) * 100
    upper_clip = float((all_values == 1.0).mean()) * 100

    # ヒストグラム描画
    _, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_values, bins=100, color="steelblue", edgecolor="none")
    ax.axvline(0.0, color="blue", linestyle="--", label=f"lower_clip (0.0): {lower_clip_ratio:.1f}%")
    ax.axvline(1.0, color="red", linestyle="--", label=f"ceil (1.0): {upper_clip:.1f}%")
    ax.set_xlabel("rss_gt_norm")
    ax.set_ylabel("count")
    ax.set_title(
        f"RSS Histogram ({area_id})\n"
        f"lower={rss_lower:.1f} dBm ({lower_pct}%ile), "
        f"upper={rss_upper:.1f} dBm ({upper_pct}%ile)"
    )
    ax.legend()
    plt.tight_layout()

    save_path = processed_dir / "rss_histogram.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] saved → {save_path}")
    print(f"  lower_clip (0.0): {lower_clip_ratio:.1f}%")
    print(f"  ceil  (1.0): {upper_clip:.1f}%")


if __name__ == "__main__":
    main()
