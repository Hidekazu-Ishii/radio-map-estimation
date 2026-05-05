"""
正規化処理: ray tracing で生成した rss_best / rss_observed を [0, 1] に正規化する.

正規化方針:
    - 下限: 全 trial の rss_best の 5 パーセンタイル (global) → 0.0 にクリップ
    - 上限: 全 trial の rss_best の 95 パーセンタイル (global) → 1.0 にクリップ
    - rss_observed も同じ統計量でクリップ・正規化
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def compute_rss_bounds(
    sim_dir: Path,
    n_trials: int,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[float, float]:
    """全 trial の rss_best から下限・上限パーセンタイルを計算する.

    Parameters
    ----------
    sim_dir : Path
        data/simulation/{lat}_{lon}/ ディレクトリ.
    n_trials : int
        trial 数.
    lower_percentile : float
        下限パーセンタイル (例: 5.0).
    upper_percentile : float
        下限パーセンタイル (例: 95.0).

    Returns
    -------
    float
        rss_percentile_dbm [dBm].
    """
    rss_values: list[NDArray[np.floating]] = []
    for trial_idx in range(n_trials):
        npz = np.load(sim_dir / f"{trial_idx:03d}.npz")
        arr = npz["rss_best"].ravel()
        rss_values.append(arr[np.isfinite(arr)])

    all_values = np.concatenate(rss_values)
    rss_lower_dbm = float(np.percentile(all_values, lower_percentile))
    rss_upper_dbm = float(np.percentile(all_values, upper_percentile))
    return rss_lower_dbm, rss_upper_dbm


def clip_rss_dbm(
    rss_dbm: np.ndarray,
    rss_lower_dbm: float,
    rss_upper_dbm: float,
) -> np.ndarray:
    """RSS [dBm] を [rss_lower_dbm, rss_upper_dbm] にクリップする."""
    return np.clip(rss_dbm, rss_lower_dbm, rss_upper_dbm)


def normalize_rss(
    rss_dbm: np.ndarray,
    rss_lower_dbm: float,
    rss_upper_dbm: float,
) -> np.ndarray:
    """RSS [dBm] を [0, 1] に正規化する."""
    normalized = (rss_dbm - rss_lower_dbm) / (rss_upper_dbm - rss_lower_dbm)
    return normalized.astype(np.float32)


def validate_trial(npz_path: Path) -> None:
    """保存済み npz に nan/inf がなく. 正規化済み配列の値域が [0, 1] であることを確認する."""
    npz = np.load(npz_path)

    # nan/inf チェック
    for key in ["rss_gt_norm", "rss_observed_norm", "rss_gt", "rss_observed"]:
        if not np.isfinite(npz[key]).all():
            raise ValueError(f"[validate] {npz_path.name}: '{key}' に nan/inf が含まれています")

    # 値域チェック
    for key in ["rss_gt_norm", "rss_observed_norm"]:
        arr = npz[key]
        if arr.min() < 0.0 or arr.max() > 1.0:
            raise ValueError(
                f"[validate] {npz_path.name}: '{key}' の値域が [0, 1] を外れています "
                f"(min={arr.min():.4f}, max={arr.max():.4f})"
            )


def process_trial(
    trial_idx: int,
    sim_dir: Path,
    out_dir: Path,
    rss_lower_dbm: float,
    rss_upper_dbm: float,
) -> None:
    """1 trial 分の npz を正規化して保存する."""
    npz = np.load(sim_dir / f"{trial_idx:03d}.npz")

    # rss_best, rss_observed を [rss_lower_dbm, rss_upper_dbm] にクリップ
    rss_gt = npz["rss_best"]
    rss_gt = np.where(np.isnan(rss_gt), rss_lower_dbm, rss_gt)  # nan → 下限値
    rss_gt = clip_rss_dbm(rss_gt, rss_lower_dbm, rss_upper_dbm)
    rss_observed = npz["rss_observed"]
    rss_observed = np.where(np.isnan(rss_observed), rss_lower_dbm, rss_observed)  # nan → 下限値
    rss_observed = clip_rss_dbm(rss_observed, rss_lower_dbm, rss_upper_dbm)

    np.savez(
        out_dir / f"{trial_idx:03d}.npz",
        rss_gt_norm=normalize_rss(rss_gt, rss_lower_dbm, rss_upper_dbm),
        rss_observed_norm=normalize_rss(rss_observed, rss_lower_dbm, rss_upper_dbm),
        rss_gt=rss_gt,
        rss_observed=rss_observed,
        tx_locations=npz["tx_locations"],
    )
    validate_trial(out_dir / f"{trial_idx:03d}.npz")


def save_stats(
    out_dir: Path,
    rss_lower_dbm: float,
    rss_upper_dbm: float,
    lower_percentile: float,
    upper_percentile: float,
) -> None:
    """正規化統計量を保存する (逆正規化に使用)."""
    np.savez(
        out_dir / "stats.npz",
        rss_lower_dbm=rss_lower_dbm,
        rss_upper_dbm=rss_upper_dbm,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    print(
        f"[stats] rss_lower={rss_lower_dbm:.1f} dBm ({lower_percentile}%ile), rss_upper={rss_upper_dbm:.2f} dBm ({upper_percentile}%ile)"
    )
