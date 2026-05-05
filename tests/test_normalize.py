"""normalize.py のスモークテスト."""

import numpy as np
import pytest

from radio_map_estimation.simulater.normalize import (
    clip_rss_dbm,
    compute_rss_bounds,
    normalize_rss,
    process_trial,
    save_stats,
)

# ---------------------------------------------------------------------------
# fixtures: 実験条件
# ---------------------------------------------------------------------------

GRID = 10
N_TRIALS = 3
LOWER = -110.0  # 下限 [dBm]
UPPER = -60.0  # 上限 [dBm]
LOWER_PCT = 5.0
UPPER_PCT = 95.0


@pytest.fixture
def sim_dir(tmp_path):
    """rss_best / rss_observed / tx_locations を含む npz を生成する."""
    rng = np.random.default_rng(seed=0)
    for i in range(N_TRIALS):
        rss_best = rng.uniform(LOWER, UPPER, size=(GRID, GRID)).astype(np.float32)
        np.savez(
            tmp_path / f"{i:03d}.npz",
            rss_best=rss_best,
            rss_observed=rss_best + rng.normal(0.0, 3.0, size=(GRID, GRID)).astype(np.float32),
            tx_locations=rng.uniform(0.0, 100.0, size=(3, 2)).astype(np.float32),
        )
    return tmp_path


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_clip_rss_dbm():
    # 下限未満 → LOWER, 上限超過 → UPPER, 範囲内はそのまま
    rss = np.array([[LOWER - 10.0, -80.0, UPPER + 10.0]], dtype=np.float32)
    out = clip_rss_dbm(rss, LOWER, UPPER)
    assert out[0, 0] == pytest.approx(LOWER)
    assert out[0, 1] == pytest.approx(-80.0)
    assert out[0, 2] == pytest.approx(UPPER)


def test_normalize_rss():
    # 下限 → 0.0, 上限 → 1.0
    rss = np.array([[LOWER, UPPER]], dtype=np.float32)
    out = normalize_rss(rss, LOWER, UPPER)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, 1] == pytest.approx(1.0)


def test_compute_rss_bounds(sim_dir):
    # 戻り値が float で [LOWER, UPPER] の範囲内であること
    lower, upper = compute_rss_bounds(sim_dir, N_TRIALS, LOWER_PCT, UPPER_PCT)
    assert isinstance(lower, float) and isinstance(upper, float)
    assert LOWER <= lower < upper <= UPPER


def test_process_trial(sim_dir, tmp_path):
    # 出力npzのキーと正規化済み配列の値域 [0, 1] を確認
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    process_trial(0, sim_dir, out_dir, LOWER, UPPER)

    npz = np.load(out_dir / "000.npz")
    assert {"rss_gt_norm", "rss_observed_norm", "rss_gt", "rss_observed", "tx_locations"}.issubset(npz.files)
    assert npz["rss_gt_norm"].min() >= 0.0 and npz["rss_gt_norm"].max() <= 1.0


def test_save_stats(tmp_path):
    # stats.npz に正しい値が保存されること
    save_stats(tmp_path, LOWER, UPPER, LOWER_PCT, UPPER_PCT)
    npz = np.load(tmp_path / "stats.npz")
    assert float(npz["rss_lower_dbm"]) == pytest.approx(LOWER)
    assert float(npz["rss_upper_dbm"]) == pytest.approx(UPPER)
