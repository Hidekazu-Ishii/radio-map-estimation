"""
best_server モジュールのテスト.
- add_observation_noise : RSS マップへのノイズ付加
"""

import numpy as np
import pytest
from numpy.random import default_rng

from src.radio_map_estimation.simulate.observation_noise import add_observation_noise


@pytest.fixture
def rss_map() -> np.ndarray:
    """3 x 3 の定数 RSS マップ [dBm]."""
    return np.full((3, 3), -70.0)


def test_output_shape(rss_map):
    """出力 shape が入力と同じであること."""
    out = add_observation_noise(rss_map, noise_std_db=5.0, rng=default_rng(0))
    assert out.shape == rss_map.shape


def test_noise_is_added(rss_map):
    """ノイズが付加されていること (入力と完全一致しない)."""
    out = add_observation_noise(rss_map, noise_std_db=5.0, rng=default_rng(0))
    assert not np.allclose(out, rss_map)


def test_reproducibility(rss_map):
    """同じ seed なら同じ出力になること."""
    out1 = add_observation_noise(rss_map, noise_std_db=5.0, rng=default_rng(42))
    out2 = add_observation_noise(rss_map, noise_std_db=5.0, rng=default_rng(42))
    np.testing.assert_array_equal(out1, out2)


def test_zero_noise_returns_original(rss_map):
    """noise_std_db=0 なら入力と同一の値が返ること."""
    out = add_observation_noise(rss_map, noise_std_db=0.0, rng=default_rng(0))
    np.testing.assert_array_equal(out, rss_map)


@pytest.mark.parametrize("noise_std_db", [1.0, 5.0, 10.0])
def test_noise_std_approximately_correct(noise_std_db):
    """付加ノイズの標準偏差がおおよそ noise_std_db に近いこと."""
    rss = np.zeros((1000, 1000))
    out = add_observation_noise(rss, noise_std_db=noise_std_db, rng=default_rng(0))
    assert np.std(out - rss) == pytest.approx(noise_std_db, rel=0.05)
