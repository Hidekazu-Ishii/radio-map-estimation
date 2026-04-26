"""
tests/test_radiomap.py
"""

import numpy as np
import pytest
from numpy.random import default_rng

from radio_map_estimation.generater.radiomap import (
    compute_los_flag,
    generate_rss_map,
    generate_shadowing_field,
    path_loss_uma_db,
    place_bs_ppp,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return default_rng(0)


@pytest.fixture
def area_size() -> float:
    return 500.0


@pytest.fixture
def grid_size() -> int:
    return 16


@pytest.fixture
def sparse_heights(grid_size) -> np.ndarray:
    """建物が数棟だけ存在するグリッド (確率モデルまで走る最小構成)."""
    h = np.zeros((grid_size, grid_size))
    h[2, 3] = 15.0
    h[8, 10] = 20.0
    return h


@pytest.fixture
def grid_coords(grid_size, area_size) -> np.ndarray:
    dx = area_size / grid_size
    xs = np.arange(grid_size) * dx + dx / 2
    ys = np.arange(grid_size) * dx + dx / 2
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (P, 2)


@pytest.fixture
def bs_xy(area_size) -> np.ndarray:
    return np.array([area_size / 2, area_size / 2])  # エリア中央


# ---------------------------------------------------------------------------
# place_bs_ppp
# ---------------------------------------------------------------------------


def test_place_bs_ppp_shape_and_range(rng, area_size):
    bs = place_bs_ppp(rng, area_size, intensity=1e-4)
    assert bs.shape[1] == 2
    assert (bs >= 0).all() and (bs <= area_size).all()


def test_place_bs_ppp_at_least_one(rng, area_size):
    # intensity=0 でも最低 1 局
    assert len(place_bs_ppp(rng, area_size, intensity=0.0)) >= 1


# ---------------------------------------------------------------------------
# path_loss_uma_db
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [10.0, 100.0, 500.0])
def test_path_loss_los_lt_nlos(d):
    # LOS < NLOS であること
    d_arr = np.array([d])
    pl_los = path_loss_uma_db(d_arr, 3.5e9, np.array([True]))
    pl_nlos = path_loss_uma_db(d_arr, 3.5e9, np.array([False]))
    assert pl_los[0] < pl_nlos[0]


def test_path_loss_monotone_distance():
    # 距離増加でパスロスが単調増加すること
    ds = np.array([10.0, 50.0, 100.0, 500.0])
    pl = path_loss_uma_db(ds, 3.5e9, np.ones(len(ds), dtype=bool))
    assert (np.diff(pl) > 0).all()


# ---------------------------------------------------------------------------
# generate_shadowing_field
# ---------------------------------------------------------------------------


def test_shadowing_std(area_size):
    # 標準偏差が sigma_db に収束すること (大グリッドで検証)
    shadow = generate_shadowing_field(default_rng(0), 64, area_size, sigma_db=8.0, corr_distance=50.0)
    assert shadow.shape == (64**2,)
    assert abs(shadow.std() - 8.0) / 8.0 < 0.05


# ---------------------------------------------------------------------------
# compute_los_flag
# ---------------------------------------------------------------------------


def test_los_flag_with_buildings(rng, bs_xy, grid_coords, sparse_heights, area_size):
    # 建物ありエリア → 幾何判定 + 確率モデルの両方が走ること
    is_los = compute_los_flag(rng, bs_xy, 25.0, grid_coords, 1.5, sparse_heights, area_size, n_sample=40)
    assert is_los.dtype == bool
    assert 0 < is_los.mean() < 1.0  # LOS/NLOS が混在すること


def test_los_flag_blocked_by_wall(rng, area_size):
    # 200m の壁で遮蔽された UE は必ず NLOS になること (幾何判定の検証)
    grid_size = 10
    dx = area_size / grid_size
    heights = np.zeros((grid_size, grid_size))
    heights[:, grid_size // 2] = 200.0  # 中央列に壁

    bs_local = np.array([dx / 2, area_size / 2])
    ue_far = np.array([[area_size - dx / 2, area_size / 2]])

    is_los = compute_los_flag(rng, bs_local, 10.0, ue_far, 1.5, heights, area_size, n_sample=100)
    assert not is_los[0]


# ---------------------------------------------------------------------------
# generate_rss_map
# ---------------------------------------------------------------------------


def test_rss_decreases_with_distance(rng, sparse_heights, area_size):
    # shadowing=0, BS を原点に配置 → 距離が増えるほど RSS が低下すること
    bs_local = np.array([0.0, 0.0])
    ds = np.linspace(10, area_size, 10)
    coords = np.stack([ds, np.zeros_like(ds)], axis=-1)
    rss, _ = generate_rss_map(
        rng,
        bs_local,
        25.0,
        coords,
        1.5,
        shadowing=np.zeros(len(ds)),
        building_heights=sparse_heights,
        area_size=area_size,
        tx_power_dbm=46.0,
        frequency_hz=3.5e9,
        n_sample=20,
    )
    assert rss[:3].mean() > rss[-3:].mean()
