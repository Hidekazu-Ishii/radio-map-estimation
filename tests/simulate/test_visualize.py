"""
visualize モジュールのテスト.

方針:
    全関数が plt.savefig で終わる設計のため,
    「指定パスにファイルが生成されること」のみを確認する (smoke test).
"""

import geopandas as gpd
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # GUI なし環境で動作させる

from src.radio_map_estimation.scene.schema import AreaSpec, BuildingData
from src.radio_map_estimation.simulate.visualize import (
    plot_building_data,
    plot_radio_map,
    plot_tx_association,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def building_data() -> BuildingData:
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    heights = np.where(mask, 20.0, 0.0)
    area_spec = AreaSpec(
        center_lat=35.0,
        center_lon=139.0,
        area_size_m=100.0,
        cell_size_m=10.0,
        crs="EPSG:3857",
        bbox_xmin=0.0,
        bbox_ymin=0.0,
        bbox_xmax=100.0,
        bbox_ymax=100.0,
    )
    gdf = gpd.GeoDataFrame({"geometry": [], "height_m": []}, crs="EPSG:3857")
    return BuildingData(gdf=gdf, building_mask=mask, building_heights=heights, area_spec=area_spec)


@pytest.fixture
def rss_dbm() -> np.ndarray:
    """(10, 10) の RSS マップ [dBm]."""
    return np.full((10, 10), -80.0)


@pytest.fixture
def tx_positions() -> np.ndarray:
    """3 TX の (x, y, z) 座標."""
    return np.array([[20.0, 30.0, 28.0], [50.0, 50.0, 25.0], [80.0, 70.0, 30.0]])


# ---------------------------------------------------------------------------
# smoke tests
# ---------------------------------------------------------------------------


def test_plot_building_data(building_data, tmp_path):
    save_path = tmp_path / "building.png"
    plot_building_data(building_data, save_path)
    assert save_path.exists()


def test_plot_radio_map(building_data, rss_dbm, tx_positions, tmp_path):
    save_path = tmp_path / "radio_map.png"
    plot_radio_map(
        rss_dbm=rss_dbm,
        building_mask=building_data.building_mask,
        area_size_m=100.0,
        tx_positions=tx_positions,
        frequency_hz=2.4e9,
        save_path=save_path,
    )
    assert save_path.exists()


def test_plot_tx_association(building_data, tx_positions, tmp_path):
    # (num_tx, H, W) の RSS マップ
    rss_per_tx = np.random.default_rng(0).uniform(-100, -60, size=(3, 10, 10))
    save_path = tmp_path / "association.png"
    plot_tx_association(
        rss_dbm_per_tx=rss_per_tx,
        building_mask=building_data.building_mask,
        area_size_m=100.0,
        tx_positions=tx_positions,
        save_path=save_path,
    )
    assert save_path.exists()


@pytest.mark.parametrize(
    "func_name,kwargs_extra",
    [
        ("plot_building_data", {}),
    ],
)
def test_returns_none(building_data, tmp_path, func_name, kwargs_extra):
    """可視化関数は None を返すこと."""
    from src.radio_map_estimation.simulate import visualize as vm

    func = getattr(vm, func_name)
    result = func(building_data, tmp_path / "out.png", **kwargs_extra)
    assert result is None
