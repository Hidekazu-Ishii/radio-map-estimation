"""
visualize_sionna モジュールのテスト.

方針:
    - show_building_data  : matplotlib のみ依存 → 直接テスト
    - show_radio_map 系   : Sionna 依存 → MagicMock でスタブ化
    - save_figure         : ファイル保存の確認
"""

from unittest.mock import MagicMock

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")  # GUI なし環境で Figure を生成する

from src.radio_map_estimation.scene.schema import AreaSpec, BuildingData
from src.radio_map_estimation.simulate.visualize_sionna import (
    save_figure,
    show_building_data,
    show_radio_map,
    show_radio_map_cdf,
    show_tx_association,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def building_data() -> BuildingData:
    """最小の BuildingData."""
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
def mock_radio_map() -> MagicMock:
    """PlanarRadioMap のスタブ. show/cdf/show_association が Figure を返す."""
    rm = MagicMock()
    rm.show.return_value = Figure()
    rm.cdf.return_value = (Figure(),)
    rm.show_association.return_value = Figure()
    return rm


# ---------------------------------------------------------------------------
# show_building_data
# ---------------------------------------------------------------------------


def test_show_building_data_returns_figure(building_data):
    """Figure が返ること."""
    fig = show_building_data(building_data)
    assert isinstance(fig, Figure)


def test_show_building_data_has_two_panels(building_data):
    """2 パネル (axes) で構成されること."""
    fig = show_building_data(building_data)
    assert len(fig.axes) == 3  # imshow 2 + colorbar 1


# ---------------------------------------------------------------------------
# show_radio_map / show_radio_map_cdf / show_tx_association
# ---------------------------------------------------------------------------


def test_show_radio_map_returns_figure(mock_radio_map):
    fig = show_radio_map(mock_radio_map)
    assert isinstance(fig, Figure)


def test_show_radio_map_cdf_returns_figure(mock_radio_map):
    fig = show_radio_map_cdf(mock_radio_map)
    assert isinstance(fig, Figure)


def test_show_tx_association_returns_figure(mock_radio_map):
    fig = show_tx_association(mock_radio_map)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------


def test_save_figure_creates_file(building_data, tmp_path):
    """指定パスにファイルが生成されること."""
    fig = show_building_data(building_data)
    save_path = tmp_path / "test.png"
    save_figure(fig, save_path)
    assert save_path.exists()


@pytest.mark.parametrize("suffix", [".png", ".pdf"])
def test_save_figure_formats(building_data, tmp_path, suffix):
    """png / pdf 形式で保存できること."""
    fig = show_building_data(building_data)
    save_path = tmp_path / f"out{suffix}"
    save_figure(fig, save_path)
    assert save_path.exists()
