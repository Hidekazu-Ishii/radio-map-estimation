"""OSM通信なし, 純粋関数・データ構造のみ検証."""

import dataclasses

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box as shapely_box

from radio_map_estimation.simulater.osm_buildings import (
    AreaSpec,
    BuildingData,
    _estimate_heights,
    _rasterize_buildings,
    save_building_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spec() -> AreaSpec:
    """100m x 100m, 10m セル, bbox原点 = (0, 0)."""
    return AreaSpec(
        center_lat=35.7,
        center_lon=139.7,
        area_size_m=100.0,
        cell_size_m=10.0,
        crs="EPSG:32654",
        bbox_xmin=0.0,
        bbox_ymin=0.0,
        bbox_xmax=100.0,
        bbox_ymax=100.0,
    )


@pytest.fixture
def gdf() -> gpd.GeoDataFrame:
    """(10,10)-(30,30) の建物 1 棟, 高さ 15m."""
    return gpd.GeoDataFrame(
        {"geometry": [shapely_box(10, 10, 30, 30)], "height_m": [15.0]},
        crs="EPSG:32654",
    )


@pytest.fixture
def bd(gdf, spec) -> BuildingData:
    mask, heights = _rasterize_buildings(gdf, spec.bbox_m, spec.cell_size_m, spec.area_size_m)
    return BuildingData(gdf=gdf, building_mask=mask, building_heights=heights, area_spec=spec)


# ---------------------------------------------------------------------------
# AreaSpec
# ---------------------------------------------------------------------------


def test_grid_size(spec):
    assert spec.grid_size == 10  # 100 / 10


def test_frozen(spec):
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.area_size_m = 999.0


# ---------------------------------------------------------------------------
# _estimate_heights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "col,val,expected_h",
    [
        ("height", "20.0", 20.0),  # height タグ優先
        ("building:levels", "3", 12.0),  # 3階 x 4.0m
    ],
)
def test_estimate_heights(col, val, expected_h):
    gdf = gpd.GeoDataFrame(
        {"geometry": [shapely_box(0, 0, 1, 1)], col: [val]},
        crs="EPSG:32654",
    )
    h = _estimate_heights(gdf, meters_per_level=4.0, default_height_m=8.0, building_type_levels={})
    assert h[0] == pytest.approx(expected_h)


def test_estimate_heights_default():
    # タグなし → default
    gdf = gpd.GeoDataFrame({"geometry": [shapely_box(0, 0, 1, 1)]}, crs="EPSG:32654")
    h = _estimate_heights(gdf, meters_per_level=4.0, default_height_m=8.0, building_type_levels={})
    assert h[0] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# _rasterize_buildings / BuildingData
# ---------------------------------------------------------------------------


def test_rasterize_shape(bd, spec):
    gs = spec.grid_size
    assert bd.building_mask.shape == (gs, gs)
    assert bd.building_heights.shape == (gs, gs)


@pytest.mark.parametrize(
    "r,c,expected",
    [
        (1, 1, True),  # ポリゴン内
        (0, 0, False),  # ポリゴン外
    ],
)
def test_rasterize_cells(bd, r, c, expected):
    assert bd.building_mask[r, c] == expected


def test_heights_positive_where_masked(bd):
    assert (bd.building_heights[bd.building_mask] > 0).all()


# ---------------------------------------------------------------------------
# save_building_data
# ---------------------------------------------------------------------------


def test_save_roundtrip(bd, tmp_path):
    save_building_data(bd, tmp_path)
    npz = np.load(tmp_path / "building_data.npz")
    np.testing.assert_array_equal(npz["building_mask"], bd.building_mask)
    assert float(npz["area_size_m"]) == pytest.approx(bd.area_spec.area_size_m)
