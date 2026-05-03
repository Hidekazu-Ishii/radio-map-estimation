"""Sionna/mitsuba 不要, 純粋関数のみ検証."""

import geopandas as gpd
import numpy as np
import pytest
import trimesh
from numpy.random import default_rng
from shapely.geometry import box as shapely_box

from radio_map_estimation.simulater.osm_buildings import AreaSpec, BuildingData
from radio_map_estimation.simulater.radiomap import (
    build_ground_mesh,
    compute_best_server_map,
    extrude_polygon_to_mesh,
    place_tx_ppp,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spec() -> AreaSpec:
    """500m x 500m, 10m セル."""
    return AreaSpec(
        center_lat=35.7,
        center_lon=139.7,
        area_size_m=500.0,
        cell_size_m=10.0,
        crs="EPSG:32654",
        bbox_xmin=0.0,
        bbox_ymin=0.0,
        bbox_xmax=500.0,
        bbox_ymax=500.0,
    )


@pytest.fixture
def building_data(spec) -> BuildingData:
    """(10,10)-(30,30) の建物 1 棟, 高さ 15m."""
    gdf = gpd.GeoDataFrame(
        {"geometry": [shapely_box(10, 10, 30, 30)], "height_m": [15.0]},
        crs="EPSG:32654",
    )
    gs = spec.grid_size
    mask = np.zeros((gs, gs), dtype=bool)
    heights = np.zeros((gs, gs), dtype=float)
    return BuildingData(gdf=gdf, building_mask=mask, building_heights=heights, area_spec=spec)


# ---------------------------------------------------------------------------
# extrude_polygon_to_mesh
# ---------------------------------------------------------------------------


def test_extrude_is_trimesh():
    poly = shapely_box(0, 0, 10, 10)
    mesh = extrude_polygon_to_mesh(poly, height=5.0)
    assert isinstance(mesh, trimesh.Trimesh)


def test_extrude_z_range():
    # z は [0, height] に収まる
    mesh = extrude_polygon_to_mesh(shapely_box(0, 0, 10, 10), height=5.0)
    assert mesh.vertices[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    assert mesh.vertices[:, 2].max() == pytest.approx(5.0, abs=1e-6)


# ---------------------------------------------------------------------------
# build_ground_mesh
# ---------------------------------------------------------------------------


def test_ground_mesh_z_zero(building_data):
    mesh = build_ground_mesh(building_data)
    assert (mesh.vertices[:, 2] == 0.0).all()


def test_ground_mesh_xy_extent(building_data):
    mesh = build_ground_mesh(building_data)
    s = building_data.area_spec.area_size_m
    assert mesh.vertices[:, 0].max() == pytest.approx(s)
    assert mesh.vertices[:, 1].max() == pytest.approx(s)


# ---------------------------------------------------------------------------
# place_tx_ppp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("intensity", [1e-4, 1e-3])
def test_place_tx_ppp_at_least_one(intensity):
    rng = default_rng(0)
    tx = place_tx_ppp(rng, area_size_m=500.0, intensity=intensity)
    assert tx.shape[1] == 2
    assert len(tx) >= 1


def test_place_tx_ppp_in_area():
    rng = default_rng(0)
    tx = place_tx_ppp(rng, area_size_m=500.0, intensity=1e-3)
    assert (tx >= 0.0).all() and (tx <= 500.0).all()


# ---------------------------------------------------------------------------
# compute_best_server_map
# ---------------------------------------------------------------------------


def test_best_server_shape():
    maps = [np.full((10, 10), float(v)) for v in [-80, -70, -90]]
    best = compute_best_server_map(maps)
    assert best.shape == (10, 10)


def test_best_server_values():
    # 各セルで最大値が選ばれる
    maps = [np.full((10, 10), float(v)) for v in [-80, -70, -90]]
    best = compute_best_server_map(maps)
    np.testing.assert_allclose(best, -70.0)


def test_best_server_nan_ignored():
    # nan は無視して有効値が選ばれる
    m1 = np.full((5, 5), np.nan)
    m2 = np.full((5, 5), -80.0)
    best = compute_best_server_map([m1, m2])
    np.testing.assert_allclose(best, -80.0)
