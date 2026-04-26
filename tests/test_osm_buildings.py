"""
tests/test_osm_buildings.py

src/osm_buildings.py のユニットテスト.
OSM通信は行わず, GeoDataFrame を直接構築してロジックを検証する.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box as shapely_box

from radio_map_estimation.generater.osm_buildings import buildings_to_grid

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bbox_m() -> tuple[float, float, float, float]:
    """100m x 100m の正方形エリア."""
    return (0.0, 0.0, 100.0, 100.0)


@pytest.fixture
def simple_gdf(bbox_m) -> gpd.GeoDataFrame:
    """左下 1/4 のセルに 10m の建物が 1 棟だけある GeoDataFrame."""
    xmin, ymin, xmax, ymax = bbox_m
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    poly = shapely_box(xmin, ymin, cx, cy)  # 左下 1/4 エリアを占める建物
    gdf = gpd.GeoDataFrame({"geometry": [poly], "height_m": [10.0]}, crs="EPSG:32654")
    return gdf


@pytest.fixture
def empty_gdf(bbox_m) -> gpd.GeoDataFrame:
    """建物が存在しない空の GeoDataFrame."""
    return gpd.GeoDataFrame({"geometry": [], "height_m": []}, crs="EPSG:32654")


# ---------------------------------------------------------------------------
# buildings_to_grid のテスト
# ---------------------------------------------------------------------------


class TestBuildingsToGrid:
    """buildings_to_grid の出力形状・値を検証する."""

    @pytest.mark.parametrize("grid_size", [4, 8, 16])
    def test_output_shape(self, simple_gdf, bbox_m, grid_size):
        """mask と heights の形状が (grid_size, grid_size) であること."""
        mask, heights = buildings_to_grid(simple_gdf, bbox_m, grid_size, area_size_m=100.0)
        assert mask.shape == (grid_size, grid_size)
        assert heights.shape == (grid_size, grid_size)

    def test_output_dtype(self, simple_gdf, bbox_m):
        """mask が bool, heights が float であること."""
        mask, heights = buildings_to_grid(simple_gdf, bbox_m, grid_size=4, area_size_m=100.0)
        assert mask.dtype == bool
        assert heights.dtype == float

    def test_building_cell_is_true(self, simple_gdf, bbox_m):
        """建物ポリゴンが占める左下セルが True になること."""
        # grid_size=2 → 各セル 50m x 50m → 左下セルは [row=0, col=0]
        mask, _ = buildings_to_grid(simple_gdf, bbox_m, grid_size=2, area_size_m=100.0)
        assert mask[0, 0] is np.bool_(True)

    def test_non_building_cell_is_false(self, simple_gdf, bbox_m):
        """建物が存在しないセルが False になること."""
        mask, _ = buildings_to_grid(simple_gdf, bbox_m, grid_size=2, area_size_m=100.0)
        # 右上セル [row=1, col=1] には建物なし
        assert mask[1, 1] is np.bool_(False)

    def test_height_matches_polygon(self, simple_gdf, bbox_m):
        """建物セルの高さが GeoDataFrame の height_m と一致すること."""
        _, heights = buildings_to_grid(simple_gdf, bbox_m, grid_size=2, area_size_m=100.0)
        assert heights[0, 0] == pytest.approx(10.0)

    def test_no_building_height_is_zero(self, simple_gdf, bbox_m):
        """建物なしセルの高さが 0.0 であること."""
        _, heights = buildings_to_grid(simple_gdf, bbox_m, grid_size=2, area_size_m=100.0)
        assert heights[1, 1] == pytest.approx(0.0)

    def test_empty_gdf_returns_all_false(self, empty_gdf, bbox_m):
        """建物が存在しない場合, mask がすべて False であること."""
        mask, heights = buildings_to_grid(empty_gdf, bbox_m, grid_size=4, area_size_m=100.0)
        assert not mask.any()
        assert (heights == 0.0).all()

    def test_max_height_is_adopted(self, bbox_m):
        """複数の建物が重なるセルでは最大高さが採用されること."""
        # 同じセルに高さ 5m と 20m の建物を重ねる
        poly_a = shapely_box(0, 0, 50, 50)
        poly_b = shapely_box(0, 0, 30, 30)
        gdf = gpd.GeoDataFrame(
            {"geometry": [poly_a, poly_b], "height_m": [5.0, 20.0]},
            crs="EPSG:32654",
        )
        _, heights = buildings_to_grid(gdf, bbox_m, grid_size=2, area_size_m=100.0)
        assert heights[0, 0] == pytest.approx(20.0)

    @pytest.mark.parametrize(
        "grid_size, area_size_m",
        [(4, 100.0), (8, 200.0), (16, 500.0)],
    )
    def test_mask_sum_positive(self, simple_gdf, grid_size, area_size_m):
        """建物がある場合, mask の True セル数が 1 以上であること."""
        # bbox を area_size_m に合わせてスケール
        bbox = (0.0, 0.0, area_size_m, area_size_m)
        mask, _ = buildings_to_grid(simple_gdf, bbox, grid_size, area_size_m)
        assert mask.sum() >= 1
