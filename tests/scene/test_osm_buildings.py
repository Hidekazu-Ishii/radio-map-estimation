"""
osm_buildings モジュールのテスト.

方針:
    - fetch_buildings_osm は OSMnx の外部通信を伴うため, ここではテストしない.
      (integration test / E2E test の対象)
    - 内部関数 _estimate_heights, _rasterize_buildings を直接テストする.
"""

import geopandas as gpd
import pytest
from shapely.geometry import box as shapely_box
from src.radio_map_estimation.scene.osm_buildings import (
    _estimate_heights,
    _rasterize_buildings,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gdf(*height_tags: str | None, levels_tags: list[str | None] | None = None) -> gpd.GeoDataFrame:
    """テスト用ダミー GeoDataFrame を生成する."""
    n = len(height_tags)
    data: dict = {
        "geometry": [shapely_box(0, 0, 10, 10)] * n,
        "height": list(height_tags),
    }
    if levels_tags is not None:
        data["building:levels"] = levels_tags
    return gpd.GeoDataFrame(data, crs="EPSG:3857")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def meters_per_level() -> float:
    """階あたりの高さ [m]."""
    return 3.0


@pytest.fixture
def simple_bbox_m() -> tuple[float, float, float, float]:
    """100m x 100m の正方形エリア."""
    return (0.0, 0.0, 100.0, 100.0)


@pytest.fixture
def cell_size_m() -> float:
    return 10.0


@pytest.fixture
def area_size_m() -> float:
    return 100.0


# ---------------------------------------------------------------------------
# _estimate_heights
# ---------------------------------------------------------------------------


class TestEstimateHeights:
    def test_height_tag_used_first(self, meters_per_level):
        """height タグが優先されること."""
        gdf = _make_gdf("20.0", levels_tags=["2"])  # levels → 6m だが height → 20m が勝つ
        heights = _estimate_heights(gdf, meters_per_level)
        assert heights[0] == pytest.approx(20.0)

    def test_levels_tag_fallback(self, meters_per_level):
        """height タグが None のとき building:levels にフォールバックすること."""
        gdf = _make_gdf(None, levels_tags=["4"])  # 4 * 3.0 = 12.0 m
        heights = _estimate_heights(gdf, meters_per_level)
        assert heights[0] == pytest.approx(12.0)

    def test_default_height_is_mean_of_known(self, meters_per_level):
        """height/levels 両方 None のセルにはタグ既知建物の平均が使われること."""
        # 既知: 10.0, 20.0 → mean = 15.0
        gdf = _make_gdf("10.0", "20.0", None)
        heights = _estimate_heights(gdf, meters_per_level)
        assert heights[2] == pytest.approx(15.0)

    def test_raises_when_no_height_info(self, meters_per_level):
        """height / levels タグが 1 件もない場合 ValueError を送出すること."""
        gdf = _make_gdf(None, None)
        with pytest.raises(ValueError, match="height"):
            _estimate_heights(gdf, meters_per_level)

    @pytest.mark.parametrize(
        "height_str,expected",
        [
            ("15.0", 15.0),
            ("0.5", 0.5),
            ("100", 100.0),
        ],
    )
    def test_various_height_strings(self, height_str, expected, meters_per_level):
        """数値文字列が正しくパースされること."""
        gdf = _make_gdf(height_str)
        heights = _estimate_heights(gdf, meters_per_level)
        assert heights[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _rasterize_buildings
# ---------------------------------------------------------------------------


class TestRasterizeBuildings:
    def test_building_at_center_is_rasterized(self, simple_bbox_m, cell_size_m, area_size_m):
        """グリッド中央に建物があるとき, 対応セルが True になること."""
        # エリア中央付近の建物 (40m ~ 60m)
        poly = shapely_box(40, 40, 60, 60)
        gdf = gpd.GeoDataFrame({"geometry": [poly], "height_m": [15.0]}, crs="EPSG:3857")

        mask, _ = _rasterize_buildings(gdf, simple_bbox_m, cell_size_m, area_size_m)

        # 中央セル (row=4..5, col=4..5) が建物になっているはず
        assert mask[4, 4], "中央セルが建物として検出されていない"

    def test_height_assigned_to_rasterized_cell(self, simple_bbox_m, cell_size_m, area_size_m):
        """ラスタライズされたセルに建物高さが正しく代入されること."""
        poly = shapely_box(5, 5, 15, 15)  # col=0, row=0 のセルを中心とする建物
        gdf = gpd.GeoDataFrame({"geometry": [poly], "height_m": [25.0]}, crs="EPSG:3857")

        _, heights = _rasterize_buildings(gdf, simple_bbox_m, cell_size_m, area_size_m)

        assert heights[1, 1] == pytest.approx(25.0)

    def test_overlapping_buildings_take_max_height(self, simple_bbox_m, cell_size_m, area_size_m):
        """同一セルに複数建物が重なる場合, 最大高さが採用されること."""
        poly = shapely_box(5, 5, 15, 15)
        gdf = gpd.GeoDataFrame(
            {"geometry": [poly, poly], "height_m": [10.0, 30.0]},
            crs="EPSG:3857",
        )

        _, heights = _rasterize_buildings(gdf, simple_bbox_m, cell_size_m, area_size_m)

        assert heights[1, 1] == pytest.approx(30.0)

    def test_empty_gdf_returns_zero_arrays(self, simple_bbox_m, cell_size_m, area_size_m):
        """建物がない場合, mask は全 False, heights は全 0 になること."""
        gdf = gpd.GeoDataFrame({"geometry": [], "height_m": []}, crs="EPSG:3857")

        mask, heights = _rasterize_buildings(gdf, simple_bbox_m, cell_size_m, area_size_m)

        assert not mask.any()
        assert (heights == 0.0).all()

    def test_output_shape(self, simple_bbox_m, cell_size_m, area_size_m):
        """出力グリッドの shape が (grid_size, grid_size) になること."""
        gdf = gpd.GeoDataFrame({"geometry": [], "height_m": []}, crs="EPSG:3857")
        mask, heights = _rasterize_buildings(gdf, simple_bbox_m, cell_size_m, area_size_m)

        expected = (int(area_size_m / cell_size_m),) * 2
        assert mask.shape == expected
        assert heights.shape == expected

    @pytest.mark.parametrize(
        "cell_size_m,area_size_m",
        [
            (5.0, 50.0),
            (10.0, 100.0),
            (20.0, 200.0),
        ],
    )
    def test_grid_size_scales_with_cell_size(self, simple_bbox_m, cell_size_m, area_size_m):
        """grid_size = area_size_m / cell_size_m が正しく計算されること."""
        bbox_m = (0.0, 0.0, area_size_m, area_size_m)
        gdf = gpd.GeoDataFrame({"geometry": [], "height_m": []}, crs="EPSG:3857")
        mask, _ = _rasterize_buildings(gdf, bbox_m, cell_size_m, area_size_m)

        expected_size = int(area_size_m / cell_size_m)
        assert mask.shape == (expected_size, expected_size)
