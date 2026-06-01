"""
tests/test_building_completion.py

building_completion モジュールのテスト.
- _fill_enclosed_cells      : 内部ロジック
- apply_building_completion : BuildingData ラッパー
"""

import geopandas as gpd
import numpy as np
import pytest
from src.radio_map_estimation.scene.building_completion import (
    _fill_enclosed_cells,
    apply_building_completion,
)

from src.radio_map_estimation.scene.schema import AreaSpec, BuildingData

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grid_with_hole() -> tuple[np.ndarray, np.ndarray]:
    """
    11x11 グリッド: 中央に厚さ1の建物リング, その内外に余白あり.

    . . . . . . . . . . .
    . . . . . . . . . . .
    . . B B B B B B B . .
    . . B . . . . . B . .
    . . B . . . . . B . .
    . . B . . . . . B . .
    . . B . . . . . B . .
    . . B . . . . . B . .
    . . B B B B B B B . .
    . . . . . . . . . . .
    . . . . . . . . . . .

    Note:
        外周 2 セルを余白にすることで closing_size=3 のカーネルが
        グリッド端に届かず, binary_fill_holes が正しく内部を検出できる.
    """
    mask = np.zeros((11, 11), dtype=bool)
    mask[2, 2:9] = mask[8, 2:9] = True  # 上下の壁
    mask[2:9, 2] = mask[2:9, 8] = True  # 左右の壁
    heights = np.where(mask, 10.0, 0.0)
    return mask, heights


@pytest.fixture
def grid_no_hole() -> tuple[np.ndarray, np.ndarray]:
    """囲まれた領域が存在しない L 字型グリッド."""
    mask = np.zeros((11, 11), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    heights = np.where(mask, 8.0, 0.0)
    return mask, heights


@pytest.fixture
def building_data_with_hole(grid_with_hole) -> BuildingData:
    mask, heights = grid_with_hole
    area_spec = AreaSpec(
        center_lat=35.0,
        center_lon=139.0,
        area_size_m=110.0,
        cell_size_m=10.0,
        crs="EPSG:3857",
        bbox_xmin=0.0,
        bbox_ymin=0.0,
        bbox_xmax=110.0,
        bbox_ymax=110.0,
    )
    gdf = gpd.GeoDataFrame({"geometry": [], "height_m": []}, crs="EPSG:3857")
    return BuildingData(gdf=gdf, building_mask=mask, building_heights=heights, area_spec=area_spec)


# ---------------------------------------------------------------------------
# _fill_enclosed_cells
# ---------------------------------------------------------------------------


class TestFillEnclosedCells:
    def test_enclosed_cells_become_building(self, grid_with_hole):
        """囲まれた内部セルが建物として補完されること."""
        mask, heights = grid_with_hole
        filled_mask, _ = _fill_enclosed_cells(mask, heights)
        # 建物リング内側 (row=3..7, col=3..7) がすべて True になるはず
        assert filled_mask[3:8, 3:8].all()

    def test_original_mask_not_mutated(self, grid_with_hole):
        """入力マスクが破壊的変更されないこと."""
        mask, heights = grid_with_hole
        original = mask.copy()
        _fill_enclosed_cells(mask, heights)
        np.testing.assert_array_equal(mask, original)

    def test_filled_height_equals_nearest_building(self, grid_with_hole):
        """補完セルの高さが最近傍建物高さ (10.0) になること."""
        mask, heights = grid_with_hole
        _, filled_heights = _fill_enclosed_cells(mask, heights)
        assert (filled_heights[3:8, 3:8] == 10.0).all()

    def test_no_hole_returns_original(self, grid_no_hole):
        """囲まれた領域がない場合, 入力と同一の配列が返ること."""
        mask, heights = grid_no_hole
        filled_mask, filled_heights = _fill_enclosed_cells(mask, heights)
        np.testing.assert_array_equal(filled_mask, mask)
        np.testing.assert_array_equal(filled_heights, heights)

    def test_filled_mask_is_superset(self, grid_with_hole):
        """補完後マスクは元マスクの上位集合であること."""
        mask, heights = grid_with_hole
        filled_mask, _ = _fill_enclosed_cells(mask, heights)
        assert filled_mask[mask].all()

    @pytest.mark.parametrize("closing_size", [1, 3, 5])
    def test_various_closing_sizes(self, grid_with_hole, closing_size):
        """closing_size を変えても補完後マスクが元マスクの上位集合であること."""
        mask, heights = grid_with_hole
        filled_mask, _ = _fill_enclosed_cells(mask, heights, closing_size=closing_size)
        assert filled_mask[mask].all()


# ---------------------------------------------------------------------------
# apply_building_completion
# ---------------------------------------------------------------------------


class TestApplyBuildingCompletion:
    def test_returns_new_object(self, building_data_with_hole):
        """replace により新しい BuildingData が返ること (immutability)."""
        result = apply_building_completion(building_data_with_hole)
        assert result is not building_data_with_hole

    def test_building_count_increases(self, building_data_with_hole):
        """補完後の建物セル数が元より多いこと."""
        result = apply_building_completion(building_data_with_hole)
        assert result.building_mask.sum() > building_data_with_hole.building_mask.sum()

    def test_area_spec_preserved(self, building_data_with_hole):
        """area_spec が変更されないこと."""
        result = apply_building_completion(building_data_with_hole)
        assert result.area_spec == building_data_with_hole.area_spec
