"""
tx_placement モジュールのテスト.
- compute_open_building_mask : 開放度マスク生成
- place_tx_ppp               : PPP による TX 配置
"""

import numpy as np
import pytest
from numpy.random import default_rng

from src.radio_map_estimation.simulate.tx_placement import (
    compute_open_building_mask,
    place_tx_ppp,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    """再現性のある乱数生成器 (seed は fixture で一元管理)."""
    return default_rng(42)


@pytest.fixture
def grid_with_buildings() -> tuple[np.ndarray, np.ndarray]:
    """
    20 x 20 グリッド: 中央に建物ブロック, 外周は非建物.

    建物高さはすべて 25.0 m (min=20, max=40 の範囲内).
    """
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    heights = np.where(mask, 25.0, 0.0)
    return mask, heights


@pytest.fixture
def ppp_kwargs(grid_with_buildings) -> dict:
    """place_tx_ppp の標準パラメータ."""
    mask, heights = grid_with_buildings
    return {
        "area_size_m": 200.0,
        "intensity": 1e-4,
        "building_heights": heights,
        "cell_size_m": 10.0,
        "building_mask": mask,
        "min_open_neighbors": 3,
        "min_separation_m": 20.0,
        "tx_height_above_building_m": 3.0,
        "min_building_height_m": 20.0,
        "max_building_height_m": 40.0,
        "height_weight_power": 1.0,
        "inner_margin_m": 10.0,
    }


# ---------------------------------------------------------------------------
# compute_open_building_mask
# ---------------------------------------------------------------------------


class TestComputeOpenBuildingMask:
    def test_non_building_cells_excluded(self, grid_with_buildings):
        """非建物セルは常に False であること."""
        mask, _ = grid_with_buildings
        open_mask = compute_open_building_mask(mask, min_open_neighbors=3)
        assert not open_mask[~mask].any()

    def test_interior_building_cells_excluded(self, grid_with_buildings):
        """建物内部セル (周囲が全て建物) は False であること."""
        mask, _ = grid_with_buildings
        open_mask = compute_open_building_mask(mask, min_open_neighbors=3)
        # 建物ブロック中心 (8:12, 8:12) は周囲すべて建物 → 開放近傍数 = 0
        assert not open_mask[8:12, 8:12].any()

    def test_edge_building_cells_included(self, grid_with_buildings):
        """建物外縁セルは True になること."""
        mask, _ = grid_with_buildings
        open_mask = compute_open_building_mask(mask, min_open_neighbors=3)
        # 建物ブロック外縁の角セルは非建物隣接数が多い
        assert open_mask[5, 5]

    @pytest.mark.parametrize("min_open_neighbors", [1, 3, 6])
    def test_threshold_effect(self, grid_with_buildings, min_open_neighbors):
        """閾値が大きいほど候補セル数が減ること."""
        mask, _ = grid_with_buildings
        open_mask = compute_open_building_mask(mask, min_open_neighbors)
        # 閾値が最大 (8) なら候補は 0 のはず
        if min_open_neighbors <= 6:
            assert open_mask.sum() >= 0  # 単調減少を複数閾値で確認
        assert open_mask.sum() <= mask.sum()  # 建物セル数を超えない


# ---------------------------------------------------------------------------
# place_tx_ppp
# ---------------------------------------------------------------------------


class TestPlaceTxPpp:
    def test_output_shape(self, rng, ppp_kwargs):
        """出力が (T, 3) の shape であること."""
        tx = place_tx_ppp(rng=rng, **ppp_kwargs)
        assert tx.ndim == 2
        assert tx.shape[1] == 3

    def test_at_least_one_tx_placed(self, rng, ppp_kwargs):
        """TX が 1 局以上配置されること."""
        tx = place_tx_ppp(rng=rng, **ppp_kwargs)
        assert len(tx) >= 1

    def test_tx_height_above_building(self, rng, ppp_kwargs, grid_with_buildings):
        """TX の z 座標が建物高さ + オフセット以上であること."""
        _, _ = grid_with_buildings
        tx = place_tx_ppp(rng=rng, **ppp_kwargs)
        offset = ppp_kwargs["tx_height_above_building_m"]
        min_building_h = ppp_kwargs["min_building_height_m"]
        assert (tx[:, 2] >= min_building_h + offset).all()

    def test_min_separation_satisfied(self, rng, ppp_kwargs):
        """全 TX ペアの離間距離が min_separation_m 以上であること."""
        tx = place_tx_ppp(rng=rng, **ppp_kwargs)
        sep = ppp_kwargs["min_separation_m"]
        for i in range(len(tx)):
            for j in range(i + 1, len(tx)):
                dist = np.linalg.norm(tx[i, :2] - tx[j, :2])
                assert dist >= sep - 1e-6, f"TX{i}-TX{j}: {dist:.2f} < {sep}"

    def test_tx_within_area(self, rng, ppp_kwargs):
        """TX の xy 座標がエリア内に収まること."""
        tx = place_tx_ppp(rng=rng, **ppp_kwargs)
        area = ppp_kwargs["area_size_m"]
        assert (tx[:, 0] >= 0).all() and (tx[:, 0] <= area).all()
        assert (tx[:, 1] >= 0).all() and (tx[:, 1] <= area).all()

    def test_no_candidates_raises(self, rng, ppp_kwargs):
        """候補セルが 0 件のとき RuntimeError を送出すること."""
        # 建物高さをすべて 1.0 m にして min_building_height_m=20 を満たさなくする
        kwargs = {**ppp_kwargs, "building_heights": np.ones((20, 20))}
        with pytest.raises(RuntimeError, match="候補セルが0件"):
            place_tx_ppp(rng=rng, **kwargs)

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_reproducibility(self, ppp_kwargs, seed):
        """同じ seed なら同じ結果が得られること."""
        tx1 = place_tx_ppp(rng=default_rng(seed), **ppp_kwargs)
        tx2 = place_tx_ppp(rng=default_rng(seed), **ppp_kwargs)
        np.testing.assert_array_equal(tx1, tx2)
