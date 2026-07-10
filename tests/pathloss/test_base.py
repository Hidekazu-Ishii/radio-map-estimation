"""base.py (FitResult, PathLossModel の共通ユーティリティ) のテスト"""

import numpy as np
import pytest

from radio_map_estimation.loader.dataset import GridInfo
from radio_map_estimation.pathloss.base import FitResult, PathLossModel

# ------------------------------------------------------------------
# fixtures: 実験条件を明示
# ------------------------------------------------------------------


@pytest.fixture
def fit_result() -> FitResult:
    """表示系メソッドのテスト用サンプル"""
    return FitResult(
        model_name="ci",
        params={"n": 2.5, "path_order": [1, 2, 3]},
        norm_stats={"pl_min": 60.0, "pl_max": 140.0},
        n_samples=100,
        rmse_db=3.14159,
    )


@pytest.fixture
def grid_info() -> GridInfo:
    """10x10 の bldg_mask、中央 3x3 のみ建物ありのグリッド"""
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:7, 4:7] = True
    return GridInfo(
        bldg_mask=mask,
        bldg_cell_size_m=1.0,
        cell_size_m=1.0,
        area_size_m=10.0,
        margin_m=0.0,
    )


# ------------------------------------------------------------------
# FitResult
# ------------------------------------------------------------------


def test_formatted_params_float_and_list(fit_result: FitResult) -> None:
    # float は %.4g、list[int] は "[a, b, ...]" 形式になること
    formatted = fit_result.formatted_params()
    assert formatted["n"] == "2.5"
    assert formatted["path_order"] == "[1, 2, 3]"


def test_str_contains_model_name_and_rmse(fit_result: FitResult) -> None:
    text = str(fit_result)
    assert "model=ci" in text
    assert "rmse=3.142dB" in text


# ------------------------------------------------------------------
# compute_3d_distance
# ------------------------------------------------------------------


def test_compute_3d_distance_known_value() -> None:
    # 3-4-5の直角三角形で d=5 になることを確認
    coords = np.array([[0.0, 0.0]])
    tx_coords = np.array([[3.0, 4.0, 0.0]])
    rx_height_m = np.array([[0.0]])
    d = PathLossModel.compute_3d_distance(coords, tx_coords, rx_height_m)
    assert d.shape == (1, 1)
    assert d[0, 0] == pytest.approx(5.0)


def test_compute_3d_distance_clipped_near_zero() -> None:
    # TXとRXがほぼ同じ位置でも d は 1e-3 未満にならないこと
    coords = np.array([[0.0, 0.0]])
    tx_coords = np.array([[0.0, 0.0, 0.0]])
    rx_height_m = np.array([[0.0]])
    d = PathLossModel.compute_3d_distance(coords, tx_coords, rx_height_m)
    assert d[0, 0] == pytest.approx(1e-3)


# ------------------------------------------------------------------
# compute_azimuth
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dx", "dy", "expected_rad"),
    [
        (1.0, 0.0, 0.0),  # 真東
        (0.0, 1.0, np.pi / 2),  # 真北
        (-1.0, 0.0, np.pi),  # 真西
        (0.0, -1.0, -np.pi / 2),  # 真南
    ],
)
def test_compute_azimuth_cardinal_directions(dx: float, dy: float, expected_rad: float) -> None:
    tx_coords = np.array([[0.0, 0.0, 0.0]])
    coords = np.array([[dx, dy]])
    azimuth = PathLossModel.compute_azimuth(coords, tx_coords)
    assert azimuth[0, 0] == pytest.approx(expected_rad)


# ------------------------------------------------------------------
# _bresenham_line
# ------------------------------------------------------------------


def test_bresenham_line_includes_both_endpoints() -> None:
    rows, cols = PathLossModel._bresenham_line(0, 0, 3, 3)
    assert (rows[0], cols[0]) == (0, 0)
    assert (rows[-1], cols[-1]) == (3, 3)


def test_bresenham_line_same_point() -> None:
    # 始点=終点でも1点だけ返ること
    rows, cols = PathLossModel._bresenham_line(2, 2, 2, 2)
    assert rows.shape == (1,)
    assert cols.shape == (1,)


# ------------------------------------------------------------------
# compute_ray_crossing_count
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rx_xy", "tx_xy", "expected_min_crossing"),
    [
        ((0.5, 5.5), (9.5, 5.5), 1),  # 中央の建物ブロックを1回貫通
        ((0.5, 0.5), (0.5, 1.5), 0),  # 建物なし領域のみ移動 → 0回
    ],
)
def test_compute_ray_crossing_count(
    grid_info: GridInfo, rx_xy: tuple[float, float], tx_xy: tuple[float, float], expected_min_crossing: int
) -> None:
    coords = np.array([rx_xy])
    tx_coords = np.array([[tx_xy[0], tx_xy[1], 0.0]])
    crossing = PathLossModel.compute_ray_crossing_count(coords, tx_coords, grid_info)
    assert crossing.shape == (1, 1)
    assert crossing[0, 0] >= expected_min_crossing


# ------------------------------------------------------------------
# compute_bldg_count_in_fresnel_ellipse
# ------------------------------------------------------------------


@pytest.mark.parametrize("fresnel_zone_order", [1, 2, 3])
def test_fresnel_ellipse_bldg_count_nonnegative(grid_info: GridInfo, fresnel_zone_order: int) -> None:
    # 建物を挟む配置ではゾーン次数によらず count >= 0 (次数が大きいほど広がるので単調非減少)
    coords = np.array([[1.0, 5.0]])
    tx_coords = np.array([[9.0, 5.0, 0.0]])
    freq_hz = np.array([[2.4e9]])
    count = PathLossModel.compute_bldg_count_in_fresnel_ellipse(
        coords, tx_coords, grid_info, freq_hz, fresnel_zone_order=fresnel_zone_order
    )
    assert count.shape == (1, 1)
    assert count[0, 0] >= 0.0


def test_fresnel_ellipse_bldg_count_zero_without_buildings() -> None:
    # 建物が一切ないマスクでは常に count=0
    empty_mask = np.zeros((10, 10), dtype=bool)
    empty_grid = GridInfo(
        bldg_mask=empty_mask, bldg_cell_size_m=1.0, cell_size_m=1.0, area_size_m=10.0, margin_m=0.0
    )
    coords = np.array([[1.0, 5.0]])
    tx_coords = np.array([[9.0, 5.0, 0.0]])
    freq_hz = np.array([[2.4e9]])
    count = PathLossModel.compute_bldg_count_in_fresnel_ellipse(coords, tx_coords, empty_grid, freq_hz)
    assert count[0, 0] == 0.0
