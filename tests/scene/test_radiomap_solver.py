"""
radiomap_solver モジュールのテスト.

方針:
    - run_radio_map  : Sionna/Mitsuba の外部依存を sys.modules でスタブ化し,
                       関数が呼び出せることのみ確認する (smoke test).
    - radio_map_to_rss_dbm : 純粋な numpy 計算のため直接テストする.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Sionna / Mitsuba スタブ (モジュールレベルで差し込む)
# ---------------------------------------------------------------------------


def _install_stubs() -> None:
    """sionna / mitsuba を sys.modules にスタブとして登録する."""

    # --- mitsuba ---
    mi = ModuleType("mitsuba")
    mi.Point3f = MagicMock(side_effect=lambda x: x)  # type: ignore[attr-defined]
    mi.Point2f = MagicMock(side_effect=lambda x: x)  # type: ignore[attr-defined]
    sys.modules.setdefault("mitsuba", mi)

    # --- sionna 階層 ---
    for name in ("sionna", "sionna.rt"):
        sys.modules.setdefault(name, ModuleType(name))

    rt = sys.modules["sionna.rt"]

    # load_scene が返す scene オブジェクト
    mock_scene = MagicMock()
    rt.load_scene = MagicMock(return_value=mock_scene)  # type: ignore[attr-defined]
    rt.PlanarArray = MagicMock()  # type: ignore[attr-defined]
    rt.Transmitter = MagicMock()  # type: ignore[attr-defined]

    # RadioMapSolver()(scene, ...) が返す radio_map
    mock_radio_map = MagicMock()
    mock_radio_map.rss.numpy.return_value = np.full((1, 4, 4), 1e-6)  # 1 µW
    rt.RadioMapSolver = MagicMock(return_value=MagicMock(return_value=mock_radio_map))  # type: ignore[attr-defined]


_install_stubs()

# スタブ登録後にインポートする
from src.radio_map_estimation.scene.radiomap_solver import (  # noqa: E402
    radio_map_to_rss_dbm,
    run_radio_map,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_building_data():
    """BuildingData の最小スタブ (area_spec のみ参照される)."""
    spec = MagicMock()
    spec.area_size_m = 100.0
    data = MagicMock()
    data.area_spec = spec
    return data


@pytest.fixture
def rss_w_3x3() -> np.ndarray:
    """3 x 3 の RSS [W] サンプル (num_tx=1)."""
    return np.array([[[1e-3, 0.0, 1e-6], [1e-9, 1e-3, 0.0], [0.0, 1e-12, 1e-3]]])


# ---------------------------------------------------------------------------
# run_radio_map  (smoke test)
# ---------------------------------------------------------------------------


class TestRunRadioMap:
    def test_smoke(self, dummy_building_data, tmp_path):
        """Sionna をスタブ化した状態で例外なく実行できること."""
        scene_xml = tmp_path / "scene.xml"
        scene_xml.touch()

        result = run_radio_map(
            scene_xml=scene_xml,
            tx_positions=[(50.0, 50.0, 10.0)],
            frequency_hz=2.4e9,
            tx_power_dbm=23.0,
            building_data=dummy_building_data,
            rx_height_m=1.5,
            max_depth=3,
            cell_size_m=5.0,
            samples_per_tx=100,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# radio_map_to_rss_dbm
# ---------------------------------------------------------------------------


class TestRadioMapToRssDbm:
    def test_output_shape(self, rss_w_3x3):
        """出力 shape が入力と同じであること."""
        mock_rm = MagicMock()
        mock_rm.rss.numpy.return_value = rss_w_3x3
        rss_dbm = radio_map_to_rss_dbm(mock_rm)
        assert rss_dbm.shape == rss_w_3x3.shape

    def test_no_nan_in_output(self, rss_w_3x3):
        """出力に nan が含まれないこと (0 W セルも置換されるため)."""
        mock_rm = MagicMock()
        mock_rm.rss.numpy.return_value = rss_w_3x3
        rss_dbm = radio_map_to_rss_dbm(mock_rm)
        assert not np.isnan(rss_dbm).any()

    def test_zero_rss_replaced_by_floor(self, rss_w_3x3):
        """rss=0 のセルが -120 dBm (1e-15 W 相当) に置換されること."""
        mock_rm = MagicMock()
        mock_rm.rss.numpy.return_value = rss_w_3x3
        rss_dbm = radio_map_to_rss_dbm(mock_rm)

        floor_dbm = 10.0 * np.log10(1e-15) + 30.0  # = -120.0
        zero_mask = rss_w_3x3 == 0.0
        np.testing.assert_allclose(rss_dbm[zero_mask], floor_dbm, atol=1e-6)

    def test_known_value(self):
        """既知の RSS [W] → [dBm] 変換が正しいこと. 1e-3 W = 0 dBm."""
        mock_rm = MagicMock()
        mock_rm.rss.numpy.return_value = np.array([[[1e-3]]])
        rss_dbm = radio_map_to_rss_dbm(mock_rm)
        assert rss_dbm[0, 0, 0] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize(
        "rss_w,expected_dbm",
        [
            (1.0, 30.0),  # 1 W    = 30 dBm
            (1e-3, 0.0),  # 1 mW   =  0 dBm
            (1e-6, -30.0),  # 1 µW   = -30 dBm
        ],
    )
    def test_parametrized_conversion(self, rss_w, expected_dbm):
        """W → dBm 変換の各ケースを検証する."""
        mock_rm = MagicMock()
        mock_rm.rss.numpy.return_value = np.array([[[rss_w]]])
        rss_dbm = radio_map_to_rss_dbm(mock_rm)
        assert rss_dbm[0, 0, 0] == pytest.approx(expected_dbm, abs=1e-4)
