"""FFNNModel / FFNNLosModel のテスト"""

import numpy as np
import pytest

from radio_map_estimation.loader.dataset import GridInfo
from radio_map_estimation.pathloss.ffnn import FFNNModel
from radio_map_estimation.pathloss.ffnn_los import FFNNLosModel

N_SAMPLES = 40


# ------------------------------------------------------------------
# fixtures: 実験条件を明示
# ------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def grid_info() -> GridInfo:
    """20x20 の bldg_mask、中央に建物ブロックを配置"""
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 8:12] = True
    return GridInfo(
        bldg_mask=mask,
        bldg_cell_size_m=1.0,
        cell_size_m=1.0,
        area_size_m=20.0,
        margin_m=0.0,
    )


@pytest.fixture
def synthetic_dataset(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """log-distance則 + ノイズによる合成観測データ (小規模スモークテスト用)"""
    coords = rng.uniform(0.0, 20.0, size=(N_SAMPLES, 2))
    tx_coords = np.tile(np.array([[10.0, 10.0, 20.0]]), (N_SAMPLES, 1))
    rx_height_m = np.full((N_SAMPLES, 1), 1.5)
    freq_hz = np.full((N_SAMPLES, 1), 2.4e9)
    tx_power_dbm = np.full((N_SAMPLES, 1), 20.0)

    # log-distance則のシンプルなパスロス生成 (3D距離をその場で計算)
    rx_xyz = np.hstack([coords, rx_height_m])
    d = np.clip(np.linalg.norm(rx_xyz - tx_coords, axis=1, keepdims=True), 1e-3, None)
    pathloss_db = 30.0 + 20.0 * np.log10(d) + rng.normal(0.0, 1.0, size=(N_SAMPLES, 1))
    rss_dbm_obs = tx_power_dbm - pathloss_db

    return {
        "coords": coords,
        "tx_coords": tx_coords,
        "rx_height_m": rx_height_m,
        "freq_hz": freq_hz,
        "tx_power_dbm": tx_power_dbm,
        "rss_dbm_obs": rss_dbm_obs,
    }


# ------------------------------------------------------------------
# FFNNModel
# ------------------------------------------------------------------


@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_ffnn_fit_predict_shape(
    synthetic_dataset: dict[str, np.ndarray], grid_info: GridInfo, rng: np.random.Generator, n_layers: int
) -> None:
    # fit → predict_mean のパイプラインが通り、出力shapeが (N,1) になること
    model = FFNNModel(n_neurons=8, n_layers=n_layers, n_epochs=2, batch_size=16, lr=0.01)
    fit_result = model.fit(**synthetic_dataset, grid_info=grid_info, rng=rng)
    assert fit_result.n_samples == N_SAMPLES
    assert fit_result.rmse_db >= 0.0

    rss_pred = model.predict_mean(
        coords=synthetic_dataset["coords"],
        tx_coords=synthetic_dataset["tx_coords"],
        rx_height_m=synthetic_dataset["rx_height_m"],
        freq_hz=synthetic_dataset["freq_hz"],
        tx_power_dbm=synthetic_dataset["tx_power_dbm"],
        grid_info=grid_info,
    )
    assert rss_pred.shape == (N_SAMPLES, 1)


def test_ffnn_reproducibility_with_same_seed(
    synthetic_dataset: dict[str, np.ndarray], grid_info: GridInfo
) -> None:
    # 同じseedのrngから作った2つのモデルは同じ予測を返すこと (再現性の確認)
    model_a = FFNNModel(n_neurons=8, n_layers=1, n_epochs=2, batch_size=16, lr=0.01)
    model_b = FFNNModel(n_neurons=8, n_layers=1, n_epochs=2, batch_size=16, lr=0.01)

    model_a.fit(**synthetic_dataset, grid_info=grid_info, rng=np.random.default_rng(123))
    model_b.fit(**synthetic_dataset, grid_info=grid_info, rng=np.random.default_rng(123))

    pred_a = model_a.predict_mean(
        synthetic_dataset["coords"],
        synthetic_dataset["tx_coords"],
        synthetic_dataset["rx_height_m"],
        synthetic_dataset["freq_hz"],
        synthetic_dataset["tx_power_dbm"],
        grid_info,
    )
    pred_b = model_b.predict_mean(
        synthetic_dataset["coords"],
        synthetic_dataset["tx_coords"],
        synthetic_dataset["rx_height_m"],
        synthetic_dataset["freq_hz"],
        synthetic_dataset["tx_power_dbm"],
        grid_info,
    )
    np.testing.assert_allclose(pred_a, pred_b)


def test_ffnn_params_before_fit_raises() -> None:
    # fit前にparamsへアクセスするとRuntimeError
    model = FFNNModel(n_neurons=8, n_layers=1, n_epochs=1, batch_size=16, lr=0.01)
    with pytest.raises(RuntimeError):
        _ = model.params


def test_ffnn_predict_before_fit_raises(grid_info: GridInfo) -> None:
    model = FFNNModel(n_neurons=8, n_layers=1, n_epochs=1, batch_size=16, lr=0.01)
    coords = np.zeros((1, 2))
    tx_coords = np.zeros((1, 3))
    rx_height_m = np.zeros((1, 1))
    freq_hz = np.full((1, 1), 2.4e9)
    tx_power_dbm = np.full((1, 1), 20.0)
    with pytest.raises(RuntimeError):
        model.predict_mean(coords, tx_coords, rx_height_m, freq_hz, tx_power_dbm, grid_info)


# ------------------------------------------------------------------
# FFNNLosModel
# ------------------------------------------------------------------


@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_ffnn_los_fit_predict_shape(
    synthetic_dataset: dict[str, np.ndarray], grid_info: GridInfo, rng: np.random.Generator, n_layers: int
) -> None:
    # LOS遮蔽特徴付きモデルでもパイプラインが通ること (入力4次元)
    model = FFNNLosModel(n_neurons=8, n_layers=n_layers, n_epochs=2, batch_size=16, lr=0.01)
    fit_result = model.fit(**synthetic_dataset, grid_info=grid_info, rng=rng)
    assert fit_result.model_name == "ffnn_los"
    assert "bldg_count_min" in fit_result.norm_stats

    rss_pred = model.predict_mean(
        coords=synthetic_dataset["coords"],
        tx_coords=synthetic_dataset["tx_coords"],
        rx_height_m=synthetic_dataset["rx_height_m"],
        freq_hz=synthetic_dataset["freq_hz"],
        tx_power_dbm=synthetic_dataset["tx_power_dbm"],
        grid_info=grid_info,
    )
    assert rss_pred.shape == (N_SAMPLES, 1)


def test_ffnn_los_params_before_fit_raises() -> None:
    model = FFNNLosModel(n_neurons=8, n_layers=1, n_epochs=1, batch_size=16, lr=0.01)
    with pytest.raises(RuntimeError):
        _ = model.params
