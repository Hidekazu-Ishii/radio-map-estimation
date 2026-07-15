"""
GPShadowingModel の pytest テストスイート

fit → predict_mean / predict_with_uncertainty → covariance_matrix の
一連のパイプラインをスモークテストし、GP エンジンの数値的性質
 (分散の非負性、観測点近傍での分散縮小、解析的勾配の正しさ、再現性) を検証する
"""

from __future__ import annotations

import numpy as np
import pytest

from radio_map_estimation.pathloss.base import FitResult
from radio_map_estimation.shadowing.gp import GPShadowingModel
from radio_map_estimation.shadowing.kernel.gudmundson import GudmundsonKernel

# ------------------------------------------------------------------
# fixtures (実験条件の明示)
# ------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """テストデータ生成専用の Generator (seed はテストの入口でのみ固定する)"""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_data(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """滑らかな空間パターン + 小さな観測ノイズを持つ合成シャドウイングデータ

    真値を sin/cos の滑らかな関数とすることで、GP が学習可能な
    空間相関構造を持たせる
    """
    n = 20
    coords = rng.uniform(0.0, 100.0, size=(n, 2))
    tx_coords = np.tile(np.array([[50.0, 50.0, 10.0]]), (n, 1))
    freq_hz = np.full((n, 1), 3.5e9)

    true_shadowing = 3.0 * np.sin(coords[:, [0]] / 20.0) + 2.0 * np.cos(coords[:, [1]] / 25.0)
    noise = rng.normal(0.0, 0.3, size=(n, 1))
    residuals = true_shadowing + noise

    return {
        "coords": coords,
        "tx_coords": tx_coords,
        "freq_hz": freq_hz,
        "residuals": residuals,
    }


@pytest.fixture
def gp_model() -> GPShadowingModel:
    """軽量設定 (シングルスタート・少ない反復数) の GPShadowingModel"""
    kernel = GudmundsonKernel(sigma_2_init=4.0, d_cor_init=20.0)
    return GPShadowingModel(
        kernel=kernel,
        sigma_n_2_init=0.5,
        sigma_n_2_min=1e-3,
        sigma_n_2_max=10.0,
        n_restarts=1,
        max_iter=50,
        ftol=1e-6,
        gtol=1e-5,
    )


# ------------------------------------------------------------------
# スモークテスト: fit → predict のパイプライン
# ------------------------------------------------------------------


def test_fit_returns_valid_fitresult(gp_model, synthetic_data, rng):
    """fit() が妥当な FitResult を返すことを確認する"""
    result = gp_model.fit(rng=rng, **synthetic_data)

    assert isinstance(result, FitResult)
    assert result.n_samples == synthetic_data["coords"].shape[0]
    assert result.rmse_db >= 0.0
    assert "sigma_n_2" in result.params


def test_predict_mean_shape(gp_model, synthetic_data, rng):
    """predict_mean() が (M, 1) 形状を返すことを確認する"""
    gp_model.fit(rng=rng, **synthetic_data)

    m = 5
    coords_test = rng.uniform(0.0, 100.0, size=(m, 2))
    tx_coords_test = np.tile(np.array([[50.0, 50.0, 10.0]]), (m, 1))
    freq_hz_test = np.full((m, 1), 3.5e9)

    mean = gp_model.predict_mean(coords_test, tx_coords_test, freq_hz_test)
    assert mean.shape == (m, 1)


def test_predict_with_uncertainty_variance_nonnegative(gp_model, synthetic_data, rng):
    """事後分散が常に非負であることを確認する"""
    gp_model.fit(rng=rng, **synthetic_data)

    coords_test = synthetic_data["coords"]
    tx_coords_test = synthetic_data["tx_coords"]
    freq_hz_test = synthetic_data["freq_hz"]

    mean, var = gp_model.predict_with_uncertainty(coords_test, tx_coords_test, freq_hz_test)
    assert mean.shape == var.shape == (coords_test.shape[0], 1)
    assert np.all(var >= 0.0)


def test_variance_shrinks_near_training_points(gp_model, synthetic_data, rng):
    """訓練点そのものでは、訓練領域から遠く離れた未観測点より事後分散が小さいことを確認する"""
    gp_model.fit(rng=rng, **synthetic_data)

    coords_train = synthetic_data["coords"]
    tx_coords_train = synthetic_data["tx_coords"]
    freq_hz_train = synthetic_data["freq_hz"]

    # 訓練領域 [0, 100]^2 から大きく離れた点 (事前分散に近いはず)
    coords_far = np.array([[500.0, 500.0]])
    tx_coords_far = np.array([[50.0, 50.0, 10.0]])
    freq_hz_far = np.array([[3.5e9]])

    _, var_train = gp_model.predict_with_uncertainty(coords_train, tx_coords_train, freq_hz_train)
    _, var_far = gp_model.predict_with_uncertainty(coords_far, tx_coords_far, freq_hz_far)

    assert np.mean(var_train) < var_far.item()


# ------------------------------------------------------------------
# API 契約: フィット前アクセス禁止
# ------------------------------------------------------------------


def test_params_raises_before_fit(gp_model):
    """fit() 前に params にアクセスすると RuntimeError を送出することを確認する"""
    with pytest.raises(RuntimeError):
        _ = gp_model.params


def test_predict_raises_before_fit(gp_model, synthetic_data):
    """fit() 前に predict_mean() を呼ぶと RuntimeError を送出することを確認する"""
    with pytest.raises(RuntimeError):
        gp_model.predict_mean(
            synthetic_data["coords"], synthetic_data["tx_coords"], synthetic_data["freq_hz"]
        )


# ------------------------------------------------------------------
# 共分散行列の性質
# ------------------------------------------------------------------


def test_covariance_matrix_symmetric_psd(gp_model, synthetic_data, rng):
    """covariance_matrix() が対称・半正定値であることを確認する"""
    gp_model.fit(rng=rng, **synthetic_data)

    k = gp_model.covariance_matrix(synthetic_data["coords"], synthetic_data["tx_coords"])

    assert np.allclose(k, k.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(k)
    assert np.all(eigvals >= -1e-6)  # 数値誤差を許容した半正定値性チェック


# ------------------------------------------------------------------
# 再現性: 同じ Generator (同じ seed) なら同じ結果
# ------------------------------------------------------------------


def test_fit_is_reproducible_with_same_seed(synthetic_data):
    """同じ seed の Generator を渡せば、fit 結果 (パラメータ) が一致することを確認する"""

    def build_and_fit(seed: int) -> dict[str, float]:
        kernel = GudmundsonKernel(sigma_2_init=4.0, d_cor_init=20.0)
        model = GPShadowingModel(
            kernel=kernel,
            sigma_n_2_init=0.5,
            sigma_n_2_min=1e-3,
            sigma_n_2_max=10.0,
            n_restarts=3,
            max_iter=50,
            ftol=1e-6,
            gtol=1e-5,
        )
        result = model.fit(rng=np.random.default_rng(seed), **synthetic_data)
        return result.params  # type: ignore

    params_a = build_and_fit(seed=123)
    params_b = build_and_fit(seed=123)

    for key in params_a:
        assert params_a[key] == pytest.approx(params_b[key])


# ------------------------------------------------------------------
# 勾配の正しさ: 解析的勾配 vs 中心差分による数値勾配
# ------------------------------------------------------------------


def test_nlml_gradient_matches_finite_difference(gp_model, synthetic_data):
    """_nlml_grad の解析的勾配が中心差分による数値勾配と一致することを確認する

    fit() を呼ばず、_nlml / _nlml_grad を直接比較する (最適化の収束性とは独立に検証)
    """
    eps = 1e-5
    kernel = gp_model._kernel
    y = synthetic_data["residuals"].ravel()
    k_input = kernel.make_input(
        synthetic_data["coords"],
        synthetic_data["coords"],
        tx_coords_a=synthetic_data["tx_coords"],
        tx_coords_b=synthetic_data["tx_coords"],
    )
    x0 = np.append(kernel.log_params_init, np.log(0.5))

    analytic_grad = gp_model._nlml_grad(x0, k_input, y)

    numeric_grad = np.zeros_like(x0)
    for i in range(len(x0)):
        x_plus, x_minus = x0.copy(), x0.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        numeric_grad[i] = (gp_model._nlml(x_plus, k_input, y) - gp_model._nlml(x_minus, k_input, y)) / (
            2 * eps
        )

    assert analytic_grad == pytest.approx(numeric_grad, abs=1e-3, rel=1e-2)


# ------------------------------------------------------------------
# ハイパーパラメータの組み合わせ (multi-start の頑健性)
# ------------------------------------------------------------------


@pytest.mark.parametrize("n_restarts", [1, 3])
@pytest.mark.parametrize("d_cor_init", [5.0, 20.0, 50.0])
def test_fit_converges_across_hyperparameter_grid(synthetic_data, n_restarts, d_cor_init):
    """n_restarts と d_cor_init の組み合わせによらず fit が正常終了することを確認する"""
    kernel = GudmundsonKernel(sigma_2_init=4.0, d_cor_init=d_cor_init)
    model = GPShadowingModel(
        kernel=kernel,
        sigma_n_2_init=0.5,
        sigma_n_2_min=1e-3,
        sigma_n_2_max=10.0,
        n_restarts=n_restarts,
        max_iter=50,
        ftol=1e-6,
        gtol=1e-5,
    )
    rng = np.random.default_rng(7)

    result = model.fit(rng=rng, **synthetic_data)

    assert np.isfinite(result.rmse_db)
    assert result.params["sigma_n_2"] > 0.0  # type: ignore
