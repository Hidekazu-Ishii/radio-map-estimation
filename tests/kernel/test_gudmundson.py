"""
GudmundsonKernel の pytest テストスイート

解析式に対して、
make_input / eval / grad_at / set_log_params / __call__ の各契約
 (Kernel 抽象基底クラスが要求する性質) を検証する
"""

from __future__ import annotations

import numpy as np
import pytest

from radio_map_estimation.shadowing.kernels.gudmundson import GudmundsonKernel

# ------------------------------------------------------------------
# fixtures (実験条件の明示)
# ------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """テストデータ生成専用の Generator (seed はテストの入口でのみ固定する)"""
    return np.random.default_rng(0)


@pytest.fixture
def coords(rng: np.random.Generator) -> np.ndarray:
    """ランダムな受信点座標 (x, y) [m]、shape (5, 2)"""
    return rng.uniform(0.0, 100.0, size=(5, 2))


@pytest.fixture
def kernel() -> GudmundsonKernel:
    """初期値 sigma_2=4.0 [dB²], d_cor=20.0 [m] の GudmundsonKernel"""
    return GudmundsonKernel(sigma_2_init=4.0, d_cor_init=20.0)


# ------------------------------------------------------------------
# make_input: ユークリッド距離行列
# ------------------------------------------------------------------


def test_make_input_matches_euclidean_distance(kernel, coords):
    """make_input() が正しいユークリッド距離行列を返すことを確認する"""
    d = kernel.make_input(coords, coords)

    # scipy を使わず素朴な二重ループで正解を計算し、突き合わせる
    n = coords.shape[0]
    expected = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            expected[i, j] = np.linalg.norm(coords[i] - coords[j])

    assert d.shape == (n, n)
    assert np.allclose(d, expected)


def test_make_input_diagonal_is_zero(kernel, coords):
    """自分自身との距離は 0 であることを確認する"""
    d = kernel.make_input(coords, coords)
    assert np.allclose(np.diag(d), 0.0)


def test_make_input_ignores_extra_kwargs(kernel, coords):
    """GudmundsonKernel は TX 座標などの追加 kwargs を無視することを確認する"""
    d_with_kwargs = kernel.make_input(coords, coords, tx_coords_a=coords, tx_coords_b=coords)
    d_without_kwargs = kernel.make_input(coords, coords)
    assert np.allclose(d_with_kwargs, d_without_kwargs)


# ------------------------------------------------------------------
# eval: カーネル行列の解析式との一致
# ------------------------------------------------------------------


def test_eval_matches_analytic_formula(kernel, coords):
    """eval() が k と一致することを確認する"""
    d = kernel.make_input(coords, coords)
    log_params = np.log(np.array([4.0, 20.0]))  # sigma_2=4.0, d_cor=20.0

    k = kernel.eval(d, log_params)
    expected = 4.0 * np.exp(-d * np.log(2.0) / 20.0)

    assert np.allclose(k, expected)


def test_eval_diagonal_equals_sigma_2(kernel, coords):
    """距離 0 (自己共分散) では k が相関 1 になることを確認する"""
    d = kernel.make_input(coords, coords)
    log_params = np.log(np.array([4.0, 20.0]))

    k = kernel.eval(d, log_params)
    assert np.allclose(np.diag(k), 4.0)


def test_eval_is_symmetric(kernel, coords):
    """距離行列が対称なので、カーネル行列も対称であることを確認する"""
    d = kernel.make_input(coords, coords)
    log_params = np.log(np.array([4.0, 20.0]))

    k = kernel.eval(d, log_params)
    assert np.allclose(k, k.T)


def test_eval_decays_monotonically_with_distance(kernel):
    """距離が大きいほどカーネル値 (相関) が単調に減少することを確認する"""
    distances = np.array([[0.0, 10.0, 50.0, 200.0]])  # (1, 4)
    log_params = np.log(np.array([4.0, 20.0]))

    k = kernel.eval(distances, log_params).ravel()
    assert np.all(np.diff(k) < 0.0)  # 距離が増えるほど単調減少


def test_eval_does_not_mutate_internal_state(kernel, coords):
    """eval() は最適化ループ内で呼ばれるため、内部状態を書き換えないことを確認する"""
    d = kernel.make_input(coords, coords)
    params_before = kernel.params

    # 内部状態とは異なる log_params で eval を呼ぶ
    kernel.eval(d, np.log(np.array([99.0, 1.0])))

    assert kernel.params == params_before


# ------------------------------------------------------------------
# grad_at: 解析的勾配 vs 中心差分による数値勾配
# ------------------------------------------------------------------


@pytest.mark.parametrize("sigma_2, d_cor", [(4.0, 20.0), (1.0, 5.0), (10.0, 50.0)])
def test_grad_matches_finite_difference(kernel, coords, sigma_2, d_cor):
    """grad_at() の解析的勾配が中心差分による数値勾配と一致することを確認する"""
    eps = 1e-6
    d = kernel.make_input(coords, coords)
    log_params = np.log(np.array([sigma_2, d_cor]))

    analytic = kernel.grad_at(d, log_params)

    for i, name in enumerate(("sigma_2", "d_cor")):
        params_plus, params_minus = log_params.copy(), log_params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps

        k_plus = kernel.eval(d, params_plus)
        k_minus = kernel.eval(d, params_minus)
        numeric = (k_plus - k_minus) / (2 * eps)

        assert analytic[name] == pytest.approx(numeric, abs=1e-4, rel=1e-3)


def test_grad_at_does_not_mutate_internal_state(kernel, coords):
    """grad_at() も内部状態を書き換えないことを確認する"""
    d = kernel.make_input(coords, coords)
    params_before = kernel.params

    kernel.grad_at(d, np.log(np.array([99.0, 1.0])))

    assert kernel.params == params_before


# ------------------------------------------------------------------
# set_log_params: fit 完了後の内部状態更新
# ------------------------------------------------------------------


def test_set_log_params_updates_internal_state(kernel):
    """set_log_params() が sigma_2 / d_cor を正しく更新することを確認する"""
    kernel.set_log_params(np.log(np.array([9.0, 30.0])))
    assert kernel.params == pytest.approx({"sigma_2": 9.0, "d_cor": 30.0})


def test_call_reflects_updated_params_after_set_log_params(kernel, coords):
    """set_log_params() 後の __call__ が、更新されたパラメータでの eval と一致することを確認する

    __call__ は「fit 後の内部状態でカーネル行列を計算する」契約であり、
    コンストラクタ時の初期値ではなく最新の内部状態を使うことを確認する
    """
    d = kernel.make_input(coords, coords)
    new_log_params = np.log(np.array([9.0, 30.0]))
    kernel.set_log_params(new_log_params)

    k_call = kernel(d)
    k_eval = kernel.eval(d, new_log_params)

    assert np.allclose(k_call, k_eval)


# ------------------------------------------------------------------
# プロパティ契約
# ------------------------------------------------------------------


def test_n_params_is_two(kernel):
    """Gudmundson カーネルのハイパーパラメータ数は 2 (sigma_2, d_cor) であることを確認する"""
    assert kernel.n_params == 2


def test_log_params_init_matches_constructor_values():
    """log_params_init がコンストラクタで渡した初期値の log を返すことを確認する"""
    kernel = GudmundsonKernel(sigma_2_init=2.5, d_cor_init=15.0)
    assert kernel.log_params_init == pytest.approx(np.log(np.array([2.5, 15.0])))


def test_param_bounds_has_no_limits(kernel):
    """param_bounds が sigma_2 / d_cor ともに上下限なしを返すことを確認する"""
    bounds = kernel.param_bounds
    assert bounds == [(None, None), (None, None)]


def test_params_property_matches_constructor_values():
    """params プロパティが元スケールの現在値を辞書で返すことを確認する"""
    kernel = GudmundsonKernel(sigma_2_init=4.0, d_cor_init=20.0)
    assert kernel.params == {"sigma_2": 4.0, "d_cor": 20.0}
