# ruff: noqa: F722
"""
観測可能点 (mask_observed=True) から train / test 点をサンプリングするモジュール

サンプリング戦略:
    1. mask_observed=True の全セルインデックスを取得
    2. 全体から train_size 点を一括サンプリング
    3. 残りから test_size 点をサンプリング (train と disjoint を保証)
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Int


def sample_train_test_indices(
    mask_observed: np.ndarray,
    train_size: int,
    test_size: int,
    rng: np.random.Generator,
) -> tuple[
    Int[np.ndarray, "N 1"],  # train_rows
    Int[np.ndarray, "N 1"],  # train_cols
    Int[np.ndarray, "M 1"],  # test_rows
    Int[np.ndarray, "M 1"],  # test_cols
]:
    """観測可能点から train / test のセルインデックスをサンプリングする

    Parameters
    ----------
    mask_observed : (H, W) bool  観測可能マスク
    train_size    : 学習点数
    test_size     : 予測 (評価) 点数
    rng           : 乱数生成器 (外部から受け取る)

    Returns
    -------
    train_rows, train_cols : shape (N, 1) 学習点のセルインデックス
    test_rows,  test_cols  : shape (M, 1) 予測点のセルインデックス (train と disjoint)

    Raises
    ------
    ValueError
        train_size + test_size が観測可能点数を超える場合
    """
    obs_rows, obs_cols = np.where(mask_observed)  # (num_obs,) それぞれ
    num_obs = len(obs_rows)

    if train_size + test_size > num_obs:
        raise ValueError(
            f"train_size + test_size ({train_size + test_size}) exceeds observable cells ({num_obs})"
        )

    all_indices = np.arange(num_obs)

    # train をサンプリング
    train_idx = rng.choice(all_indices, size=train_size, replace=False)
    train_idx.sort()  # 再現確認のため整列

    # 残りから test をサンプリング (disjoint 保証)
    remaining = np.setdiff1d(all_indices, train_idx)
    test_idx = rng.choice(remaining, size=test_size, replace=False)
    test_idx.sort()

    # shape を (N, 1) に統一
    return (
        obs_rows[train_idx].reshape(-1, 1),
        obs_cols[train_idx].reshape(-1, 1),
        obs_rows[test_idx].reshape(-1, 1),
        obs_cols[test_idx].reshape(-1, 1),
    )


def cell_lower_left(
    rows: Int[np.ndarray, "N 1"],
    cols: Int[np.ndarray, "N 1"],
    cell_size_m: float,
) -> Float[np.ndarray, "N 2"]:
    """セルインデックス (row, col) → 左下端座標 (x, y) [m] に変換する

    グリッド定義:
        x = col * cell_size_m
        y = row * cell_size_m

    Parameters
    ----------
    rows, cols   : shape (N, 1) セルの行・列インデックス
    cell_size_m  : セルサイズ [m]

    Returns
    -------
    coords : shape (N, 2) 左下端座標 (x, y) [m]
    """
    x = cols * cell_size_m  # (N, 1)
    y = rows * cell_size_m  # (N, 1)
    return np.concatenate([x, y], axis=1)  # (N, 2)


def resolve_tx_positions(
    rows: Int[np.ndarray, "N 1"],
    cols: Int[np.ndarray, "N 1"],
    tx_association: Int[np.ndarray, "H W"],
    tx_positions: Float[np.ndarray, "T 3"],
) -> tuple[Float[np.ndarray, "N 3"], Int[np.ndarray, "N 1"]]:
    """セルインデックスから接続TX座標と接続TXインデックスを取得する

    Parameters
    ----------
    rows, cols      : shape (N, 1) サンプリング済みセルの行・列インデックス
    tx_association  : (H, W) 各セルの接続TXインデックス (-1 = 未到達)
    tx_positions    : (T, 3) TX位置 [m] (npz の tx_positions キー)

    Returns
    -------
    tx_coords   : shape (N, 3) 接続TX座標 (モデルへの入力特徴量)
    tx_indices  : shape (N, 1) 接続TXインデックス
    """
    # rows, cols は (N, 1) のため squeeze して 2D インデックスとして使用
    tx_indices = tx_association[rows.squeeze(1), cols.squeeze(1)]  # (N,)
    tx_coords = tx_positions[tx_indices]  # (N, 3)
    return tx_coords, tx_indices.reshape(-1, 1)  # (N, 3), (N, 1)
