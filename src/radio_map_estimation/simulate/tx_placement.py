# src/radio_map_estimation/simulate/tx_placement.py

import numpy as np
from numpy.random import Generator
from scipy.ndimage import generic_filter


def compute_open_building_mask(
    building_mask: np.ndarray,
    min_open_neighbors,
) -> np.ndarray:
    """
    開放度が高い建物セルマスクを返す.

    min_open_neighbors 以上の非建物セルに隣接する建物セルを返す.

    Parameters
    ----------
    building_mask : ndarray of shape (H, W), dtype bool
    min_open_neighbors : int
        TX 配置候補として「周囲が開けている建物セル」を選ぶための指標.

    Returns
    -------
    corner_mask : ndarray of shape (H, W), dtype bool
    """

    def count_open_neighbors(values: np.ndarray) -> float:
        """8近傍の非建物セル数を返す (中心セル自身を除く)."""
        center_idx = len(values) // 2
        neighbors = np.concatenate([values[:center_idx], values[center_idx + 1 :]])
        return float(np.sum(neighbors == 0))

    open_neighbor_count: np.ndarray = generic_filter(
        building_mask.astype(float),
        function=count_open_neighbors,
        size=3,
        mode="nearest",  # グリッド外は最近傍セルで埋める (inner_mask で端は除外済みのため実質無関係)
    ).astype(int)

    # 建物セル かつ 開放近傍数が閾値以上
    corner_mask: np.ndarray = building_mask & (open_neighbor_count >= min_open_neighbors)
    return corner_mask


def place_tx_ppp(
    rng: Generator,
    area_size_m: float,
    intensity: float,
    building_heights: np.ndarray,
    cell_size_m: float,
    building_mask: np.ndarray,
    min_open_neighbors: int,
    min_separation_m: float,
    tx_height_above_building_m: float,
    min_building_height_m: float,
    max_building_height_m: float,
    height_weight_power: float,
    inner_margin_m: float,
) -> np.ndarray:
    """
    現実的な制約を考慮した PPP で TX を配置する.

    制約:
        1. 建物屋上への設置:
            building_mask=True のセルのみ候補
        2. 道路近接:
            建物隅セルのみ (隣接8セルに min_open_neighbors 以上の非建物セルが存在)
        3. 高さ制限:
            min_building_height_m 以上 max_building_height_m 以下の建物のみが対象
        4. 高い建物ほど選ばれやすい:
            height_weight_power による重み付きサンプリング
        5. 最小離間距離:
            min_separation_m
        6. エリア内部への配置:
            inner_margin_m 分だけエリア端を除外

    基地局を配置可能なセルがなくなった場合, その時点での配置とする.

    Parameters
    ----------
    rng : Generator
        呼び出し元から受け渡す乱数生成器.
    area_size_m : float
        エリア一辺 [m].
    intensity : float
        TX 密度 [TX/m^2]. 期待配置数 = intensity x area_size_m^2.
    building_heights : ndarray of shape (H, W)
        建物高さグリッド [m].
    cell_size_m : float
    proximity_mask : ndarray of shape (H, W), dtype bool
        compute_road_proximity_mask() の出力.
        True のセルのみ TX 候補.
    min_separation_m : float
        TX 間の最小離間距離 [m].
    tx_height_above_building_m : float
        建物高さへの上乗せ (基地局自体の高さ) [m] (屋上からのオフセット).
    min_building_height_m : float
        TX を設置する建物の最低高さ [m]. これ未満の建物は候補から除外.
    max_building_height_m : float
        TX を設置する建物の最高高さ [m]. これ超の建物は候補から除外.
    height_weight_power : float
        高さの重みの指数.
    inner_margin_m : float
        エリア端から除外する幅 [m].

    Returns
    -------
    tx_positions : np.ndarray of shape (T, 3) [m]
        TX の (x, y, z) 座標. ローカル座標系 (bbox左下を原点).
        z = 建物高さ + tx_height_above_building_m.
    """
    grid_size = building_mask.shape[0]

    # --- 候補セルの生成 ---

    # 1. 建物隅セル (道路近接の代理指標)
    corner_mask = compute_open_building_mask(building_mask, min_open_neighbors)

    # 2. 高さ制限フィルタ (20〜40m の建物のみ)
    height_filter: np.ndarray = (building_heights >= min_building_height_m) & (
        building_heights <= max_building_height_m
    )

    # 3. エリア内部フィルタ (inner_margin_m 分だけ端を除外)
    margin_cells = int(inner_margin_m / cell_size_m)
    inner_mask = np.zeros((grid_size, grid_size), dtype=bool)
    inner_mask[margin_cells:-margin_cells, margin_cells:-margin_cells] = True

    # 4. 全制約の AND
    candidate_mask: np.ndarray = corner_mask & height_filter & inner_mask

    rows, cols = np.where(candidate_mask)
    if len(rows) == 0:
        raise RuntimeError(
            "候補セルが0件です. "
            "min_building_height_m / max_building_height_m / inner_margin_m を確認してください."
        )

    print(
        f"[tx] {len(rows)} candidate cells "
        f"(height {min_building_height_m:.0f}~{max_building_height_m:.0f}m, "
        f"inner {area_size_m - 2 * inner_margin_m:.0f}m x {area_size_m - 2 * inner_margin_m:.0f}m)"
    )

    # 高さ重み (線形)
    heights_at_candidates: np.ndarray = building_heights[rows, cols]
    weights: np.ndarray = heights_at_candidates**height_weight_power
    weights /= weights.sum()

    expected_n = intensity * area_size_m**2
    n_tx = max(int(rng.poisson(expected_n)), 1)

    placed: list[tuple[float, float, float]] = []
    max_attempts = n_tx * 500

    attempts = 0
    while len(placed) < n_tx and attempts < max_attempts:
        idx = int(rng.choice(len(rows), p=weights))
        r, c = rows[idx], cols[idx]

        x = (c + rng.uniform(0.0, 1.0)) * cell_size_m
        y = (r + rng.uniform(0.0, 1.0)) * cell_size_m

        too_close = any(
            float(np.linalg.norm(np.array([x, y]) - np.array([px, py]))) < min_separation_m
            for px, py, _ in placed
        )
        if too_close:
            attempts += 1
            continue

        # TX 高さ = 建物高さ + オフセット
        z = float(building_heights[r, c]) + tx_height_above_building_m
        placed.append((x, y, z))
        attempts += 1

    if not placed:
        raise RuntimeError(
            f"TX を1局も配置できませんでした (requested={n_tx}). "
            "min_separation_m を小さくするか intensity を下げてください."
        )

    print(f"[tx] {len(placed)} TX placed")
    return np.array(placed, dtype=float)  # (T, 3)
