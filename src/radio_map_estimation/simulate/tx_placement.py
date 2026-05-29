# src/radio_map_estimation/simulate/tx_placement.py

import numpy as np
from numpy.random import Generator
from scipy.ndimage import generic_filter


def compute_open_building_mask(
    building_mask: np.ndarray,
    min_open_neighbors: int,
) -> np.ndarray:
    """
    開放度が高い建物セルマスクを返す.

    min_open_neighbors 以上の非建物セルに隣接する建物セルを返す.

    Parameters
    ----------
    building_mask : ndarray of shape (H, W), dtype bool
    min_open_neighbors : int

    Returns
    -------
    corner_mask : ndarray of shape (H, W), dtype bool
    """

    def count_open_neighbors(values: np.ndarray) -> float:
        center_idx = len(values) // 2
        neighbors = np.concatenate([values[:center_idx], values[center_idx + 1 :]])
        return float(np.sum(neighbors == 0))

    open_neighbor_count: np.ndarray = generic_filter(
        building_mask.astype(float),
        function=count_open_neighbors,
        size=3,
        mode="nearest",
    ).astype(int)

    return building_mask & (open_neighbor_count >= min_open_neighbors)


def _build_candidate_mask(
    building_mask: np.ndarray,
    building_heights: np.ndarray,
    min_open_neighbors: int,
    min_building_height_m: float,
    max_building_height_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TX 配置候補セルのマスクと (rows, cols) を返す.

    条件:
        1. 建物セルかつ開放近傍数 >= min_open_neighbors
        2. min_building_height_m <= 建物高さ <= max_building_height_m

    Returns
    -------
    candidate_mask : ndarray of shape (H, W), dtype bool
    rows, cols     : ndarray of shape (N,)
    """
    corner_mask = compute_open_building_mask(building_mask, min_open_neighbors)
    height_filter = (building_heights >= min_building_height_m) & (building_heights <= max_building_height_m)
    candidate_mask = corner_mask & height_filter
    rows, cols = np.where(candidate_mask)
    return candidate_mask, rows, cols


def _sample_with_min_separation(
    rng: Generator,
    rows: np.ndarray,
    cols: np.ndarray,
    building_heights: np.ndarray,
    cell_size_m: float,
    n_tx: int,
    min_separation_m: float,
    height_above_building_m: float,
    forbidden_positions: list[tuple[float, float]] | None = None,
    forbidden_dist_m: float = 0.0,
) -> np.ndarray:
    """
    候補セルから n_tx 局を最小離間距離制約つきでサンプリングする.

    Parameters
    ----------
    rows, cols             : 候補セルのインデックス
    forbidden_positions    : 他タイプ TX の (x, y) リスト (UMi から UMa を除外するため)
    forbidden_dist_m       : forbidden_positions との最小距離 [m]

    Returns
    -------
    tx_positions : ndarray of shape (n_placed, 3)  [m]
        z = 建物高さ + height_above_building_m
    """
    # 一様サンプリング (高さ重みなし)
    perm = rng.permutation(len(rows))
    placed: list[tuple[float, float, float]] = []

    for idx in perm:
        if len(placed) >= n_tx:
            break

        r, c = rows[idx], cols[idx]
        x = (c + rng.uniform(0.0, 1.0)) * cell_size_m  # セル内ランダム
        y = (r + rng.uniform(0.0, 1.0)) * cell_size_m

        # 配置済み TX との最小離間距離チェック
        too_close_placed = any(
            float(np.linalg.norm(np.array([x, y]) - np.array([px, py]))) < min_separation_m
            for px, py, _ in placed
        )
        if too_close_placed:
            continue

        # forbidden_positions (UMa) との距離チェック
        if forbidden_positions and forbidden_dist_m > 0.0:
            too_close_forbidden = any(
                float(np.linalg.norm(np.array([x, y]) - np.array([fx, fy]))) < forbidden_dist_m
                for fx, fy in forbidden_positions
            )
            if too_close_forbidden:
                continue

        z = float(building_heights[r, c]) + height_above_building_m
        placed.append((x, y, z))  # type: ignore

    if len(placed) < n_tx:
        raise RuntimeError(
            f"要求 {n_tx} 局に対して {len(placed)} 局しか配置できませんでした. "
            "min_separation_m を小さくするか，建物高さ範囲を広げてください."
        )

    return np.array(placed, dtype=float)


def place_uma_tx(
    rng: Generator,
    building_heights: np.ndarray,
    building_mask: np.ndarray,
    cell_size_m: float,
    n_tx: int,
    min_open_neighbors: int,
    min_building_height_m: float,
    max_building_height_m: float,
    height_above_building_m: float,
    min_separation_m: float,
) -> np.ndarray:
    """
    UMa (Urban Macro) TX を配置する.

    TR 38.901 UMa 条件:
        hBS = 建物高さ + 2.0 m ≈ 25 m (±5 m)
        → min_building_height_m=18, max_building_height_m=23, height_above_building_m=2.0

    配置条件:
        1. 建物屋上 (building_mask=True)
        2. 開放近傍数 >= min_open_neighbors (道路近接の代理指標)
        3. min_building_height_m <= 建物高さ <= max_building_height_m
        4. TX 間の最小離間距離 >= min_separation_m

    Parameters
    ----------
    rng                      : 呼び出し元から受け渡す乱数生成器
    building_heights         : ndarray of shape (H, W) [m]
    building_mask            : ndarray of shape (H, W), dtype bool
    cell_size_m              : セルサイズ [m]
    n_tx                     : 配置する TX 数 (決定論的)
    min_open_neighbors       : 開放近傍の最小数
    min_building_height_m    : 候補建物の最低高さ [m]
    max_building_height_m    : 候補建物の最高高さ [m]
    height_above_building_m  : 屋上からのオフセット [m]
    min_separation_m         : TX 間の最小離間距離 [m]

    Returns
    -------
    tx_positions : ndarray of shape (n_tx, 3) [m]  (x, y, z)
    """
    _, rows, cols = _build_candidate_mask(
        building_mask,
        building_heights,
        min_open_neighbors,
        min_building_height_m,
        max_building_height_m,
    )

    if len(rows) == 0:
        raise RuntimeError(
            f"UMa 候補セルが 0 件です. "
            f"建物高さ範囲 [{min_building_height_m}, {max_building_height_m}] m を確認してください."
        )

    print(f"[UMa] {len(rows)} candidate cells → placing {n_tx} TX")

    tx_positions = _sample_with_min_separation(
        rng=rng,
        rows=rows,
        cols=cols,
        building_heights=building_heights,
        cell_size_m=cell_size_m,
        n_tx=n_tx,
        min_separation_m=min_separation_m,
        height_above_building_m=height_above_building_m,
    )

    print(f"[UMa] {len(tx_positions)} TX placed")
    return tx_positions


def place_umi_tx(
    rng: Generator,
    building_heights: np.ndarray,
    building_mask: np.ndarray,
    cell_size_m: float,
    n_tx: int,
    min_open_neighbors: int,
    min_building_height_m: float,
    max_building_height_m: float,
    height_above_building_m: float,
    min_separation_m: float,
    uma_positions: np.ndarray,
    min_dist_from_uma_m: float,
) -> np.ndarray:
    """
    UMi (Urban Micro) TX を配置する.

    TR 38.901 UMi 条件:
        hBS = 建物高さ + 0.0 m ≈ 10 m (±2 m)  (建物外壁取り付け想定)
        → min_building_height_m=8, max_building_height_m=12, height_above_building_m=0.0

    配置条件:
        1. 建物セル (building_mask=True)
        2. 開放近傍数 >= min_open_neighbors
        3. min_building_height_m <= 建物高さ <= max_building_height_m
        4. TX 間の最小離間距離 >= min_separation_m
        5. UMa TX から min_dist_from_uma_m 以上離れていること

    Parameters
    ----------
    rng                   : 呼び出し元から受け渡す乱数生成器
    building_heights      : ndarray of shape (H, W) [m]
    building_mask         : ndarray of shape (H, W), dtype bool
    cell_size_m           : セルサイズ [m]
    n_tx                  : 配置する TX 数 (決定論的)
    min_open_neighbors    : 開放近傍の最小数
    min_building_height_m : 候補建物の最低高さ [m]
    max_building_height_m : 候補建物の最高高さ [m]
    height_above_building_m : 建物高さへの上乗せ [m]
    min_separation_m      : UMi TX 間の最小離間距離 [m]
    uma_positions         : place_uma_tx の出力 ndarray of shape (T, 3)
    min_dist_from_uma_m   : UMa TX との最小離間距離 [m]

    Returns
    -------
    tx_positions : ndarray of shape (n_tx, 3) [m]  (x, y, z)
    """
    _, rows, cols = _build_candidate_mask(
        building_mask,
        building_heights,
        min_open_neighbors,
        min_building_height_m,
        max_building_height_m,
    )

    if len(rows) == 0:
        raise RuntimeError(
            f"UMi 候補セルが 0 件です. "
            f"建物高さ範囲 [{min_building_height_m}, {max_building_height_m}] m を確認してください."
        )

    print(f"[UMi] {len(rows)} candidate cells → placing {n_tx} TX")

    uma_xy: list[tuple[float, float]] = [(float(p[0]), float(p[1])) for p in uma_positions]

    tx_positions = _sample_with_min_separation(
        rng=rng,
        rows=rows,
        cols=cols,
        building_heights=building_heights,
        cell_size_m=cell_size_m,
        n_tx=n_tx,
        min_separation_m=min_separation_m,
        height_above_building_m=height_above_building_m,
        forbidden_positions=uma_xy,
        forbidden_dist_m=min_dist_from_uma_m,
    )

    print(f"[UMi] {len(tx_positions)} TX placed")
    return tx_positions
