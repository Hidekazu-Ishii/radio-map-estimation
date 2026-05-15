# src/radio_map_estimation/scene/building_completion.py
"""
建物マスク・高さグリッドの後処理補完.

処理フロー:
    building_mask
        ↓ binary_closing()   壁の小さな穴を塞ぐ
        ↓ binary_fill_holes() 囲まれた内部領域を検出
    enclosed セル → 周囲建物高さの平均で高さ補完
"""

from dataclasses import replace

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, distance_transform_edt

from .schema import BuildingData


def _fill_enclosed_cells(
    building_mask: np.ndarray,
    building_heights: np.ndarray,
    closing_size: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    周囲を建物に囲まれた非建物セルを建物セルとして補完する.

    高さの補完には distance_transform_edt による最近傍建物セルの高さを使用する.

    Parameters
    ----------
    building_mask : ndarray of shape (H, W), dtype bool
    building_heights : ndarray of shape (H, W), dtype float
    closing_size : int
        binary_closing のカーネルサイズ.

    Returns
    -------
    filled_mask : ndarray of shape (H, W), dtype bool
    filled_heights : ndarray of shape (H, W), dtype float
    """
    # Step 1: 形態学的閉包で建物壁の小さな穴を塞ぐ
    kernel = np.ones((closing_size, closing_size), dtype=bool)
    closed = binary_closing(building_mask, structure=kernel)

    # Step 2: 閉包後のマスクで囲まれた領域を検出
    filled = binary_fill_holes(closed)
    enclosed: np.ndarray = filled & ~building_mask
    num_enclosed = int(enclosed.sum())

    if num_enclosed == 0:
        print("[completion] No enclosed cells found.")
        return building_mask, building_heights

    # Step 3: 最近傍建物セルの高さで補完
    # 「非建物セルから最も近い建物セル」のインデックスを取得
    nearest_indices: np.ndarray = distance_transform_edt(  # type: ignore
        ~building_mask,
        return_distances=False,
        return_indices=True,
    )
    # nearest_indices: shape (2, H, W) → [0]=row方向, [1]=col方向

    filled_mask = building_mask.copy()
    filled_mask[enclosed] = True

    filled_heights = building_heights.copy()
    # 各囲まれたセルに対して最近傍建物セルの高さを代入
    enclosed_rows, enclosed_cols = np.where(enclosed)
    nearest_rows = nearest_indices[0][enclosed_rows, enclosed_cols]
    nearest_cols = nearest_indices[1][enclosed_rows, enclosed_cols]
    filled_heights[enclosed_rows, enclosed_cols] = building_heights[nearest_rows, nearest_cols]

    return filled_mask, filled_heights


def apply_building_completion(
    data: BuildingData,
    closing_size: int = 3,
) -> BuildingData:
    """
    BuildingData に建物マスク補完を適用して新しい BuildingData を返す.

    Parameters
    ----------
    data : BuildingData
    closing_size : int
        binary_closing のカーネルサイズ.

    Returns
    -------
    BuildingData
        補完済みの BuildingData.
    """
    filled_mask, filled_heights = _fill_enclosed_cells(
        building_mask=data.building_mask,
        building_heights=data.building_heights,
        closing_size=closing_size,
    )

    total = data.building_mask.size
    n_before = int(data.building_mask.sum())
    n_after = int(filled_mask.sum())
    print(
        f"[completion] building coverage: "
        f"{n_after / total:.1%} "
        f"(+{(n_after - n_before) / total:.1%}, {n_after - n_before} cells)"
    )

    # 新オブジェクトの生成

    return replace(
        data,
        building_mask=filled_mask,
        building_heights=filled_heights,
    )
