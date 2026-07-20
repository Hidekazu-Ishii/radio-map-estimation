# ruff: noqa: F722
"""
Bresenhamラインアルゴリズム (整数格子上の直線経路列挙)

設計方針
--------
- 2点間を結ぶ格子点を1ピクセルずつ厳密に辿る、任意ペアに使える幾何プリミティブ
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Int
from numpy import ndarray


def bresenham_line(r0: int, c0: int, r1: int, c1: int) -> tuple[Int[ndarray, "L 1"], Int[ndarray, "L 1"]]:
    """Bresenham ラインアルゴリズムで (r0,c0)-(r1,c1) 間の格子点を列挙する

    Parameters
    ----------
    r0, c0 : 始点の (row, col)
    r1, c1 : 終点の (row, col)

    Returns
    -------
    rows, cols : ライン上の行・列インデックス (両端含む、重複なし)
    """
    rows: list[int] = []
    cols: list[int] = []

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        rows.append(r)
        cols.append(c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)
