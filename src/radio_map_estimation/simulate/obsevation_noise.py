# src/radio_map_estimation/simulate/best_server.py
"""
観測ノイズ付加
"""

import numpy as np
from numpy.random import Generator


def add_observation_noise(
    rss_dbm: np.ndarray,
    noise_std_db: float,
    rng: Generator,
) -> np.ndarray:
    """
    RSS マップに観測ノイズ N(0, noise_std_db^2) を付加する.

    建物セル (nan) はそのまま維持する.

    Parameters
    ----------
    rss_dbm : ndarray of shape (H, W)
    noise_std_db : float
    rng : Generator

    Returns
    -------
    rss_observed : ndarray of shape (H, W)
    """
    noise: np.ndarray = rng.normal(0.0, noise_std_db, size=rss_dbm.shape)
    return rss_dbm + noise
