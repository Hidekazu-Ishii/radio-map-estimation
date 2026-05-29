# src/radio_map_estimation/simulate/schema.py
"""
simulate パッケージの設定スキーマ (configs/simulation.yaml).
"""

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True)
class SimulationConfig:
    """シミュレーション設定."""

    # --- UMa (Urban Macro) ---
    uma_power_dbm: float  # 送信電力 [dBm]
    uma_n_tx: int  # 配置 TX 数
    uma_min_building_height_m: float  # 候補建物の最低高さ [m]
    uma_max_building_height_m: float  # 候補建物の最高高さ [m]
    uma_height_above_building_m: float  # 屋上オフセット [m]
    uma_min_separation_m: float  # UMa TX 間の最小離間距離 [m]

    # --- UMi (Urban Micro) ---
    umi_power_dbm: float  # 送信電力 [dBm]
    umi_n_tx: int  # 配置 TX 数
    umi_min_building_height_m: float  # 候補建物の最低高さ [m]
    umi_max_building_height_m: float  # 候補建物の最高高さ [m]
    umi_height_above_building_m: float  # 建物高さへの上乗せ [m]
    umi_min_separation_m: float  # UMi TX 間の最小離間距離 [m]
    umi_min_dist_from_uma_m: float  # UMa TX との最小離間距離 [m]

    # --- 共通 TX 配置 ---
    min_open_neighbors: int  # 開放近傍の最小数 (道路近接代理指標)

    # --- 電波設定 ---
    frequency_hz: float
    rx_height_m: float

    # --- Ray Tracing ---
    max_depth: int
    samples_per_tx: int

    # --- 観測ノイズ ---
    noise_std_db: float

    # --- 実験 ---
    n_trials: int
    master_seed: int


def load_simulation_config(config_path: Path) -> SimulationConfig:
    cfg = OmegaConf.load(config_path)
    return SimulationConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
