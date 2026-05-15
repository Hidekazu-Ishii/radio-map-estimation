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

    # 送信機
    tx_power_dbm: float
    tx_intensity: float
    min_open_neighbors: int
    tx_height_above_building_m: float
    min_separation_m: float
    min_building_height_m: float
    max_building_height_m: float
    height_weight_power: float
    inner_margin_m: float
    # 電波
    frequency_hz: float
    rx_height_m: float
    # Ray Tracing
    max_depth: int
    samples_per_tx: int
    # 材質
    building_material: str
    ground_material: str
    # 観測ノイズ
    noise_std_db: float
    # 実験
    n_trials: int
    master_seed: int


def load_simulation_config(config_path: Path) -> SimulationConfig:
    cfg = OmegaConf.load(config_path)
    return SimulationConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
