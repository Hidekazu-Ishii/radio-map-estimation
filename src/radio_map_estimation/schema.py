"""
実験設定のスキーマ定義と .yaml のloader.

- AreaConfig: 対象エリア (configs/cities/{city_name}.yaml)
- SimulationConfig: シミュレーション設定 (configs/simulation.yaml)
- NormalizeConfig: 正規化設定 (configs/normalize.yaml)
"""

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class AreaConfig:
    center_lat: float
    center_lon: float
    area_size_m: float
    cell_size_m: float
    meters_per_level: float
    default_building_height_m: float
    building_type_levels: dict[str, float]
    n_trials: int
    master_seed: int


@dataclass
class SimulationConfig:
    tx_height_m: float
    tx_power_dbm: float
    tx_intensity: float
    frequency_hz: float
    rx_height_m: float
    max_depth: int
    samples_per_tx: int
    building_material: str
    ground_material: str
    noise_std_db: float


def load_area_config(config_path: Path) -> AreaConfig:
    cfg = OmegaConf.load(config_path)
    return AreaConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]


def load_simulation_config(config_path: Path) -> SimulationConfig:
    cfg = OmegaConf.load(config_path)
    return SimulationConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
