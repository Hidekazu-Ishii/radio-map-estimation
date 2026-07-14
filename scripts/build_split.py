"""
PoolTestSplit (test_prod 固定領域 + チューニング用 pool) を確定するエントリポイント

一度だけ実行し、生成された split.npz は以後変更しない。
チューニング・本番実験のどちらも、この split.npz を読み込んで使う。

使い方:
    uv run scripts/build_split.py configs/plateau.yaml configs/sionna.yaml configs/split.yaml
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from radio_map_estimation.loader.dataset import PoolTestSplit
from radio_map_estimation.utils.naming import freq_dir_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 設定 dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """split.yaml 全体の設定

    test_size   : test_prod のセル数 (固定・以後変更しない)
    master_seed : split 確定専用の乱数シード (チューニング・本番実験の seed とは別系統)
    """

    test_size: int
    master_seed: int

    @classmethod
    def from_omega(cls, cfg: DictConfig) -> SplitConfig:
        test_size = int(cfg.test_size)
        if test_size <= 0:
            raise ValueError(f"test_size must be positive, got {test_size}")
        return cls(test_size=test_size, master_seed=int(cfg.master_seed))


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------


def main(
    plateau_config_path: Path,
    sionna_config_path: Path,
    split_config_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]

    plateau_cfg: DictConfig = OmegaConf.load(plateau_config_path)  # type: ignore[assignment]
    sionna_cfg: DictConfig = OmegaConf.load(sionna_config_path)  # type: ignore[assignment]
    split_cfg = SplitConfig.from_omega(OmegaConf.load(split_config_path))  # type: ignore[arg-type]

    # 乱数シードはエントリポイントで1回だけ固定する
    rng = np.random.default_rng(split_cfg.master_seed)

    for area in plateau_cfg.areas:
        for mesh_code in area.mesh_codes:
            for freq_hz in sionna_cfg.frequency_hz:
                freq_ghz = freq_dir_name(float(freq_hz))
                data_dir = root / "data" / "processed" / str(area.city_dir) / str(mesh_code) / freq_ghz
                radiomap_path = data_dir / "radio_map.npz"
                split_path = data_dir / "pool_test_split.npz"

                if split_path.exists():
                    logger.info(
                        "[skip] already exists: city=%s mesh=%s freq=%s -> %s",
                        area.city_dir,
                        mesh_code,
                        freq_ghz,
                        split_path.relative_to(root),
                    )
                    continue

                radiomap_data = np.load(radiomap_path)
                rss_dbm_gt: np.ndarray = radiomap_data["rss_dbm_gt"]

                split = PoolTestSplit.create(rss_dbm_gt, split_cfg.test_size, rng)
                split.save(split_path)

                logger.info(
                    "[done] city=%s mesh=%s freq=%s | test=%d pool=%d -> %s",
                    area.city_dir,
                    mesh_code,
                    freq_ghz,
                    len(split.test_flat_indices),
                    len(split.pool_flat_indices),
                    split_path.relative_to(root),
                )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <plateau.yaml> <sionna.yaml> <split.yaml>")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
