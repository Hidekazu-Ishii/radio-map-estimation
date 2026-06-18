"""
エントリポイント: データローダーの動作確認スクリプト

モデルに渡すデータ (TrainData / TestData) が正しく構築されることを確認する。
city x mesh_code x freq_hz x train_rate x trial の全組み合わせに対して
データを構築し、形状・値域を表示する。

Usage:
    uv run scripts/demo.py configs/scene.yaml configs/sionna.yaml configs/dataloader.yaml
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from omegaconf import DictConfig, OmegaConf

from radio_map_estimation.loader.loader import load_dataset
from radio_map_estimation.loader.tensors import TestTensors, TrainTensors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------------


def freq_dir_name(freq_hz: float) -> str:
    """周波数 [Hz] からディレクトリ名を生成する。例: 2.0e9 → '2.0GHz'"""
    ghz = freq_hz / 1e9
    return f"{ghz:.10g}GHz" if ghz != int(ghz) else f"{ghz:.1f}GHz"


# ------------------------------------------------------------------
# 設定
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DataLoaderConfig:
    """configs/dataloader.yaml に対応する設定"""

    train_rates: tuple[float, ...]  # 学習点のサンプリング率 [%]
    test_size: int  # 評価点数 (固定)
    n_trials: int  # 繰り返し実験回数
    master_seed: int  # trial_idx=i に対して master_seed + i を使用


def load_config(config_path: Path) -> DataLoaderConfig:
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(cfg)}")
    train_rates = tuple(float(r) for r in cfg.train_rate)
    if not train_rates:
        raise ValueError("train_rate must not be empty")
    if any(not (0.0 < r <= 100.0) for r in train_rates):
        raise ValueError(f"All train_rate values must be in (0, 100], got {train_rates}")
    return DataLoaderConfig(
        train_rates=train_rates,
        test_size=int(cfg.test_size),
        n_trials=int(cfg.n_trials),
        master_seed=int(cfg.master_seed),
    )


# ------------------------------------------------------------------
# エントリポイント
# ------------------------------------------------------------------


def main(
    scene_config_path: Path,
    sionna_config_path: Path,
    dataloader_config_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]

    # --- 設定読み込み ---
    # scene / sionna は simulate.py でバリデーション済みのため直接参照
    scene_cfg = OmegaConf.load(scene_config_path)
    sionna_cfg = OmegaConf.load(sionna_config_path)
    dl_cfg = load_config(dataloader_config_path)
    logger.info(
        "[config] train_rates=%s, test_size=%d, n_trials=%d",
        list(dl_cfg.train_rates),
        dl_cfg.test_size,
        dl_cfg.n_trials,
    )

    # --- city x mesh_code x freq_hz のループ ---
    for city in scene_cfg.cities:
        for mesh_code in city.mesh_codes:
            for freq_hz in sionna_cfg.frequency_hz:
                npz_path = (
                    root
                    / "data"
                    / "processed"
                    / str(city.city_dir)
                    / str(mesh_code)
                    / freq_dir_name(float(freq_hz))
                    / "radio_map.npz"
                )
                logger.info(
                    "[npz] city=%s, mesh_code=%s, freq=%.4gGHz",
                    city.city_dir,
                    mesh_code,
                    float(freq_hz) / 1e9,
                )

                # npz を1回だけ読み込む
                npz_data = np.load(npz_path)
                n_observable = int(np.sum(~np.isnan(npz_data["rss_dbm_gt"])))
                logger.info("[npz] observable cells: %d", n_observable)

                # --- train_rate x trial のループ ---
                for train_rate in dl_cfg.train_rates:
                    # train_rate [%] → 点数に変換
                    train_size = max(1, int(n_observable * train_rate / 100.0))

                    for trial_idx in range(dl_cfg.n_trials):
                        # trial ごとにシードを派生させて独立したサンプリングを保証
                        rng = default_rng(dl_cfg.master_seed + trial_idx)

                        train_data, test_data = load_dataset(
                            data=npz_data,
                            train_size=train_size,
                            test_size=dl_cfg.test_size,
                            rng=rng,
                        )

                        train_tensors = TrainTensors.from_train_data(train_data)
                        test_tensors = TestTensors.from_test_data(test_data)

                        logger.info(
                            "[check] train_rate=%.0f%%, trial=%d: "
                            "train=%d, test=%d, "
                            "rss_dbm_obs(train) min=%.2f max=%.2f, "
                            "rss_dbm_gt(test) min=%.2f max=%.2f",
                            train_rate,
                            trial_idx,
                            len(train_tensors),
                            len(test_tensors),
                            train_tensors.rss_dbm_obs.min().item(),
                            train_tensors.rss_dbm_obs.max().item(),
                            test_tensors.rss_dbm_gt.min().item(),
                            test_tensors.rss_dbm_gt.max().item(),
                        )

    logger.info("[done] DataLoader is working correctly.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <scene.yaml> <sionna.yaml> <dataloader.yaml>")
        sys.exit(1)

    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
