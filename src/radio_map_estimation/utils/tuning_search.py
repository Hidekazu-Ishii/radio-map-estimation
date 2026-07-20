"""パスロスモデルのハイパーパラメータチューニング (グリッドサーチ) の共通ロジック"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from radio_map_estimation.pathloss.base import FitResult


@dataclass(frozen=True, slots=True)
class FFNNSearchParams:
    """FFNN 系モデル (FFNN / FFNNLos) 共通の 1組み合わせ分のハイパーパラメータ"""

    n_layers: int
    n_neurons: int
    lr: float
    batch_size: int
    n_epochs: int

    def param_hash(self) -> str:
        """可読性重視のディレクトリ名を生成する

        例: nl1_nn100_lr1e-03_bs256_ep500
        """
        return f"nl{self.n_layers}_nn{self.n_neurons}_lr{self.lr:.0e}_bs{self.batch_size}_ep{self.n_epochs}"

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
        }


@dataclass(frozen=True)
class SearchSpace:
    """FFNN 系モデルのグリッドサーチ探索空間 (各パラメータの候補値リスト)"""

    n_layers: tuple[int, ...]
    n_neurons: tuple[int, ...]
    lr: tuple[float, ...]
    batch_size: tuple[int, ...]
    n_epochs: tuple[int, ...]

    def combinations(self) -> list[FFNNSearchParams]:
        """探索空間の直積 (全組み合わせ) を FFNNSearchParams のリストとして返す"""
        product = itertools.product(
            self.n_layers,
            self.n_neurons,
            self.lr,
            self.batch_size,
            self.n_epochs,
        )
        return [
            FFNNSearchParams(n_layers=nl, n_neurons=nn_, lr=lr, batch_size=bs, n_epochs=ep)
            for nl, nn_, lr, bs, ep in product
        ]


@dataclass(frozen=True)
class TuningConfig:
    """チューニング1回分の設定

    train_size / test_size は pool (test_prod を除いた領域) 内でのサンプリングに使う.
    test_size は None 不可: pool 全体を1回の test_tune で使い切ってしまうと、
    Monte Carlo 繰り返しのたびに train_tune との独立性が失われるため、明示指定を必須にする.
    """

    run_id: str
    train_size: int
    test_size: int
    n_trials: int
    master_seed: int
    search_space: SearchSpace


def load_tuning_config(path: Path) -> TuningConfig:
    cfg: DictConfig = OmegaConf.load(path)  # type: ignore[assignment]

    train_size = int(cfg.train_size)
    if train_size <= 0:
        raise ValueError(f"Invalid train_size: {train_size}")

    if cfg.test_size is None:
        raise ValueError(
            "test_size must be specified explicitly for tuning (sampled from the pool). "
            "None (all remaining pool cells) is not allowed, to keep train_tune / test_tune "
            "resampling independent across Monte Carlo trials."
        )
    test_size = int(cfg.test_size)
    if test_size <= 0:
        raise ValueError(f"Invalid test_size: {test_size}")

    search_space = SearchSpace(
        n_layers=tuple(int(v) for v in cfg.search_space.n_layers),
        n_neurons=tuple(int(v) for v in cfg.search_space.n_neurons),
        lr=tuple(float(v) for v in cfg.search_space.lr),
        batch_size=tuple(int(v) for v in cfg.search_space.batch_size),
        n_epochs=tuple(int(v) for v in cfg.search_space.n_epochs),
    )

    return TuningConfig(
        run_id=str(cfg.run_id),
        train_size=train_size,
        test_size=test_size,
        n_trials=int(cfg.n_trials),
        master_seed=int(cfg.master_seed),
        search_space=search_space,
    )


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def make_param_dir(
    root: Path,
    city_dir: str,
    mesh_code: str,
    freq_ghz: str,
    search_dir_name: str,
    run_id: str,
    param_hash: str,
) -> Path:
    param_dir = (
        root / "outputs" / "tuning" / city_dir / mesh_code / freq_ghz / search_dir_name / run_id / param_hash
    )
    param_dir.mkdir(parents=True, exist_ok=True)
    return param_dir


def save_param_config(param_dir: Path, params: FFNNSearchParams) -> None:
    """このparam_hashに対応する具体的なハイパーパラメータ値を保存する"""
    OmegaConf.save(config=OmegaConf.create(params.to_dict()), f=param_dir / "config.yaml")


def save_trial_result(
    param_dir: Path,
    trial_idx: int,
    pl_fit: FitResult,
    test_rmse_db: float,
) -> None:
    result = {
        "model": pl_fit.model_name,
        "params": pl_fit.params,
        "n_samples": pl_fit.n_samples,
        "train_rmse_db": pl_fit.rmse_db,
        "test_rmse_db": test_rmse_db,
    }
    with open(param_dir / f"fit_results_{trial_idx}.json", "w") as f:
        json.dump(result, f, indent=2)


def save_summary(
    param_dir: Path,
    params: FFNNSearchParams,
    train_rmse_list: list[float],
    test_rmse_list: list[float],
) -> None:
    """全trialの集約統計 (mean/std) を保存する"""
    summary = {
        "params": params.to_dict(),
        "n_trials": len(test_rmse_list),
        "train_rmse_db_mean": float(np.mean(train_rmse_list)),
        "train_rmse_db_std": float(np.std(train_rmse_list)),
        "test_rmse_db_mean": float(np.mean(test_rmse_list)),
        "test_rmse_db_std": float(np.std(test_rmse_list)),
    }
    with open(param_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
