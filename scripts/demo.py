"""
エントリポイント: パスロス + Gudmundsonカーネルによるシャドウイングモデルの動作確認スクリプト

city x mesh_code x freq_hz x train_size x trial の全組み合わせに対して
pathloss_model でフィット → 残差を GP(Gudmundsonカーネル) でシャドウイング推定 → 結果を保存する。

出力ディレクトリ構造:
    outputs/scratch/{city_dir}/{mesh_code}/{freq_ghz}/
        train{train_size}_trial{trial_idx}/{pathloss_model}_gudmundson/{}/
            ├── config.yaml        # 実験設定の完全な記録
            ├── fit_results.json   # フィット結果 (パラメータ・RMSE)
            └── predictions.npz    # 予測値・GT・座標・訓練データ

Usage:
    uv run scripts/run_gudmundson.py configs/plateau.yaml configs/sionna.yaml configs/exp_gudmundson.yaml
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.random import default_rng
from omegaconf import DictConfig, OmegaConf

from radio_map_estimation.loader.loader import load_dataset
from radio_map_estimation.pathloss.base import FitResult, PathLossModel
from radio_map_estimation.pathloss.close_in import CIModel
from radio_map_estimation.pathloss.ffnn import FFNNModel
from radio_map_estimation.pathloss.ffnn_los import FFNNLosModel
from radio_map_estimation.pathloss.floating_intercept import FIModel
from radio_map_estimation.shadowing.gp import GPShadowingModel
from radio_map_estimation.shadowing.kernels.gudmundson import GudmundsonKernel
from radio_map_estimation.utils.visualize import save_rss_png, scatter_to_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 設定
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    # 実験管理
    train_sizes: tuple[int, ...]
    test_size: int | None  # None のとき train 以外の全点を使用
    n_trials: int
    master_seed: int
    # モデル選択
    pathloss_model: str  # "ci" | "fi" | "ffnn"
    shadowing_model: str  # "gp"
    kernel: str
    # FFNN ハイパーパラメータ
    ffnn_n_neurons: int
    ffnn_n_layers: int
    ffnn_n_epochs: int
    ffnn_batch_size: int
    ffnn_lr: float
    # GP 共通ハイパーパラメータ
    gp_sigma_n_2_init: float
    gp_sigma_n_2_min: float
    gp_sigma_n_2_max: float
    n_restarts: int
    gp_max_iter: int
    gp_ftol: float
    gp_gtol: float
    # Gudmundson カーネルハイパーパラメータ
    gudmundson_sigma_2_init: float
    gudmundson_d_cor_init: float


def load_experiment_config(path: Path) -> ExperimentConfig:
    cfg: DictConfig = OmegaConf.load(path)  # type: ignore[assignment]

    train_sizes = tuple(int(s) for s in cfg.train_size)
    if not train_sizes or any(s <= 0 for s in train_sizes):
        raise ValueError(f"Invalid train_size: {train_sizes}")

    test_size = int(cfg.test_size) if cfg.test_size is not None else None

    pathloss_model = str(cfg.pathloss_model)
    if pathloss_model not in ("ci", "fi", "ffnn", "ffnn_los"):
        raise ValueError(f"Unknown pathloss_model: {pathloss_model!r}")

    match pathloss_model:
        case "ffnn" | "ffnn_los":
            ffnn_n_neurons = int(cfg.ffnn.n_neurons)
            ffnn_n_layers = int(cfg.ffnn.n_layers)
            ffnn_n_epochs = int(cfg.ffnn.n_epochs)
            ffnn_batch_size = int(cfg.ffnn.batch_size)
            ffnn_lr = float(cfg.ffnn.lr)
        case "ci" | "fi":
            pass

    shadowing_model = str(cfg.shadowing_model)
    if shadowing_model not in ("gp",):
        raise ValueError(f"Unknown shadowing_model: {shadowing_model!r}")

    return ExperimentConfig(
        train_sizes=train_sizes,
        test_size=test_size,
        n_trials=int(cfg.n_trials),
        master_seed=int(cfg.master_seed),
        pathloss_model=pathloss_model,
        shadowing_model=shadowing_model,
        kernel=str(cfg.kernel),
        ffnn_n_neurons=ffnn_n_neurons,
        ffnn_n_layers=ffnn_n_layers,
        ffnn_n_epochs=ffnn_n_epochs,
        ffnn_batch_size=ffnn_batch_size,
        ffnn_lr=ffnn_lr,
        gp_sigma_n_2_init=float(cfg.gp.sigma_n_2_init),
        gp_sigma_n_2_min=float(cfg.gp.sigma_n_2_min),
        gp_sigma_n_2_max=float(cfg.gp.sigma_n_2_max),
        n_restarts=int(cfg.gp.n_restarts),
        gp_max_iter=int(cfg.gp.max_iter),
        gp_ftol=float(cfg.gp.ftol),
        gp_gtol=float(cfg.gp.gtol),
        gudmundson_sigma_2_init=float(cfg.gudmundson.sigma_2_init),
        gudmundson_d_cor_init=float(cfg.gudmundson.d_cor_init),
    )


# ------------------------------------------------------------------
# モデル構築
# ------------------------------------------------------------------


def create_pathloss_model(
    cfg: ExperimentConfig,
    bldg_cell_size_m: float,
) -> PathLossModel:
    match cfg.pathloss_model:
        case "ci":
            return CIModel()
        case "fi":
            return FIModel()
        case "ffnn":
            return FFNNModel(
                n_neurons=cfg.ffnn_n_neurons,
                n_layers=cfg.ffnn_n_layers,
                n_epochs=cfg.ffnn_n_epochs,
                batch_size=cfg.ffnn_batch_size,
                lr=cfg.ffnn_lr,
            )
        case "ffnn_los":
            return FFNNLosModel(
                n_neurons=cfg.ffnn_n_neurons,
                n_layers=cfg.ffnn_n_layers,
                n_epochs=cfg.ffnn_n_epochs,
                batch_size=cfg.ffnn_batch_size,
                lr=cfg.ffnn_lr,
            )
        case _:
            raise NotImplementedError(f"pathloss_model={cfg.pathloss_model!r} は未実装")


def create_shadowing_model(cfg: ExperimentConfig) -> GPShadowingModel:
    """Gudmundson カーネル固定で GPShadowingModel を返す"""
    kernel = GudmundsonKernel(
        sigma_2_init=cfg.gudmundson_sigma_2_init,
        d_cor_init=cfg.gudmundson_d_cor_init,
    )
    return GPShadowingModel(
        kernel=kernel,
        sigma_n_2_init=cfg.gp_sigma_n_2_init,
        sigma_n_2_min=cfg.gp_sigma_n_2_min,
        sigma_n_2_max=cfg.gp_sigma_n_2_max,
        n_restarts=cfg.n_restarts,
        max_iter=cfg.gp_max_iter,
        ftol=cfg.gp_ftol,
        gtol=cfg.gp_gtol,
    )


# ------------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------------


def freq_dir_name(freq_hz: float) -> str:
    """周波数 [Hz] → ディレクトリ名。例: 2.0e9 → '2.0GHz'"""
    ghz = freq_hz / 1e9
    return f"{ghz:.10g}GHz" if ghz != int(ghz) else f"{ghz:.1f}GHz"


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def make_output_dir(
    root: Path,
    city_dir: str,
    mesh_code: str,
    freq_ghz: str,
    train_size: int,
    trial_idx: int,
    pathloss_model: str,
    kernel: str,
) -> Path:
    out_dir = (
        root
        / "outputs"
        / "scratch"
        / city_dir
        / mesh_code
        / freq_ghz
        / f"train{train_size}_trial{trial_idx}"
        / f"{pathloss_model}_{kernel}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_fit_results(
    out_dir: Path,
    pl_fit: FitResult,
    sh_fit: FitResult,
    pathloss_only_rmse_db: float,
    test_rmse_db: float,
    gp_gain_db: float,
) -> None:
    results = {
        "pathloss": {
            "model": pl_fit.model_name,
            "params": pl_fit.params,
            "norm_stats": pl_fit.norm_stats,
            "n_samples": pl_fit.n_samples,
            "rmse_db": pl_fit.rmse_db,
        },
        "shadowing": {
            "model": sh_fit.model_name,
            "params": sh_fit.params,
            "n_samples": sh_fit.n_samples,
            "rmse_db": sh_fit.rmse_db,
        },
        "pathloss_only_rmse_db": pathloss_only_rmse_db,
        "test_rmse_db": test_rmse_db,
        "gp_gain_db": gp_gain_db,
    }
    with open(out_dir / "fit_results.json", "w") as f:
        json.dump(results, f, indent=2)


def save_predictions(
    out_dir: Path,
    train_data,
    test_data,
    rss_mean_test: np.ndarray,
    shadowing_mean: np.ndarray,
    shadowing_var: np.ndarray,
    rss_pred: np.ndarray,
) -> None:
    np.savez(
        out_dir / "predictions.npz",
        train_coords=train_data.coords,
        train_tx_coords=train_data.tx_coords,
        train_rss_dbm_obs=train_data.rss_dbm_obs,
        test_coords=test_data.coords,
        test_tx_coords=test_data.tx_coords,
        test_rss_dbm_gt=test_data.rss_dbm_gt,
        rss_mean_test=rss_mean_test,
        shadowing_mean=shadowing_mean,
        shadowing_var=shadowing_var,
        rss_pred=rss_pred,
    )


# ------------------------------------------------------------------
# エントリポイント
# ------------------------------------------------------------------


def main(
    plateau_config_path: Path,
    sionna_config_path: Path,
    experiment_config_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]

    plateau_cfg: DictConfig = OmegaConf.load(plateau_config_path)  # type: ignore[assignment]
    sionna_cfg: DictConfig = OmegaConf.load(sionna_config_path)  # type: ignore[assignment]
    exp_cfg = load_experiment_config(experiment_config_path)

    logger.info(
        "[config] =%s, pathloss=%s, kernel=gudmundson",
        exp_cfg.pathloss_model,
    )

    for area in plateau_cfg.areas:
        for mesh_code in area.mesh_codes:
            bldgmap_path = root / "data" / "processed" / str(area.city_dir) / str(mesh_code) / "bldg_map.npz"
            bldgmap_data = np.load(bldgmap_path)
            for freq_hz in sionna_cfg.frequency_hz:
                freq_ghz = freq_dir_name(float(freq_hz))
                radiomap_path = (
                    root
                    / "data"
                    / "processed"
                    / str(area.city_dir)
                    / str(mesh_code)
                    / freq_ghz
                    / "radio_map.npz"
                )
                logger.info("[npz] city=%s, mesh=%s, freq=%s", area.city_dir, mesh_code, freq_ghz)

                radiomap_data = np.load(radiomap_path)

                for train_size in exp_cfg.train_sizes:
                    for trial_idx in range(exp_cfg.n_trials):
                        rng = default_rng(exp_cfg.master_seed + trial_idx)

                        train_data, test_data, grid_info = load_dataset(
                            bldgmap_data,
                            radiomap_data,
                            train_size=train_size,
                            test_size=exp_cfg.test_size,
                            rng=rng,
                        )

                        # --- パスロスモデル ---
                        pl_model = create_pathloss_model(
                            exp_cfg,
                            bldg_cell_size_m=grid_info.bldg_cell_size_m,
                        )
                        pl_fit = pl_model.fit(
                            coords=train_data.coords,
                            tx_coords=train_data.tx_coords,
                            rx_height_m=train_data.rx_height_m,
                            freq_hz=train_data.freq_hz,
                            tx_power_dbm=train_data.tx_power_dbm,
                            rss_dbm_obs=train_data.rss_dbm_obs,
                            grid_info=grid_info,
                            rng=rng,
                        )
                        rss_mean_train = pl_model.predict_mean(
                            coords=train_data.coords,
                            tx_coords=train_data.tx_coords,
                            rx_height_m=train_data.rx_height_m,
                            freq_hz=train_data.freq_hz,
                            tx_power_dbm=train_data.tx_power_dbm,
                            grid_info=grid_info,  # type: ignore
                        )
                        rss_mean_test = pl_model.predict_mean(
                            coords=test_data.coords,
                            tx_coords=test_data.tx_coords,
                            rx_height_m=test_data.rx_height_m,
                            freq_hz=test_data.freq_hz,
                            tx_power_dbm=test_data.tx_power_dbm,
                            grid_info=grid_info,  # type: ignore
                        )

                        # --- シャドウイングモデル ---
                        residuals = train_data.rss_dbm_obs - rss_mean_train
                        sh_model = create_shadowing_model(exp_cfg)
                        sh_fit = sh_model.fit(
                            coords=train_data.coords,
                            tx_coords=train_data.tx_coords,
                            freq_hz=train_data.freq_hz,
                            residuals=residuals,
                            rng=rng,
                        )
                        shadowing_mean, shadowing_var = sh_model.predict_with_uncertainty(
                            coords=test_data.coords,
                            tx_coords=test_data.tx_coords,
                            freq_hz=test_data.freq_hz,
                        )

                        # --- 最終予測と評価 ---
                        rss_pred = rss_mean_test + shadowing_mean
                        pathloss_only_rmse_db = rmse(rss_mean_test, test_data.rss_dbm_gt)
                        test_rmse_db = rmse(rss_pred, test_data.rss_dbm_gt)
                        gp_gain_db = pathloss_only_rmse_db - test_rmse_db

                        # --- 保存 ---
                        out_dir = make_output_dir(
                            root,
                            area.city_dir,
                            mesh_code,
                            freq_ghz,
                            train_size,
                            trial_idx,
                            pathloss_model=exp_cfg.pathloss_model,
                            kernel=exp_cfg.kernel,
                        )
                        shutil.copy(experiment_config_path, out_dir / "config.yaml")
                        save_fit_results(
                            out_dir,
                            pl_fit,
                            sh_fit,
                            pathloss_only_rmse_db,
                            test_rmse_db,
                            gp_gain_db,
                        )
                        save_predictions(
                            out_dir,
                            train_data,
                            test_data,
                            rss_mean_test,
                            shadowing_mean,
                            shadowing_var,
                            rss_pred,
                        )
                        if exp_cfg.pathloss_model in ["ffnn", "ffnn_los"]:
                            torch.save(pl_model._net.state_dict(), out_dir / "pathloss.pth")  # type: ignore

                        # --- 可視化 ---
                        all_coords = np.concatenate([train_data.coords, test_data.coords], axis=0)

                        save_rss_png(
                            rss_dbm=scatter_to_grid(
                                train_data.coords,
                                train_data.rss_dbm_obs,
                                grid_info.area_size_m,
                                grid_info.cell_size_m,
                            ),
                            tx_coords=train_data.tx_coords,
                            area_size_m=grid_info.area_size_m,
                            output_path=out_dir / "train_only.png",
                            title="Train observations",
                            bldg_mask=grid_info.bldg_mask,
                        )

                        save_rss_png(
                            rss_dbm=scatter_to_grid(
                                train_data.coords,
                                residuals,
                                grid_info.area_size_m,
                                grid_info.cell_size_m,
                            ),
                            tx_coords=train_data.tx_coords,
                            area_size_m=grid_info.area_size_m,
                            output_path=out_dir / "shadowing_train_only.png",
                            title="Shadowing (train residuals only) [db]",
                            bldg_mask=grid_info.bldg_mask,
                            vmin=-20,
                            vmax=20,
                        )

                        shadowing_gt_test = test_data.rss_dbm_gt - rss_mean_test

                        save_rss_png(
                            rss_dbm=scatter_to_grid(
                                all_coords,
                                np.concatenate([residuals, shadowing_gt_test]),
                                grid_info.area_size_m,
                                grid_info.cell_size_m,
                            ),
                            tx_coords=train_data.tx_coords,
                            area_size_m=grid_info.area_size_m,
                            output_path=out_dir / "shadowing_gt.png",
                            title="Shadowing GT [dB]",
                            bldg_mask=grid_info.bldg_mask,
                            vmin=-20,
                            vmax=20,
                        )

                        save_rss_png(
                            rss_dbm=scatter_to_grid(
                                all_coords,
                                np.concatenate([residuals, shadowing_mean]),
                                grid_info.area_size_m,
                                grid_info.cell_size_m,
                            ),
                            tx_coords=train_data.tx_coords,
                            area_size_m=grid_info.area_size_m,
                            output_path=out_dir / "shadowing_pred.png",
                            title="Shadowing Pred [dB]",
                            bldg_mask=grid_info.bldg_mask,
                            vmin=-20,
                            vmax=20,
                        )

                        shadowing_std = np.sqrt(shadowing_var)
                        train_std_zeros = np.zeros((len(train_data.coords), 1))

                        save_rss_png(
                            rss_dbm=scatter_to_grid(
                                all_coords,
                                np.concatenate([train_std_zeros, shadowing_std]),
                                grid_info.area_size_m,
                                grid_info.cell_size_m,
                            ),
                            tx_coords=train_data.tx_coords,
                            area_size_m=grid_info.area_size_m,
                            output_path=out_dir / "Uncertainty.png",
                            title="GP Predictive Uncertainty [dB]",
                            bldg_mask=grid_info.bldg_mask,
                            vmin=0,
                            vmax=8,
                        )

                        save_rss_png(
                            rss_dbm=scatter_to_grid(
                                all_coords,
                                np.concatenate([train_data.rss_dbm_obs, rss_pred]),
                                grid_info.area_size_m,
                                grid_info.cell_size_m,
                            ),
                            tx_coords=train_data.tx_coords,
                            area_size_m=grid_info.area_size_m,
                            output_path=out_dir / "train_test_pred.png",
                            title="RSS Prediction [dBm]",
                            bldg_mask=grid_info.bldg_mask,
                        )

                        logger.info(
                            "[pathloss] model=%s | params=%s | train_rmse=%.2fdB",
                            pl_fit.model_name,
                            pl_fit.formatted_params(),
                            pl_fit.rmse_db,
                        )
                        logger.info(
                            "[shadowing] model=%s | params=%s | train_rmse=%.2fdB",
                            sh_fit.model_name,
                            sh_fit.formatted_params(),
                            sh_fit.rmse_db,
                        )
                        logger.info(
                            "[result] train=%d trial=%d | pl_only=%.2fdB test=%.2fdB gain=%.2fdB -> %s",
                            train_size,
                            trial_idx,
                            pathloss_only_rmse_db,
                            test_rmse_db,
                            gp_gain_db,
                            out_dir.relative_to(root),
                        )

                        logger.info("[done] complete.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <plateau.yaml> <sionna.yaml> <experiment.yaml>")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
