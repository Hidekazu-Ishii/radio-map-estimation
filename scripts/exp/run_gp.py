"""
エントリポイント: パスロス + 定常カーネルによるシャドウイングモデルの動作確認スクリプト

city x mesh_code x freq_hz x train_size x trial の全組み合わせに対して
pathloss_model でフィット → 残差を GP(定常カーネル) でシャドウイング推定 → 結果を保存する

train_prod は事前に scripts/build_split.py で確定した PoolTestSplit.pool_flat_indices
からサンプリングされる。test_prod は同 split の test_flat_indices の全件を固定で使い、
trial ごとに再サンプリングしない (本番評価は1回だけを前提とする)。

ffnn / ffnn_los を選択した場合、ハイパーパラメータは YAML に静的に書かず、
scripts/tune_ffnn_los.py + scripts/analyze_ffnn_los_tuning.py が出力した
outputs/tuning_analysis/ffnn_los/best_{ffnn_tuning_run_id}.csv から
(city_dir, mesh_code, freq_ghz) ごとに読み込む
 (チューニング結果を Single Source of Truth とし、config側との重複・ズレを避けるため)

出力ディレクトリ構造:
    outputs/scratch/{city_dir}/{mesh_code}/{freq_ghz}/{pathloss_model}_{shadowing_model}_{kernel}/
            train{train_size}_test{n_test_prod}/trial{trial_idx}/
            ├── config.yaml        # 実験設定の完全な記録
            ├── fit_results.json   # フィット結果 (パラメータ・RMSE)
            └── predictions.npz    # 予測値・GT・座標・訓練データ

Usage:
    uv run scripts/exp/run_gp.py configs/plateau.yaml configs/sionna.yaml configs/exp/gp.yaml
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

from radio_map_estimation.loader.dataset import PoolTestSplit
from radio_map_estimation.loader.loader import load_full_map_data, load_production_data
from radio_map_estimation.pathloss.base import FitResult, PathLossModel
from radio_map_estimation.pathloss.close_in import CIModel
from radio_map_estimation.pathloss.ffnn import FFNNModel
from radio_map_estimation.pathloss.ffnn_los import FFNNLosModel
from radio_map_estimation.pathloss.floating_intercept import FIModel
from radio_map_estimation.shadowing.gp import GPShadowingModel
from radio_map_estimation.shadowing.kernels.gudmundson import GudmundsonKernel
from radio_map_estimation.utils.load_tunedparams import (
    FFNNLosHyperparams,
    get_tuned_params,
    load_tuned_ffnn_los_params,
)
from radio_map_estimation.utils.naming import freq_dir_name
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
    # test_prod のサンプリングには使わない (split ファイルで固定済み)。
    # 指定された場合は split.test_flat_indices の件数との整合性チェックにのみ使う。
    expected_test_size: int | None
    n_trials: int
    master_seed: int
    # モデル選択
    pathloss_model: str  # "ci" | "fi" | "ffnn" | "ffnn_los"
    shadowing_model: str  # "gp"
    kernel: str
    # ffnn / ffnn_los を使う場合のみ有効: チューニング結果の run_id
    ffnn_tuning_run_id: str | None
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

    expected_test_size = int(cfg.test_size) if cfg.get("test_size") is not None else None

    pathloss_model = str(cfg.pathloss_model)
    if pathloss_model not in ("ci", "fi", "ffnn", "ffnn_los"):
        raise ValueError(f"Unknown pathloss_model: {pathloss_model!r}")

    ffnn_tuning_run_id: str | None = None
    match pathloss_model:
        case "ffnn" | "ffnn_los":
            ffnn_tuning_run_id = str(cfg.ffnn_tuning_run_id)
        case "ci" | "fi":
            pass

    shadowing_model = str(cfg.shadowing_model)
    if shadowing_model not in ("gp",):
        raise ValueError(f"Unknown shadowing_model: {shadowing_model!r}")

    return ExperimentConfig(
        train_sizes=train_sizes,
        expected_test_size=expected_test_size,
        n_trials=int(cfg.n_trials),
        master_seed=int(cfg.master_seed),
        pathloss_model=pathloss_model,
        shadowing_model=shadowing_model,
        kernel=str(cfg.kernel),
        ffnn_tuning_run_id=ffnn_tuning_run_id,
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
    tuned_params: dict[tuple[str, str, str], FFNNLosHyperparams] | None,
    city_dir: str,
    mesh_code: str,
    freq_ghz: str,
) -> PathLossModel:
    """パスロスモデルを新規インスタンスとして生成する

    ffnn / ffnn_los の場合、(city_dir, mesh_code, freq_ghz) に対応するチューニング済み
    ハイパーパラメータを tuned_params から都度取得する
    毎回新規インスタンスを生成することで、trial 間でNN重みが引き継がれないことを保証する
    """
    match cfg.pathloss_model:
        case "ci":
            return CIModel()
        case "fi":
            return FIModel()
        case "ffnn":
            assert tuned_params is not None and cfg.ffnn_tuning_run_id is not None
            hp = get_tuned_params(tuned_params, city_dir, mesh_code, freq_ghz, cfg.ffnn_tuning_run_id)
            return FFNNModel(
                n_neurons=hp.n_neurons,
                n_layers=hp.n_layers,
                n_epochs=hp.n_epochs,
                batch_size=hp.batch_size,
                lr=hp.lr,
            )
        case "ffnn_los":
            assert tuned_params is not None and cfg.ffnn_tuning_run_id is not None
            hp = get_tuned_params(tuned_params, city_dir, mesh_code, freq_ghz, cfg.ffnn_tuning_run_id)
            return FFNNLosModel(
                n_neurons=hp.n_neurons,
                n_layers=hp.n_layers,
                n_epochs=hp.n_epochs,
                batch_size=hp.batch_size,
                lr=hp.lr,
            )
        case _:
            raise NotImplementedError(f"pathloss_model={cfg.pathloss_model!r} は未実装")


def create_shadowing_model(cfg: ExperimentConfig) -> GPShadowingModel:
    match cfg.kernel:
        case "gudmundson":
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
        case _:
            raise NotImplementedError(f"kernel={cfg.kernel} は未実装")


# ------------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------------


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def make_output_dir(
    root: Path,
    city_dir: str,
    mesh_code: str,
    freq_ghz: str,
    train_size: int,
    n_test_prod: int,
    trial_idx: int,
    pathloss_model: str,
    shadowing_model: str,
    kernel: str,
) -> Path:
    out_dir = (
        root
        / "outputs"
        / "scratch"
        / city_dir
        / mesh_code
        / freq_ghz
        / f"{pathloss_model}_{shadowing_model}_{kernel}"
        / f"train{train_size}_test{n_test_prod}"
        / f"trial{trial_idx}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_pathloss_weights(
    out_dir: Path,
    pathloss_model: str,
    pl_model: PathLossModel,
) -> None:
    match pathloss_model:
        case "ffnn" | "ffnn_los":
            torch.save(pl_model._net.state_dict(), out_dir / "weights.pth")  # type: ignore
        case _:
            return


def save_fit_results(
    out_dir: Path,
    pl_fit: FitResult,
    sh_fit: FitResult,
    pathloss_only_rmse_db: float,
    pathloss_shadowing_rmse_db: float,
    shadowing_gain_db: float,
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
        "pathloss_shadowing_rmse_db": pathloss_shadowing_rmse_db,
        "shadowing_gain_db": shadowing_gain_db,
    }
    with open(out_dir / "fit_results.json", "w") as f:
        json.dump(results, f, indent=2)


# ------------------------------------------------------------------
# エントリポイント
# ------------------------------------------------------------------


def main(
    plateau_config_path: Path,
    sionna_config_path: Path,
    experiment_config_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]

    plateau_cfg: DictConfig = OmegaConf.load(plateau_config_path)  # type: ignore[assignment]
    sionna_cfg: DictConfig = OmegaConf.load(sionna_config_path)  # type: ignore[assignment]
    exp_cfg = load_experiment_config(experiment_config_path)

    logger.info(
        "pathloss_model=%s, shadowing_model=%s, kernel=%s",
        exp_cfg.pathloss_model,
        exp_cfg.shadowing_model,
        exp_cfg.kernel,
    )

    # --- チューニング済みハイパーパラメータを一度だけロード ---
    tuned_params: dict[tuple[str, str, str], FFNNLosHyperparams] | None = None
    if exp_cfg.pathloss_model in ("ffnn", "ffnn_los"):
        assert exp_cfg.ffnn_tuning_run_id is not None
        tuned_params = load_tuned_ffnn_los_params(root, exp_cfg.ffnn_tuning_run_id)
        logger.info(
            "[tuning] loaded %d (city_dir, mesh_code, freq_ghz) entries from run_id=%s",
            len(tuned_params),
            exp_cfg.ffnn_tuning_run_id,
        )

    for area in plateau_cfg.areas:
        for mesh_code in area.mesh_codes:
            bldgmap_path = root / "data" / "processed" / str(area.city_dir) / str(mesh_code) / "bldg_map.npz"
            bldgmap_data = np.load(bldgmap_path)
            for freq_hz in sionna_cfg.frequency_hz:
                freq_ghz = freq_dir_name(float(freq_hz))
                freq_dir = root / "data" / "processed" / str(area.city_dir) / str(mesh_code) / freq_ghz
                radiomap_path = freq_dir / "radio_map.npz"
                split_path = freq_dir / "pool_test_split.npz"

                logger.info("[npz] city=%s, mesh=%s, freq=%s", area.city_dir, mesh_code, freq_ghz)
                radiomap_data = np.load(radiomap_path)

                # test_prod を確定した split を読み込む (fail loudly: 未生成なら build_split.py を先に実行)
                split = PoolTestSplit.load(split_path)
                n_test_prod = len(split.test_flat_indices)

                if exp_cfg.expected_test_size is not None and exp_cfg.expected_test_size != n_test_prod:
                    raise ValueError(
                        f"config test_size ({exp_cfg.expected_test_size}) does not match "
                        f"split test_prod size ({n_test_prod}) at {split_path}. "
                        "Config is out of sync with the split file."
                    )

                # 可視化専用: 全有効セル (pool + test_prod) の座標を1回だけ用意する
                full_map_data, _ = load_full_map_data(bldgmap_data, radiomap_data)

                for train_size in exp_cfg.train_sizes:
                    for trial_idx in range(exp_cfg.n_trials):
                        rng = default_rng(exp_cfg.master_seed + trial_idx)

                        # train_prod は pool からサンプリング、test_prod は split の全件を固定使用
                        train_data, test_data, grid_info = load_production_data(
                            bldgmap_data,
                            radiomap_data,
                            split=split,
                            train_size=train_size,
                            rng=rng,
                        )

                        # --- パスロスモデル (毎回新規インスタンスを生成)  ---
                        pl_model = create_pathloss_model(
                            exp_cfg,
                            tuned_params,
                            city_dir=str(area.city_dir),
                            mesh_code=str(mesh_code),
                            freq_ghz=freq_ghz,
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
                        pathloss_pred = pl_model.predict_mean(
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
                        rss_pred = pathloss_pred + shadowing_mean
                        pathloss_only_rmse_db = rmse(pathloss_pred, test_data.rss_dbm_gt)
                        pathloss_shadowing_rmse_db = rmse(rss_pred, test_data.rss_dbm_gt)
                        shadowing_gain_db = pathloss_only_rmse_db - pathloss_shadowing_rmse_db
                        shadowing_gt_test = test_data.rss_dbm_gt - pathloss_pred

                        # --- 保存 ---
                        out_dir = make_output_dir(
                            root,
                            area.city_dir,
                            mesh_code,
                            freq_ghz,
                            train_size,
                            n_test_prod=n_test_prod,
                            trial_idx=trial_idx,
                            pathloss_model=exp_cfg.pathloss_model,
                            shadowing_model=exp_cfg.shadowing_model,
                            kernel=exp_cfg.kernel,
                        )
                        shutil.copy(experiment_config_path, out_dir / "config.yaml")
                        save_fit_results(
                            out_dir,
                            pl_fit,
                            sh_fit,
                            pathloss_only_rmse_db,
                            pathloss_shadowing_rmse_db,
                            shadowing_gain_db,
                        )
                        save_pathloss_weights(out_dir, exp_cfg.pathloss_model, pl_model)
                        np.savez(
                            out_dir / "pred.npz",
                            train_coords=train_data.coords,
                            train_tx_coords=train_data.tx_coords,
                            train_rss_dbm_obs=train_data.rss_dbm_obs,
                            train_residuals=residuals,
                            test_coords=test_data.coords,
                            test_tx_coords=test_data.tx_coords,
                            test_rss_dbm_gt=test_data.rss_dbm_gt,
                            pathloss_pred=pathloss_pred,
                            shadowing_gt_test=shadowing_gt_test,
                            shadowing_mean=shadowing_mean,
                            shadowing_var=shadowing_var,
                            rss_pred=rss_pred,
                        )

                        # --- 可視化専用: 全有効セルへの予測 (評価には使わない) ---
                        full_map_pathloss_pred = pl_model.predict_mean(
                            coords=full_map_data.coords,
                            tx_coords=full_map_data.tx_coords,
                            rx_height_m=full_map_data.rx_height_m,
                            freq_hz=full_map_data.freq_hz,
                            tx_power_dbm=full_map_data.tx_power_dbm,
                            grid_info=grid_info,  # type: ignore
                        )
                        full_map_shadowing_mean, full_map_shadowing_var = sh_model.predict_with_uncertainty(
                            coords=full_map_data.coords,
                            tx_coords=full_map_data.tx_coords,
                            freq_hz=full_map_data.freq_hz,
                        )
                        full_map_shadowing_std = np.sqrt(full_map_shadowing_var)
                        full_map_rss_pred = full_map_pathloss_pred + full_map_shadowing_mean

                        np.savez(
                            out_dir / "full_map_pred.npz",
                            coords=full_map_data.coords,
                            pathloss_pred=full_map_pathloss_pred,
                            shadowing_mean=full_map_shadowing_mean,
                            shadowing_var=full_map_shadowing_var,
                            rss_pred=full_map_rss_pred,
                        )

                        save_rss_png(
                            values_db=scatter_to_grid(
                                full_map_data.coords,
                                full_map_shadowing_mean,
                                grid_info.area_size_m,
                                grid_info.cell_size_m,
                            ),
                            tx_coords=train_data.tx_coords,
                            area_size_m=grid_info.area_size_m,
                            output_path=out_dir / "full_map_shadowing_pred.png",
                            title="Full Map Shadowing Pred (visualization only, not for evaluation) [dB]",
                            bldg_mask=grid_info.bldg_mask,
                            vmin=-20,
                            vmax=20,
                        )

                        save_rss_png(
                            values_db=scatter_to_grid(
                                full_map_data.coords,
                                full_map_shadowing_std,
                                grid_info.area_size_m,
                                grid_info.cell_size_m,
                            ),
                            tx_coords=train_data.tx_coords,
                            area_size_m=grid_info.area_size_m,
                            output_path=out_dir / "full_map_uncertainty.png",
                            title="Full Map GP Predictive Uncertainty (visualization only, not for evaluation) [dB]",
                            bldg_mask=grid_info.bldg_mask,
                            vmin=0,
                            vmax=8,
                        )

                        save_rss_png(
                            values_db=scatter_to_grid(
                                full_map_data.coords,
                                full_map_rss_pred,
                                grid_info.area_size_m,
                                grid_info.cell_size_m,
                            ),
                            tx_coords=train_data.tx_coords,
                            area_size_m=grid_info.area_size_m,
                            output_path=out_dir / "full_map_rss_pred.png",
                            title="Full Map RSS Prediction (visualization only, not for evaluation) [dBm]",
                            bldg_mask=grid_info.bldg_mask,
                        )

                        # --- 可視化 ---
                        all_coords = np.concatenate([train_data.coords, test_data.coords], axis=0)

                        save_rss_png(
                            values_db=scatter_to_grid(
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
                            values_db=scatter_to_grid(
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

                        save_rss_png(
                            values_db=scatter_to_grid(
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
                            values_db=scatter_to_grid(
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
                            values_db=scatter_to_grid(
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
                            values_db=scatter_to_grid(
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
                            pathloss_shadowing_rmse_db,
                            shadowing_gain_db,
                            out_dir.relative_to(root),
                        )

                        logger.info("[done] complete.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <plateau.yaml> <sionna.yaml> <experiment.yaml>")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
