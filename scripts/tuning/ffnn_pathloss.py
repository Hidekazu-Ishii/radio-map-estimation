"""
エントリポイント: FFNN 系パスロスモデル (FFNN / FFNNLos) のハイパーパラメータチューニング (グリッドサーチ)

シャドウイング推定は含めず、パスロスモデル単体の train/test RMSE のみで評価する
train_size / test_size は全組み合わせ・全エリア・全周波数で固定し、
グリッドサーチの各ハイパーパラメータ組み合わせ (n_layers, n_neurons, lr, batch_size, n_epochs) に対して
n_trials 回 (乱数シード違い) fit -> predict_mean -> RMSE計算 を繰り返す

エリア (city_dir x mesh_code) x 周波数 (freq_hz) ごとに独立してチューニングを行う
 (電波伝搬特性がエリア・周波数に強く依存するため)

train_tune / test_tune のサンプリングは、事前に scripts/build_split.py で確定した
PoolTestSplit.pool_flat_indices の範囲内のみに制限される. test_flat_indices (test_prod)
はこのスクリプトのどの変数にも一切現れない.

出力ディレクトリ構造:
    outputs/tuning/{city_dir}/{mesh_code}/{freq_ghz}/{search_dir_name}/{run_id}/
        {param_hash}/
            config.yaml           # このparam_hashのハイパーパラメータ設定
            fit_results_{trial_idx}.json  # 各trialのtrain/test RMSE・パラメータ
            summary.json          # 全trialの集約統計 (mean/std)

Usage:
    uv run scripts/tuning/ffnn_pathloss.py ffnn configs/data/plateau.yaml configs/data/sionna.yaml configs/tuning/ffnn_model_search.yaml
    uv run scripts/tuning/ffnn_pathloss.py ffnn_los configs/data/plateau.yaml configs/data/sionna.yaml configs/tuning/ffnn_model_search.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from omegaconf import DictConfig, OmegaConf

from radio_map_estimation.loader.dataset import PoolTestSplit
from radio_map_estimation.loader.loader import load_tuning_dataset
from radio_map_estimation.pathloss.ffnn import FFNNModel
from radio_map_estimation.pathloss.ffnn_los import FFNNLosModel
from radio_map_estimation.utils.dir_naming import freq_dir_name
from radio_map_estimation.utils.tuning_search import (
    load_tuning_config,
    make_param_dir,
    rmse,
    save_param_config,
    save_summary,
    save_trial_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VALID_MODEL_NAMES = ("ffnn", "ffnn_los")

# FFNN / FFNNLos は共に (n_neurons, n_layers, n_epochs, batch_size, lr) の
# キーワード引数のみで構築可能なため、コンストラクタとして型を揃えて扱う
type PathlossModelCls = type[FFNNModel] | type[FFNNLosModel]


# ------------------------------------------------------------------
# エントリポイント
# ------------------------------------------------------------------


def main(
    model_name: str,
    plateau_config_path: Path,
    sionna_config_path: Path,
    tuning_config_path: Path,
) -> None:
    match model_name:
        case "ffnn":
            model_cls: PathlossModelCls = FFNNModel
            search_dir_name = "ffnn_search"
        case "ffnn_los":
            model_cls = FFNNLosModel
            search_dir_name = "ffnn_los_search"
        case _:
            raise ValueError(f"未対応の model_name です: {model_name} (対応: {VALID_MODEL_NAMES})")

    root = Path(__file__).resolve().parents[2]

    plateau_cfg: DictConfig = OmegaConf.load(plateau_config_path)  # type: ignore[assignment]
    sionna_cfg: DictConfig = OmegaConf.load(sionna_config_path)  # type: ignore[assignment]
    tune_cfg = load_tuning_config(tuning_config_path)

    param_combinations = tune_cfg.search_space.combinations()
    logger.info(
        "[config] model=%s, run_id=%s, train_size=%d, test_size=%d, n_trials=%d, n_combinations=%d",
        model_name,
        tune_cfg.run_id,
        tune_cfg.train_size,
        tune_cfg.test_size,
        tune_cfg.n_trials,
        len(param_combinations),
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

                logger.info(
                    "[npz] city=%s, mesh=%s, freq=%s",
                    area.city_dir,
                    mesh_code,
                    freq_ghz,
                )
                radiomap_data = np.load(radiomap_path)

                # test_prod を確定した split を読み込む. ここで得られる split から
                # 以後 pool_flat_indices しか参照しない (test_flat_indices はスコープに出さない).
                # 存在しない場合は fail loudly: scripts/build_split.py を先に実行すること.
                split = PoolTestSplit.load(split_path)

                for params in param_combinations:
                    param_dir = make_param_dir(
                        root,
                        area.city_dir,
                        mesh_code,
                        freq_ghz,
                        search_dir_name,
                        tune_cfg.run_id,
                        params.param_hash(),
                    )
                    save_param_config(param_dir, params)

                    train_rmse_list: list[float] = []
                    test_rmse_list: list[float] = []

                    for trial_idx in range(tune_cfg.n_trials):
                        # train/test分割用とモデル学習用で異なるrngを分離
                        #  (分割の再現性とNN初期化の再現性を独立に管理するため)
                        split_rng = default_rng(tune_cfg.master_seed + trial_idx)

                        train_data, test_data, grid_info = load_tuning_dataset(
                            bldgmap_data,
                            radiomap_data,
                            split=split,
                            train_size=tune_cfg.train_size,
                            test_size=tune_cfg.test_size,
                            rng=split_rng,
                        )

                        # --- 新規インスタンスを毎回生成する (重み引き継ぎ防止)  ---
                        pl_model = model_cls(
                            n_neurons=params.n_neurons,
                            n_layers=params.n_layers,
                            n_epochs=params.n_epochs,
                            batch_size=params.batch_size,
                            lr=params.lr,
                        )
                        pl_fit = pl_model.fit(
                            coords=train_data.coords,
                            tx_coords=train_data.tx_coords,
                            rx_height_m=train_data.rx_height_m,
                            freq_hz=train_data.freq_hz,
                            tx_power_dbm=train_data.tx_power_dbm,
                            rss_dbm_obs=train_data.rss_dbm_obs,
                            grid_info=grid_info,
                            rng=split_rng,
                        )
                        rss_mean_test = pl_model.predict_mean(
                            coords=test_data.coords,
                            tx_coords=test_data.tx_coords,
                            rx_height_m=test_data.rx_height_m,
                            freq_hz=test_data.freq_hz,
                            tx_power_dbm=test_data.tx_power_dbm,
                            grid_info=grid_info,
                        )
                        test_rmse_db = rmse(rss_mean_test, test_data.rss_dbm_gt)

                        save_trial_result(param_dir, trial_idx, pl_fit, test_rmse_db)
                        train_rmse_list.append(pl_fit.rmse_db)
                        test_rmse_list.append(test_rmse_db)

                        logger.info(
                            "[trial] %s trial=%d | train_rmse=%.2fdB test_rmse=%.2fdB",
                            params.param_hash(),
                            trial_idx,
                            pl_fit.rmse_db,
                            test_rmse_db,
                        )

                    save_summary(param_dir, params, train_rmse_list, test_rmse_list)
                    logger.info(
                        "[summary] %s | test_rmse_mean=%.2fdB test_rmse_std=%.2fdB -> %s",
                        params.param_hash(),
                        float(np.mean(test_rmse_list)),
                        float(np.std(test_rmse_list)),
                        param_dir.relative_to(root),
                    )

    logger.info("[done] complete.")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            f"Usage: python {sys.argv[0]} <model_name: {'|'.join(VALID_MODEL_NAMES)}> "
            "<plateau.yaml> <sionna.yaml> <tuning.yaml>"
        )
        sys.exit(1)
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
