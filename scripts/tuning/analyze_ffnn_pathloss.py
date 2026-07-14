"""
エントリポイント: FFNN 系モデルのハイパーパラメータチューニング結果の分析

outputs/tuning/{city_dir}/{mesh_code}/{freq_ghz}/{model_name}_search/{run_id}/{param_hash}/summary.json
を全件収集し、エリア(city_dir x mesh_code) x 周波数(freq_ghz) ごとに
test_rmse_db_mean が最小のハイパーパラメータ組み合わせを選定する

選定基準:
    1. 主指標: test_rmse_db_mean が最小 (グリッドサーチの一次比較基準)
    2. 同程度の候補が複数ある場合の参考指標: test_rmse_db_std (trial間のブレの小ささ)
    3. 過学習チェック用の参考指標: overfit_gap_db = test_rmse_db_mean - train_rmse_db_mean

出力:
    outputs/tuning_analysis/{model_name}/results_{run_id}.csv  # 全 param_hash x エリア x 周波数の一覧
    outputs/tuning_analysis/{model_name}/best_{run_id}.csv     # エリア x 周波数ごとの最良条件のみ

Usage:
    uv run scripts/tuning/analyze_ffnn_pathloss.py ffnn tune_v1
    uv run scripts/tuning/analyze_ffnn_pathloss.py ffnn_los tune_v1
"""

from __future__ import annotations

import sys
from pathlib import Path

from radio_map_estimation.utils.tuning_analysis import (
    collect_records,
    save_records_csv,
    select_best_per_group,
)

VALID_MODEL_NAMES = ("ffnn", "ffnn_los")


def main(model_name: str, run_id: str) -> None:
    match model_name:
        case "ffnn":
            search_dir_name = "ffnn_search"
        case "ffnn_los":
            search_dir_name = "ffnn_los_search"
        case _:
            raise ValueError(f"未対応の model_name です: {model_name} (対応: {VALID_MODEL_NAMES})")

    root = Path(__file__).resolve().parents[2]
    tuning_root = root / "outputs" / "tuning"
    out_dir = root / "outputs" / "tuning_analysis" / model_name

    records = collect_records(tuning_root, search_dir_name, run_id)
    records_sorted = sorted(records, key=lambda r: (r.city_dir, r.mesh_code, r.freq_ghz, r.test_rmse_db_mean))
    save_records_csv(records_sorted, out_dir / f"results_{run_id}.csv")

    best_records = select_best_per_group(records)
    save_records_csv(best_records, out_dir / f"best_{run_id}.csv")

    print(f"[collected] {len(records)} records -> {out_dir / f'results_{run_id}.csv'}")
    print(f"[best] {len(best_records)} groups -> {out_dir / f'best_{run_id}.csv'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <model_name: {'|'.join(VALID_MODEL_NAMES)}> <run_id>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
