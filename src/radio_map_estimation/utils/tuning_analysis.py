"""チューニング結果の分析: summary.json の収集とベスト条件選定の共通ロジック"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TuningRecord:
    """1つの param_hash に対する集約結果 (1エリア x 1周波数分)"""

    city_dir: str
    mesh_code: str
    freq_ghz: str
    param_hash: str
    n_layers: int
    n_neurons: int
    lr: float
    batch_size: int
    n_epochs: int
    n_trials: int
    train_rmse_db_mean: float
    train_rmse_db_std: float
    test_rmse_db_mean: float
    test_rmse_db_std: float
    overfit_gap_db: float


def collect_records(tuning_root: Path, search_dir_name: str, run_id: str) -> list[TuningRecord]:
    """outputs/tuning/**/{search_dir_name}/{run_id}/**/summary.json を全件読み込む

    ディレクトリ構造:
        outputs/tuning/{city_dir}/{mesh_code}/{freq_ghz}/{search_dir_name}/{run_id}/{param_hash}/summary.json
    """
    records: list[TuningRecord] = []

    for summary_path in tuning_root.glob(f"*/*/*/{search_dir_name}/{run_id}/*/summary.json"):
        param_dir = summary_path.parent
        freq_ghz = param_dir.parents[2].name
        mesh_code = param_dir.parents[3].name
        city_dir = param_dir.parents[4].name

        with open(summary_path) as f:
            summary = json.load(f)

        params = summary["params"]
        train_mean = float(summary["train_rmse_db_mean"])
        test_mean = float(summary["test_rmse_db_mean"])

        records.append(
            TuningRecord(
                city_dir=city_dir,
                mesh_code=mesh_code,
                freq_ghz=freq_ghz,
                param_hash=param_dir.name,
                n_layers=int(params["n_layers"]),
                n_neurons=int(params["n_neurons"]),
                lr=float(params["lr"]),
                batch_size=int(params["batch_size"]),
                n_epochs=int(params["n_epochs"]),
                n_trials=int(summary["n_trials"]),
                train_rmse_db_mean=train_mean,
                train_rmse_db_std=float(summary["train_rmse_db_std"]),
                test_rmse_db_mean=test_mean,
                test_rmse_db_std=float(summary["test_rmse_db_std"]),
                overfit_gap_db=test_mean - train_mean,
            )
        )

    if not records:
        raise FileNotFoundError(
            f"summary.json が見つかりません: {tuning_root}/*/*/*/{search_dir_name}/{run_id}/*/summary.json"
        )
    return records


def select_best_per_group(records: list[TuningRecord]) -> list[TuningRecord]:
    """(city_dir, mesh_code, freq_ghz) ごとに test_rmse_db_mean 最小の record を選ぶ"""
    groups: dict[tuple[str, str, str], list[TuningRecord]] = {}
    for r in records:
        key = (r.city_dir, r.mesh_code, r.freq_ghz)
        groups.setdefault(key, []).append(r)

    best_records: list[TuningRecord] = []
    for key in sorted(groups.keys()):
        group_records = groups[key]
        best = min(group_records, key=lambda r: r.test_rmse_db_mean)
        best_records.append(best)
    return best_records


def save_records_csv(records: list[TuningRecord], path: Path) -> None:
    """TuningRecord のリストを CSV として保存する"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
