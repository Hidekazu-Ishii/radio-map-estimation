"""
チューニング済みハイパーパラメータのロードユーティリティ

scripts/analyze_ffnn_los_tuning.py が出力する
outputs/tuning_analysis/ffnn_los/best_{tuning_run_id}.csv を読み込み、
(city_dir, mesh_code, freq_ghz) をキーとする辞書に変換する

本実験スクリプト側では、この辞書のみをハイパーパラメータの正 (Single Source of Truth)
として参照し、YAML設定ファイルには静的なハイパーパラメータ値を持たせない
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FFNNLosHyperparams:
    """FFNNLosModel 1組み合わせ分のチューニング済みハイパーパラメータ"""

    n_neurons: int
    n_layers: int
    n_epochs: int
    batch_size: int
    lr: float


def load_tuned_ffnn_los_params(
    root: Path,
    tuning_run_id: str,
) -> dict[tuple[str, str, str], FFNNLosHyperparams]:
    """best_{tuning_run_id}.csv を読み込み (city_dir, mesh_code, freq_ghz) → ハイパラ の辞書を返す

    Parameters
    ----------
    root          : プロジェクトルート
    tuning_run_id : scripts/tune_ffnn_los.py 実行時に指定した run_id

    Returns
    -------
    (city_dir, mesh_code, freq_ghz) をキー、FFNNLosHyperparams を値とする辞書

    Raises
    ------
    FileNotFoundError : 該当する best_{tuning_run_id}.csv が存在しない場合
    """
    csv_path = root / "outputs" / "tuning_analysis" / "ffnn_los" / f"best_{tuning_run_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"チューニング結果が見つかりません: {csv_path} "
            f"(先に scripts/tune_ffnn_los.py と scripts/analyze_ffnn_los_tuning.py を実行してください)"
        )

    tuned_params: dict[tuple[str, str, str], FFNNLosHyperparams] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["city_dir"], row["mesh_code"], row["freq_ghz"])
            tuned_params[key] = FFNNLosHyperparams(
                n_neurons=int(row["n_neurons"]),
                n_layers=int(row["n_layers"]),
                n_epochs=int(row["n_epochs"]),
                batch_size=int(row["batch_size"]),
                lr=float(row["lr"]),
            )
    return tuned_params


def get_tuned_params(
    tuned_params: dict[tuple[str, str, str], FFNNLosHyperparams],
    city_dir: str,
    mesh_code: str,
    freq_ghz: str,
    tuning_run_id: str,
) -> FFNNLosHyperparams:
    """(city_dir, mesh_code, freq_ghz) に対応するハイパラを取得する

    見つからない場合はフォールバックせず KeyError で明示的に停止する
     (未チューニングのエリアに気づかず別条件で走ることを防ぐため)
    """
    key = (city_dir, mesh_code, freq_ghz)
    try:
        return tuned_params[key]
    except KeyError:
        raise KeyError(
            f"チューニング結果が見つかりません: "
            f"city_dir={city_dir}, mesh_code={mesh_code}, freq_ghz={freq_ghz}, "
            f"tuning_run_id={tuning_run_id}"
        ) from None
