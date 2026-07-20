"""
エントリポイント: LOS/NLOS幾何接続判定 + スペクトルパワー診断 (経験的バリオグラム診断の後継)

scripts/exp/run_gp.py が出力した pred.npz を読み込み直し、
本体実験 (パスロス+シャドウイングのフィット) を再実行せずに以下を行う:

    1. train + test_prod 全ノードで LOS/NLOS 幾何接続判定 (graph/los.py)
    2. 孤立ノード (LOS隣接ゼロ) を除去
    3. config化された「妥当な初期値」を固定パラメータとしてエッジ重み関数に適用
       (最適化は行わない. 診断の目的は、Wの設計方向が同じ固有値λ_kを共有する
        モード間のパワー分散を減らす方向に効いているかを見ることであり、
        最適パラメータでの性能を測ることではないため)
    4. グラフラプラシアンを構築・固有分解 (graph/spectral.py). unnormalized (L=D-W) と
       normalized (L_sym = I - D^(-1/2)WD^(-1/2)) の両方を毎回計算する
       (normalizedは固有値が必ず[0,2]に収まるため、Wの絶対スケールに依存せず比較できる)
    5. train / heldout 別々にグラフフーリエ変換し、固有値ビンごとのパワー分散を比較

出力 (run_gp.py と同じ trial ディレクトリ直下に追加保存):
    spectral_variance.npz  # {method}__{group}__{bin_centers,mean_power,std_power,counts}
    spectral_variance.png  # unnormalized/normalized を横に並べ、それぞれ train vs heldout を重ね描き

Usage:
    uv run scripts/analyze_spectral_variance.py configs/data/plateau.yaml configs/data/sionna.yaml configs/spectral_variance.yaml
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from radio_map_estimation.graph.edge_weight.base import EdgeWeightFunction
from radio_map_estimation.graph.edge_weight.baseline import GudmundsonEdgeWeight
from radio_map_estimation.graph.los import LosAdjacencyConfig, compute_los_adjacency, filter_isolated_nodes
from radio_map_estimation.graph.spectral import (
    GraphNodeSet,
    LaplacianConfig,
    SpectralVarianceConfig,
    SpectralVarianceResult,
    build_graph_laplacian,
    compute_grouped_spectral_variance,
    eigendecompose_laplacian,
)
from radio_map_estimation.loader.dataset import PoolTestSplit
from radio_map_estimation.loader.loader import load_grid_info_and_maps
from radio_map_estimation.utils.dir_naming import build_trial_output_dir, freq_dir_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 毎回 unnormalized (絶対スケール) / normalized (固有値が[0,2]に収まる) の
# 両方を計算・保存し、Wの候補間・trial間で見比べられるようにする
_LAPLACIAN_CONFIGS: dict[str, LaplacianConfig] = {
    "unnormalized": LaplacianConfig(method="unnormalized"),
    "normalized": LaplacianConfig(method="normalized"),
}


# ------------------------------------------------------------------
# 設定
# ------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralDiagnosisConfig:
    """spectral_variance.yaml 全体の設定

    出力ディレクトリの再構築には2種類ある (混同しないこと):
        1. pred.npz の読み込み元 = run_gp.py が実際に実行したときの
           (pathloss_model, source_shadowing_model, source_kernel)
           例: shadowing_model="gp", kernel="gudmundson" (GPフィット実行時の実験)
        2. spectral_variance.npz/.png の保存先 = このスクリプト自身が
           グラフを構築したときの (pathloss_model, shadowing_model, edge_weight)
           例: shadowing_model="graph", edge_weight="gudmundson" (このスクリプトの実験)
    どちらも build_trial_output_dir() で組み立てるが、渡す値が異なるため
    別々のフィールドとして持つ (新たなフィットは一切行わない)
    """

    # 対象experiment (train_size/n_trialsは読み込み・保存の両方で共通)
    train_sizes: tuple[int, ...]
    n_trials: int
    pathloss_model: str
    # pred.npz の読み込み元 (run_gp.py 実行時の実験を特定する)
    source_shadowing_model: str
    source_kernel: str
    # spectral_variance.npz/.png の保存先 (このスクリプト自身の実験として区別する)
    shadowing_model: str
    edge_weight: str  # このスクリプトが使うエッジ重み関数の選択 (create_edge_weight_fn の match/case)
    # LOS幾何接続判定
    max_radius_m: float
    # Gudmundsonカーネルの初期値 (最適化はせず、この値をそのまま固定パラメータとして使う)
    gudmundson_sigma_2_init: float
    gudmundson_d_cor_init: float
    # スペクトルビニング
    spectral_n_bins: int
    spectral_max_lambda: float | None


def load_spectral_diagnosis_config(path: Path) -> SpectralDiagnosisConfig:
    cfg: DictConfig = OmegaConf.load(path)  # type: ignore[assignment]

    train_sizes = tuple(int(s) for s in cfg.train_size)
    if not train_sizes or any(s <= 0 for s in train_sizes):
        raise ValueError(f"Invalid train_size: {train_sizes}")

    return SpectralDiagnosisConfig(
        train_sizes=train_sizes,
        n_trials=int(cfg.n_trials),
        pathloss_model=str(cfg.pathloss_model),
        source_shadowing_model=str(cfg.source.shadowing_model),
        source_kernel=str(cfg.source.kernel),
        shadowing_model=str(cfg.shadowing_model),
        edge_weight=str(cfg.edge_weight),
        max_radius_m=float(cfg.los.max_radius_m),
        gudmundson_sigma_2_init=float(cfg.gudmundson.sigma_2_init),
        gudmundson_d_cor_init=float(cfg.gudmundson.d_cor_init),
        spectral_n_bins=int(cfg.spectral.n_bins),
        spectral_max_lambda=(
            float(cfg.spectral.max_lambda) if cfg.spectral.get("max_lambda") is not None else None
        ),
    )


# ------------------------------------------------------------------
# エッジ重み関数構築 (拡張可能)
# ------------------------------------------------------------------


def create_edge_weight_fn(cfg: SpectralDiagnosisConfig) -> EdgeWeightFunction:
    """config化された「妥当な初期値」を固定値としてエッジ重み関数に組み込む

    診断の目的は「Wの設計方向が、同じ固有値λ_kを共有するモード間のパワー分散を
    減らす方向に効いているか」を確認することであり、最適パラメータでの性能を
    測ることではない (最適化はしない). そのため run_gp.py の GP フィット結果
    (fit_results.json) には依存せず、config で明示的に指定した初期値のみを使う.
    match/case で候補を切り替えられるようにしてあり、今回は Gudmundson のみ
    実装するが、将来は方向・Fresnel特徴を使った拡張版など
    (edge_weight_design_proposal.md 参照) をここに追加していく
    """
    match cfg.edge_weight:
        case "gudmundson":
            return GudmundsonEdgeWeight(
                sigma_2_init=cfg.gudmundson_sigma_2_init,
                d_cor_init=cfg.gudmundson_d_cor_init,
            )
        case _:
            raise NotImplementedError(f"edge_weight={cfg.edge_weight!r} は未実装")


# ------------------------------------------------------------------
# pred.npz からのノード集合構築
# ------------------------------------------------------------------


def build_full_node_set(pred: np.lib.npyio.NpzFile) -> GraphNodeSet:
    """pred.npz (run_gp.pyの出力) から train + test_prod 全ノードの GraphNodeSet を組み立てる

    train_train相当   : 学習に使った train 残差 (pred["train_residuals"])
    heldout_heldout相当 : test_prod の真値残差 (pred["shadowing_gt_test"], 学習には一切使っていない)
    """
    coords = np.concatenate([pred["train_coords"], pred["test_coords"]], axis=0)
    residuals = np.concatenate([pred["train_residuals"], pred["shadowing_gt_test"]], axis=0)
    train_mask = np.concatenate(
        [
            np.ones(len(pred["train_coords"]), dtype=bool),
            np.zeros(len(pred["test_coords"]), dtype=bool),
        ]
    )
    return GraphNodeSet(coords=coords, residuals=residuals, train_mask=train_mask)


# ------------------------------------------------------------------
# 保存・可視化
# ------------------------------------------------------------------


def save_spectral_variance(
    out_dir: Path,
    results_by_method: dict[str, dict[str, SpectralVarianceResult]],
) -> None:
    """method (unnormalized/normalized) x group (train/heldout) の結果を1つのnpzにまとめて保存する"""
    payload: dict[str, np.ndarray] = {}
    for method, results in results_by_method.items():
        for group, res in results.items():
            prefix = f"{method}__{group}"
            payload[f"{prefix}__bin_centers"] = res.bin_centers
            payload[f"{prefix}__mean_power"] = res.mean_power
            payload[f"{prefix}__std_power"] = res.std_power
            payload[f"{prefix}__counts"] = res.counts
    np.savez(out_dir / "spectral_variance.npz", **payload)  # type: ignore


def plot_spectral_variance(
    out_dir: Path,
    results_by_method: dict[str, dict[str, SpectralVarianceResult]],
) -> None:
    """unnormalized / normalized を横に並べ、それぞれ train vs heldout を重ね描きする

    (帯 = 平均 ± 標準偏差). normalizedはWの絶対スケールに依存せず固有値が[0,2]に
    収まるため、unnormalizedと横並びにすることでスケールの影響を切り分けて比較できる
    """
    methods = list(results_by_method.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4), squeeze=False)
    for ax, method in zip(axes[0], methods, strict=False):
        for group, res in results_by_method[method].items():
            ax.plot(res.bin_centers, res.mean_power, marker="o", label=group)
            lower = res.mean_power - res.std_power
            upper = res.mean_power + res.std_power
            ax.fill_between(res.bin_centers, lower, upper, alpha=0.2)
        ax.set_xlabel(r"Graph Laplacian eigenvalue $\lambda$")
        ax.set_ylabel(r"Power $\bar{P}(\lambda)$")
        ax.set_title(method)
        ax.legend()
    fig.suptitle("Spectral power variance: train vs heldout")
    fig.tight_layout()
    fig.savefig(out_dir / "spectral_variance.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# エントリポイント
# ------------------------------------------------------------------


def main(
    plateau_config_path: Path,
    sionna_config_path: Path,
    spectral_config_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]

    plateau_cfg: DictConfig = OmegaConf.load(plateau_config_path)  # type: ignore[assignment]
    sionna_cfg: DictConfig = OmegaConf.load(sionna_config_path)  # type: ignore[assignment]
    cfg = load_spectral_diagnosis_config(spectral_config_path)

    los_cfg = LosAdjacencyConfig(max_radius_m=cfg.max_radius_m)
    spectral_cfg = SpectralVarianceConfig(n_bins=cfg.spectral_n_bins, max_lambda=cfg.spectral_max_lambda)

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
                arrays = load_grid_info_and_maps(bldgmap_data, radiomap_data)
                grid_info = arrays.grid_info

                # test_prod のサイズはディレクトリ名の再構築にのみ使う (再サンプリングはしない)
                split = PoolTestSplit.load(split_path)
                n_test_prod = len(split.test_coords)

                for train_size in cfg.train_sizes:
                    for trial_idx in range(cfg.n_trials):
                        # pred.npz の読み込み元: run_gp.py が実際に実行したときのディレクトリ
                        source_dir = build_trial_output_dir(
                            root,
                            area.city_dir,
                            mesh_code,
                            freq_ghz,
                            train_size,
                            n_test_prod,
                            trial_idx,
                            cfg.pathloss_model,
                            cfg.source_shadowing_model,
                            cfg.source_kernel,
                        )
                        pred_path = source_dir / "pred.npz"

                        # fail loudly: run_gp.py がまだ実行されていない組み合わせは明示的にスキップ扱いにする
                        if not pred_path.exists():
                            logger.warning("skip (run_gp.py の出力が見つかりません): %s", pred_path)
                            continue

                        pred = np.load(pred_path)

                        # --- LOS/NLOS幾何接続判定 + 孤立ノード除去 ---
                        node_set_full = build_full_node_set(pred)
                        adjacency_full = compute_los_adjacency(node_set_full.coords, grid_info, los_cfg)
                        survive_mask = filter_isolated_nodes(adjacency_full)
                        n_isolated = int(np.count_nonzero(~survive_mask))
                        if n_isolated > 0:
                            logger.info("%s: %d isolated node(s) removed", source_dir, n_isolated)
                        node_set = node_set_full.filter(survive_mask)
                        adjacency = adjacency_full[np.ix_(survive_mask, survive_mask)]

                        # --- エッジ重み適用 (最適化なし、config化された初期値を固定して使う) ---
                        edge_weight_fn = create_edge_weight_fn(cfg)
                        w_dense = edge_weight_fn.compute_weights(node_set.coords)
                        w = np.where(adjacency, w_dense, 0.0)

                        # --- グラフラプラシアン・固有分解・GFTパワースペクトル (unnormalized/normalized 両方) ---
                        results_by_method: dict[str, dict[str, SpectralVarianceResult]] = {}
                        for method, laplacian_cfg in _LAPLACIAN_CONFIGS.items():
                            laplacian = build_graph_laplacian(w, laplacian_cfg)
                            eigvals, eigvecs = eigendecompose_laplacian(laplacian)
                            results_by_method[method] = compute_grouped_spectral_variance(
                                eigvals, eigvecs, node_set, spectral_cfg
                            )

                        # --- 保存先: このスクリプト自身の実験として source_dir とは別に構築する ---
                        out_dir = build_trial_output_dir(
                            root,
                            area.city_dir,
                            mesh_code,
                            freq_ghz,
                            train_size,
                            n_test_prod,
                            trial_idx,
                            cfg.pathloss_model,
                            cfg.shadowing_model,
                            cfg.edge_weight,
                        )
                        out_dir.mkdir(parents=True, exist_ok=True)
                        save_spectral_variance(out_dir, results_by_method)
                        plot_spectral_variance(out_dir, results_by_method)
                        logger.info("saved: %s/spectral_variance.", out_dir)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
