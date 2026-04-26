"""
src/radiomap.py

目的: OSM建物マップ + 簡易電波伝播モデルで、指定エリアのラジオマップを生成して保存する.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import Generator


def load_building_map(
    building_map_path: Path,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Returns
    -------
    building_mask : (grid_size, grid_size) bool
        建物が存在するセルが True.
        インデックス [row, col] は [y_idx, x_idx] に対応し、
        [0, 0] は領域の左下 (ymin, xmin) を指す.
    building_heights : (grid_size, grid_size) float [m]
        セルの最大建物高さ [m]. 建物がない場合は 0.0.
    area_size_m : float
        エリアサイズ [m]
    grid_size : int
        グリッド一辺のセル数
    """
    data = np.load(building_map_path)
    building_mask = data["building_mask"].astype(bool)
    building_heights = data["building_heights"].astype(float)
    area_size_meter = float(data["area_size_m"])
    grid_size = int(data["grid_size"])
    return building_mask, building_heights, area_size_meter, grid_size


def place_bs_ppp(
    rng: Generator,
    area_size: float,
    intensity: float,
) -> np.ndarray:
    """
    一様な2次元ポアソン点過程 (PPP) を用いて基地局 (BS) を配置する.
    配置数も配置場所もランダムである.

    Parameters
    ----------
    rng : Generator
        乱数生成器
    area_size : float
        エリアサイズ [m]
    intensity : float
        基地局密度 [BS / m^2]

    Returns
    -------
    bs_locations : ndarray of shape (B, 2)
        基地局の座標 [x, y] [m]
    """
    expected_n = intensity * area_size**2
    n_bs = rng.poisson(expected_n)
    n_bs = max(n_bs, 1)  # 最低1局は配置する
    bs_locations = rng.uniform(0, area_size, size=(n_bs, 2))
    return bs_locations


def path_loss_uma_db(
    distances: np.ndarray,
    frequency_hz: float,
    is_los: np.ndarray,
    d_ref: float = 1.0,
) -> np.ndarray:
    """
    3GPP UMa LOS/NLOS モデルに基づくパスロス (dB) を計算する.

    LOS  : PL = 32.4 + 20*log10(f_GHz) + 21*log10(d)
    NLOS : PL = 32.4 + 20*log10(f_GHz) + 40*log10(d)

    距離依存性の違い:
        LOS  : 21 dB/decade  ≈ フリー空間 (20) に近い. 道路上の見通し伝搬.
        NLOS : 40 dB/decade  建物による回折・反射で急激に減衰.

    LOS/NLOSの差 (d=200m, f=3.5GHz の例):
        LOS  : 32.4 + 10.9 + 46.0 = 89.3 dB
        NLOS : 32.4 + 10.9 + 87.6 = 130.9 dB
        差   : 41.6 dB  ← 道路構造がRSSに明確に現れる

    Parameters
    ----------
    distances : ndarray (P,) [m]
    frequency_hz : float [Hz]
    is_los : ndarray (P,) bool
        True = LOS (見通し), False = NLOS (非見通し)
    d_ref : float [m]
        距離の下限値 (ゼロ除算防止)

    Returns
    -------
    pathloss_db : ndarray (P,) [dB]
    """
    d = np.maximum(distances, d_ref)
    f_ghz = frequency_hz / 1e9
    log_f = 20.0 * np.log10(f_ghz)
    log_d = np.log10(d)

    pl_los = 32.4 + log_f + 21.0 * log_d
    pl_nlos = 32.4 + log_f + 40.0 * log_d

    return np.where(is_los, pl_los, pl_nlos)


def generate_shadowing_field(
    rng: Generator,
    grid_size: int,
    area_size: float,
    sigma_db: float,
    corr_distance: float,
) -> np.ndarray:
    """
    FFTを用いてO(P log P)で空間相関シャドウイング場を近似生成する.

    共分散モデル:
        C(r) = sigma^2 * exp(-r / corr_distance)

    手順:
        1. グリッド上で指数型カーネルを評価しFFT
        2. 白色雑音をスペクトル領域でフィルタリング (スペクトル成形)
        3. 逆FFT後にsigma_dbへ正規化

    Parameters
    ----------
    rng : Generator
        乱数生成器
    grid_size : int
        グリッド一辺のセル数
    area_size : float
        エリアサイズ [m]
    sigma_db : float
        シャドウイング標準偏差 [dB]
    corr_distance : float
        非相関距離 [m]

    Returns
    -------
    shadowing : ndarray of shape (P,)  P = grid_size^2
        シャドウイング値 [dB]
    """
    dx = area_size / grid_size

    # グリッド上で指数型カーネルを評価 (周期境界を仮定)
    ix = np.fft.fftfreq(grid_size, d=1.0) * grid_size
    iy = np.fft.fftfreq(grid_size, d=1.0) * grid_size
    rx, ry = np.meshgrid(ix * dx, iy * dx)
    r = np.sqrt(rx**2 + ry**2)
    kernel = np.exp(-r / corr_distance)

    # カーネルのFFT → フィルタの振幅スペクトル
    # 数値誤差で虚部や微小負値が出るため実部を取りクリップ
    kernel_fft_sqrt = np.sqrt(np.maximum(np.fft.fft2(kernel).real, 0.0))

    # 白色雑音をスペクトル領域でフィルタリング → 空間相関を付与
    white = rng.standard_normal((grid_size, grid_size))
    filtered = np.fft.ifft2(np.fft.fft2(white) * kernel_fft_sqrt).real

    # sigma_db に正規化
    std = filtered.std()
    if std > 0:
        filtered = filtered / std * sigma_db

    return filtered.ravel()  # (P,)


def compute_los_flag(
    rng: Generator,
    bs_xy: np.ndarray,
    bs_z: float,
    grid_coords: np.ndarray,
    ue_z: float,
    building_heights: np.ndarray,
    area_size: float,
    n_sample: int,
) -> np.ndarray:
    """
    LOS/NLOS判定: 3D幾何判定 x 3GPP TR 38.901 UMa確率モデルの組み合わせ.

    判定ロジック:
        Step 1: 3D幾何判定
            BS-UE間を結ぶ3次元直線上に n_sample 点をサンプリングし、
            「建物高さ > LOS経路の高さ」となるセルが1つでもあれば NLOS とする.
            n_sample の目安: √2 * grid_size (セル対角線上に十分なサンプル数)

        Step 2: 3GPP TR 38.901 Table 7.4.2-1 UMa LOS確率モデル
            幾何学的にLOSでも、建物の細かな凹凸・看板・植栽・車両等による
            回折・散乱で実効的にNLOSになる場合がある.
            P_LOS(d2D) = min(18/d, 1) * (1 - exp(-d/63)) + exp(-d/63)
            で距離依存のLOS確率を計算し、Bernoulliサンプリングで判定する.
            距離依存の特性:
                d=  10m : P_LOS ≈ 1.000  (近距離はほぼLOS)
                d= 100m : P_LOS ≈ 0.348
                d= 500m : P_LOS ≈ 0.036  (遠距離はほぼNLOS)

        最終判定: 幾何学的NLOSは確率モデルに関わらずNLOS.
    Parameters
    ----------
    bs_xy : (2,)
        BS水平座標 [m]
    bs_z : float
        BS高さ [m]
    grid_coords : (P, 2)
        グリッド点水平座標 [m]
    ue_z : float
        UE高さ [m]
    building_heights : (grid_size, grid_size)
        建物高さグリッド [m]
    area_size : float
        エリアサイズ [m]
    n_sample : int
        LOS経路上のサンプル数

    Returns
    -------
    is_los : (P,) bool
        True = LOS, False = NLOS
    """
    # Step 1: 3D幾何判定
    grid_size = building_heights.shape[0]
    dx = area_size / grid_size
    ts = np.linspace(0, 1, n_sample)  # (S,)

    path_xy = bs_xy[None, None, :] + ts[None, :, None] * (grid_coords[:, None, :] - bs_xy[None, None, :])
    path_z = bs_z + ts[None, :] * (ue_z - bs_z)

    ixs = np.clip((path_xy[..., 0] / dx).astype(int), 0, grid_size - 1)
    iys = np.clip((path_xy[..., 1] / dx).astype(int), 0, grid_size - 1)

    cell_heights = building_heights[iys, ixs]  # (P, S)
    blocked = cell_heights > path_z  # (P, S)
    geom_los = ~blocked.any(axis=1)  # (P,)

    # Step 2: 3GPP TR 38.901 UMa LOS確率モデル (2D距離で評価)
    diff = grid_coords - bs_xy[None, :]
    d2d = np.maximum(np.sqrt((diff**2).sum(axis=-1)), 1.0)  # (P,)
    p_los = np.minimum(18.0 / d2d, 1.0) * (1.0 - np.exp(-d2d / 63.0)) + np.exp(-d2d / 63.0)
    stochastic_los = rng.uniform(0.0, 1.0, size=len(grid_coords)) < p_los  # (P,)

    return geom_los & stochastic_los  # (P,)


def generate_rss_map(
    rng: Generator,
    bs_xy: np.ndarray,
    bs_z: float,
    grid_coords: np.ndarray,
    ue_z: float,
    shadowing: np.ndarray,
    building_heights: np.ndarray,
    area_size: float,
    tx_power_dbm: float,
    frequency_hz: float,
    n_sample: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1つの基地局 (BS) から各グリッド点における RSS [dBm] を計算する.

    RSS = P_tx - PL_LOS/NLOS(d) - shadowing

    LOS/NLOSはBS-UE間の3D経路上の建物遮蔽で判定する.
    building_loss_db は廃止し、パスロスの指数差 (21 vs 40) で遮蔽を表現する.

    Parameters
    ----------
    bs_xy : (2,)
        BS水平座標 [m]
    bs_z : float
        BS高さ [m]
    grid_coords : (P, 2)
        グリッド点水平座標 [m]
    ue_z : float
        UE高さ [m]
    shadowing : (P,)
        シャドウイング値 [dB]
    building_heights : (grid_size, grid_size)
        建物高さグリッド [m]
    area_size : float
        エリアサイズ [m]
    tx_power_dbm : float
        送信電力 [dBm]
    frequency_hz : float
        周波数 [Hz]
    n_sample : int
        LOS判定サンプル数

    Returns
    -------
    rss : (P,) float [dBm]
        各グリッド点における RSS [dBm]
    is_los : (P,) bool
        LOS判定結果 (保存・可視化用)
    """
    diff = grid_coords - bs_xy[None, :]
    distances = np.sqrt((diff**2).sum(axis=-1))

    is_los = compute_los_flag(rng, bs_xy, bs_z, grid_coords, ue_z, building_heights, area_size, n_sample)

    pl = path_loss_uma_db(distances, frequency_hz, is_los)
    rss = tx_power_dbm - pl - shadowing

    return rss, is_los


def plot_results(
    building_mask: np.ndarray,
    bs_locations: np.ndarray,
    rss_map: np.ndarray,
    is_los_best: np.ndarray,
    area_size: float,
    save_path: Path,
) -> None:
    grid_size = building_mask.shape[0]
    rss_grid = rss_map.reshape(grid_size, grid_size)
    los_grid = is_los_best.reshape(grid_size, grid_size)

    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 建物マップと基地局配置
    ax = axes[0]
    building_img = np.zeros((grid_size, grid_size, 3))
    building_img[building_mask] = [0.2, 0.2, 0.6]
    building_img[~building_mask] = [0.95, 0.95, 0.95]
    ax.imshow(building_img, origin="lower", extent=[0, area_size, 0, area_size])
    ax.scatter(bs_locations[:, 0], bs_locations[:, 1], c="red", marker="*", s=200, zorder=5, label="BS")
    ax.set_title("Building Map & BS Deployment")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()

    # LOS/NLOSマップ
    ax = axes[1]
    ax.imshow(
        los_grid.astype(float),
        origin="lower",
        extent=[0, area_size, 0, area_size],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    ax.scatter(bs_locations[:, 0], bs_locations[:, 1], c="blue", marker="*", s=200, zorder=5)
    ax.set_title("LOS Map (green=LOS, red=NLOS)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # RSSマップ
    ax = axes[2]
    im = ax.imshow(rss_grid, origin="lower", extent=[0, area_size, 0, area_size], cmap="jet")
    ax.scatter(bs_locations[:, 0], bs_locations[:, 1], c="white", marker="*", s=200, zorder=5)
    plt.colorbar(im, ax=ax, label="RSS [dBm]")
    ax.set_title("RSS Map (best server)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] saved → {save_path}")
