# ruff: noqa: F722
"""
パスロスモデルの抽象基底クラスとフィット結果データクラス
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int
from numpy import ndarray
from numpy.random import Generator

from ..loader.dataset import GridInfo

_SPEED_OF_LIGHT_M_S: float = 299_792_458.0  # 光速 [m/s]


@dataclass(frozen=True, slots=True)
class FitResult:
    """モデルフィット結果

    Attributes
    ----------
    model_name  : モデル識別子 ("ci", "abg", "ffnn", "ffnn_los")
    params      : フィット後の内部パラメータ名→値の辞書
    norm_stats  : 正規化統計量 (min/max) . FFNN系のみ実質使用、他は空辞書
    n_samples   : フィットに使用したサンプル数
    rmse_db     : フィット時のRMSE [dB] (訓練誤差の健全性確認用)
    """

    model_name: str
    params: dict[str, float | list[int]]
    norm_stats: dict[str, float]
    n_samples: int
    rmse_db: float

    def formatted_params(self) -> dict[str, str]:
        """params を表示用に文字列化した辞書を返す (ログ出力などで使用)

        float は "%.4g" 形式、list[int] は "[a, b, ...]" 形式で表示する.
        """
        result: dict[str, str] = {}
        for k, v in self.params.items():
            if isinstance(v, list):
                result[k] = f"[{', '.join(str(x) for x in v)}]"
            else:
                result[k] = f"{v:.4g}"
        return result

    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.formatted_params().items())
        norm_str = ", ".join(f"{k}={v:.4g}" for k, v in self.norm_stats.items())
        return (
            f"FitResult(model={self.model_name}, "
            f"params=[{params_str}], "
            f"norm_stats=[{norm_str}], "
            f"n_samples={self.n_samples}, "
            f"rmse={self.rmse_db:.3f}dB)"
        )


class PathLossModel(ABC):
    """パスロスモデルの抽象基底クラス

    すべてのパスロスモデルはこのインターフェースを実装する.
    入出力はすべて numpy 配列 (shape は jaxtyping 記法で明示) .

    bldg_mask は predict_mean() にも渡される. 建物マスクを使わないモデル
     (CI, ABG, FFNN 等) は単に無視すればよい.
    """

    @abstractmethod
    def fit(
        self,
        coords: Float[ndarray, "N 2"],
        tx_coords: Float[ndarray, "N 3"],
        rx_height_m: Float[ndarray, "N 1"],
        freq_hz: Float[ndarray, "N 1"],
        tx_power_dbm: Float[ndarray, "N 1"],
        rss_dbm_obs: Float[ndarray, "N 1"],
        grid_info: GridInfo,
        rng: Generator,
    ) -> FitResult:
        """観測データからモデルパラメータをフィットする

        Parameters
        ----------
        coords       : セル左下端座標 (x, y) [m]
        tx_coords    : 接続TX座標 (x, y, z) [m]
        rx_height_m  : 受信機高さ [m]
        freq_hz      : 搬送波周波数 [Hz]
        tx_power_dbm : 送信電力 [dBm]
        rss_dbm_obs  : RSS観測値 [dBm] (ノイズ付加済み)
        grid_info    : グリッド全体の静的情報
        rng          : 乱数生成器 (再現性のため外部から受け取る)

        Returns
        -------
        FitResult : フィット結果 (パラメータ・誤差・サンプル数)
        """
        ...

    @abstractmethod
    def predict_mean(
        self,
        coords: Float[ndarray, "M 2"],
        tx_coords: Float[ndarray, "M 3"],
        rx_height_m: Float[ndarray, "M 1"],
        freq_hz: Float[ndarray, "M 1"],
        tx_power_dbm: Float[ndarray, "M 1"],
        grid_info: Bool[ndarray, "H W"],
    ) -> Float[ndarray, "M 1"]:
        """パスロスモデルによる平均RSS [dBm] を返す (不確かさなし)

        決定論的モデル (CI, ABG) はこの値のみを持つ.
        確率的モデル (GP等) では predict_mean は平均関数 m(x) に相当する.

        Parameters
        ----------
        coords       : セル左下端座標 (x, y) [m]
        tx_coords    : 接続TX座標 (x, y, z) [m]
        rx_height_m  : 受信機高さ [m]
        freq_hz      : 搬送波周波数 [Hz]
        tx_power_dbm : 送信電力 [dBm]
        grid_info    : グリッド全体の静的情報

        Returns
        -------
        rss_dbm_mean : 平均予測RSS [dBm]、shape (M, 1)
        """
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, float | list[int]]:
        """現在の内部パラメータを返す (フィット後に有効)"""
        ...

    # ------------------------------------------------------------------
    # 共通ユーティリティ
    # ------------------------------------------------------------------

    @staticmethod
    def compute_3d_distance(
        coords: Float[ndarray, "N 2"],
        tx_coords: Float[ndarray, "N 3"],
        rx_height_m: Float[ndarray, "N 1"],
    ) -> Float[ndarray, "N 1"]:
        """3次元距離 d [m] を計算する

        受信機位置は (coords[:, 0], coords[:, 1], rx_height_m) とする.
        d = || rx_pos - tx_pos ||_2

        Parameters
        ----------
        coords      : (N,2) セル座標 (x, y) [m]
        tx_coords   : (N,3) TX座標 (x, y, z) [m]
        rx_height_m : (N,1) 受信機高さ [m]

        Returns
        -------
        d : (N,1) 3次元距離 [m]、d >= 1e-3 にクリップ
        """
        rx_xyz = np.hstack([coords, rx_height_m])  # (N,3)
        d = np.linalg.norm(rx_xyz - tx_coords, axis=1, keepdims=True)  # (N,1)
        return np.clip(d, 1e-3, None)

    @staticmethod
    def compute_azimuth(
        coords: Float[ndarray, "N 2"],
        tx_coords: Float[ndarray, "N 3"],
    ) -> Float[ndarray, "N 1"]:
        """TX→RX 水平角 azimuth [rad] を計算する

        azimuth = atan2(dy, dx)、範囲 (-π, π]

        Parameters
        ----------
        coords    : (N,2) セル座標 (x, y) [m]
        tx_coords : (N,3) TX座標 (x, y, z) [m]

        Returns
        -------
        azimuth : (N,1) 水平角 [rad]
        """
        dx = coords[:, 0:1] - tx_coords[:, 0:1]  # (N,1)
        dy = coords[:, 1:2] - tx_coords[:, 1:2]  # (N,1)
        return np.arctan2(dy, dx)

    @staticmethod
    def compute_ray_crossing_count(
        coords: Float[ndarray, "N 2"],
        tx_coords: Float[ndarray, "N 3"],
        grid_info: GridInfo,
    ) -> Float[ndarray, "N 1"]:
        """RX-TX間のレイが建物マスクを横切った回数 (棟数の近似) を計算する

        RX・TX の物理座標→インデックス変換は grid_info.coord_to_bldg_index に委譲する
         (margin オフセット・範囲外クリップの実体は grid_transform.py に一元化済み) .

        Bresenham ラインアルゴリズムで RX-TX 間のセルを1ピクセルずつ厳密に辿り、
        「建物なし→あり」の遷移回数 (連続する建物領域に何回入ったか) を数える.
        厚い建物1棟を貫通する場合も、薄い建物を複数貫通する場合も区別できる
         (後者の方が値が大きくなる) . ピクセル単位で厳密に辿るため、
        n_ray_samples のような恣意的なサンプリング密度パラメータに依存しない.

        Parameters
        ----------
        coords    : (N,2) RXセル座標 (x, y) [m]
        tx_coords : (N,3) TX座標 (x, y, z) [m]
        grid_info : グリッド全体の静的情報

        Returns
        -------
        crossing_count : (N,1) レイが建物領域に進入した回数
        """
        n = coords.shape[0]

        rx_idx = grid_info.coord_to_bldg_index(coords[:, :2])  # (N,2) row,col
        tx_idx = grid_info.coord_to_bldg_index(tx_coords[:, :2])  # (N,2) row,col

        crossing_count = np.zeros((n, 1), dtype=np.float64)
        for i in range(n):
            rows, cols = PathLossModel._bresenham_line(rx_idx[i, 0], rx_idx[i, 1], tx_idx[i, 0], tx_idx[i, 1])
            hits = grid_info.bldg_mask[rows, cols]  # (line_len,) bool

            prev = np.concatenate([[False], hits[:-1]])
            crossing_count[i, 0] = np.count_nonzero(hits & ~prev)

        return crossing_count  # (N,1)

    @staticmethod
    def _bresenham_line(
        r0: int, c0: int, r1: int, c1: int
    ) -> tuple[Int[ndarray, "L 1"], Int[ndarray, "L 1"]]:
        """Bresenham ラインアルゴリズムで (r0,c0)-(r1,c1) 間の格子点を列挙する

        Returns
        -------
        rows, cols : ライン上の行・列インデックス (両端含む、重複なし)
        """
        rows: list[int] = []
        cols: list[int] = []

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc

        r, c = r0, c0
        while True:
            rows.append(r)
            cols.append(c)
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

        return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)

    @staticmethod
    def compute_bldg_count_in_fresnel_ellipse(
        coords: Float[ndarray, "N 2"],
        tx_coords: Float[ndarray, "N 3"],
        grid_info: GridInfo,
        freq_hz: Float[ndarray, "N 1"],
        fresnel_zone_order: int = 1,
    ) -> Float[ndarray, "N 1"]:
        """RX-TXを焦点とする第nフレネルゾーン楕円内の建物ピクセル数を計算する

        bbox の四隅、および bbox 内グリッド点の物理座標復元は、いずれも
        grid_info.coord_to_bldg_index / grid_info.bldg_index_to_coord に委譲する
         (margin オフセットの実体は grid_transform.py に一元化済み) .
        bbox の四隅は floor ベースのセル包含判定 (RSS の point_to_cell_index と同じ)
        でインデックス化される.

        画素の代表点は build_bldg_mask と同じ「セルの左下端点」を用いる
         (セル中心ではない) .

        Parameters
        ----------
        coords             : (N,2) RXセル座標 (x, y) [m]
        tx_coords          : (N,3) TX座標 (x, y, z) [m]
        grid_info          : グリッド全体の静的情報
        freq_hz             : (N,1) サンプルごとの搬送波周波数 [Hz]
        fresnel_zone_order : フレネルゾーン次数 n (デフォルト第1ゾーン)

        Returns
        -------
        bldg_count : (N,1) 楕円内の建物ピクセル数 (非負整数の float 表現)
        """
        bldg_mask = grid_info.bldg_mask
        n_samples = coords.shape[0]

        rx_xy = coords[:, :2]
        tx_xy = tx_coords[:, :2]
        wavelength_m = _SPEED_OF_LIGHT_M_S / freq_hz[:, 0]

        diff = tx_xy - rx_xy
        d = np.linalg.norm(diff, axis=1)
        d_safe = np.where(d > 0.0, d, 1.0)
        e_u = diff / d_safe[:, None]
        e_v = np.stack([-e_u[:, 1], e_u[:, 0]], axis=1)

        c = d / 2.0
        b = 0.5 * np.sqrt(fresnel_zone_order * wavelength_m * d_safe)
        a = np.sqrt(b**2 + c**2)
        center = (rx_xy + tx_xy) / 2.0

        half_extent_x = np.sqrt((a * e_u[:, 0]) ** 2 + (b * e_v[:, 0]) ** 2)
        half_extent_y = np.sqrt((a * e_u[:, 1]) ** 2 + (b * e_v[:, 1]) ** 2)

        # bbox 四隅の物理座標 → grid_info 経由でインデックス化 (snap + offset + clip)
        corner_min = np.stack([center[:, 0] - half_extent_x, center[:, 1] - half_extent_y], axis=1)  # (N,2)
        corner_max = np.stack([center[:, 0] + half_extent_x, center[:, 1] + half_extent_y], axis=1)  # (N,2)
        idx_min = grid_info.coord_to_bldg_index(corner_min)  # (N,2) row,col
        idx_max = grid_info.coord_to_bldg_index(corner_max)  # (N,2) row,col

        row_min, col_min = idx_min[:, 0], idx_min[:, 1]
        row_max, col_max = idx_max[:, 0], idx_max[:, 1]

        bldg_count = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            if d[i] <= 0.0:
                continue

            cols = np.arange(col_min[i], col_max[i] + 1)
            rows = np.arange(row_min[i], row_max[i] + 1)
            if cols.size == 0 or rows.size == 0:
                continue

            row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")  # (Hi,Wi)
            pts = grid_info.bldg_index_to_coord(row_grid.ravel(), col_grid.ravel())  # (Hi*Wi, 2)
            grid_x = pts[:, 0].reshape(row_grid.shape)
            grid_y = pts[:, 1].reshape(row_grid.shape)

            rel_x = grid_x - center[i, 0]
            rel_y = grid_y - center[i, 1]
            u = rel_x * e_u[i, 0] + rel_y * e_u[i, 1]
            v = rel_x * e_v[i, 0] + rel_y * e_v[i, 1]

            inside_ellipse = (u / a[i]) ** 2 + (v / b[i]) ** 2 <= 1.0
            bldg_patch = bldg_mask[row_min[i] : row_max[i] + 1, col_min[i] : col_max[i] + 1]

            bldg_count[i] = np.count_nonzero(inside_ellipse & bldg_patch)

        return bldg_count.reshape(n_samples, 1)
