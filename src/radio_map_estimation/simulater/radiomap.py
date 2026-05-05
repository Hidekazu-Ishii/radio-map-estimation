"""
目的: osm_buildings.py の出力 (BuildingData) を受け取り,
      Sionna RT でray tracingして電波マップ (path_gain) を生成する.

パイプライン:
    BuildingData (建物ポリゴン + height_m + AreaSpec)
        ↓ build_building_meshes()
    PLY メッシュファイル群 (meshes/buildings.ply, meshes/ground.ply)
        ↓ write_mitsuba_xml()
    Mitsuba XML シーンファイル (scene.xml)
        ↓ run_radio_map()
    電波マップ ndarray (grid_size, grid_size) [dB]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from numpy.random import Generator
from shapely.geometry import MultiPolygon, Polygon

from .osm_buildings import BuildingData

Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]


# ---------------------------------------------------------------------------
# Step 1: 建物ポリゴン → 3D メッシュ (trimesh)
# ---------------------------------------------------------------------------


def extrude_polygon_to_mesh(poly: Polygon, height: float) -> trimesh.Trimesh:
    """
    2D ポリゴンを高さ height [m] で押し出した閉じた3Dメッシュを返す.

    Parameters
    ----------
    poly : Polygon
        建物フットプリント (ローカル座標系).
    height : float
        建物高さ [m].

    Returns
    -------
    trimesh.Trimesh
        押し出されたソリッドメッシュ.
    """
    return trimesh.creation.extrude_polygon(poly, height, engine="earcut")


def build_building_meshes(
    building_data: BuildingData,
    min_height_m: float = 3.0,
) -> list[trimesh.Trimesh]:
    """BuildingData からローカル座標系の建物メッシュを生成する."""
    gdf_local = building_data.to_local_gdf()

    meshes: list[trimesh.Trimesh] = []
    for _, row in gdf_local.iterrows():
        geom = row.geometry
        height = float(row["height_m"])
        if height < min_height_m:
            continue
        polys: list[Polygon] = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        for poly in polys:
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
            try:
                meshes.append(extrude_polygon_to_mesh(poly, height))
            except Exception as e:
                print(f"[warn] skipped polygon: {e}")  # 原因を表示
                continue

    if not meshes:
        raise RuntimeError("No valid building meshes were generated.")

    return meshes


def build_ground_mesh(building_data: BuildingData) -> trimesh.Trimesh:
    """BuildingData から地面メッシュ (z=0 平面) を生成する."""
    s = building_data.area_spec.area_size_m
    vertices: np.ndarray = np.array(
        [[0.0, 0.0, 0.0], [s, 0.0, 0.0], [s, s, 0.0], [0.0, s, 0.0]],
        dtype=np.float64,
    )
    faces: np.ndarray = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# ---------------------------------------------------------------------------
# Step 2: PLY ファイルの書き出し
# ---------------------------------------------------------------------------


def save_meshes_to_ply(
    building_meshes: list[trimesh.Trimesh],
    ground_mesh: trimesh.Trimesh,
    mesh_dir: Path,
) -> tuple[list[Path], Path]:
    mesh_dir.mkdir(parents=True, exist_ok=True)
    building_plys: list[Path] = []
    for i, mesh in enumerate(building_meshes):
        ply_path = mesh_dir / f"building_{i:05d}.ply"
        mesh.export(str(ply_path))
        building_plys.append(ply_path)
    ground_ply = mesh_dir / "ground.ply"
    ground_mesh.export(str(ground_ply))
    print(f"[mesh] {len(building_plys)} buildings → {mesh_dir}")
    return building_plys, ground_ply


# ---------------------------------------------------------------------------
# Step 3: Mitsuba XML シーンファイルの生成
# ---------------------------------------------------------------------------


def write_mitsuba_xml(
    scene_dir: Path,
    building_plys: list[Path],  # list に変更
    ground_ply: Path,
    building_material: str = "concrete",
    ground_material: str = "concrete",
) -> Path:
    shapes_xml = ""
    for i, ply in enumerate(building_plys):
        rel = ply.relative_to(scene_dir)
        shapes_xml += f"""
    <shape type="ply" id="building_{i:05d}">
        <string name="filename" value="{rel}"/>
        <ref id="mat-itu-buildings"/>
    </shape>
"""
    ground_rel = ground_ply.relative_to(scene_dir)
    xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<scene version="3.0.0">

    <bsdf type="itu-radio-material" id="mat-itu-buildings">
        <string name="type" value="{building_material}"/>
    </bsdf>

    <bsdf type="itu-radio-material" id="mat-itu-ground">
        <string name="type" value="{ground_material}"/>
    </bsdf>
{shapes_xml}
    <shape type="ply" id="ground">
        <string name="filename" value="{ground_rel}"/>
        <ref id="mat-itu-ground"/>
    </shape>

</scene>
"""
    scene_xml = scene_dir / "scene.xml"
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_xml.write_text(xml_content, encoding="utf-8")
    print(f"[xml] scene → {scene_xml}")
    return scene_xml


# ---------------------------------------------------------------------------
# Step 4: TX 配置 (PPP)
# ---------------------------------------------------------------------------


def place_tx_ppp(
    rng: Generator,
    area_size_m: float,
    intensity: float,
) -> np.ndarray:
    """
    一様な2次元ポアソン点過程 (PPP) で送信機を配置する.

    配置数も位置もランダム. 最低1局は保証する.

    Parameters
    ----------
    rng : Generator
        呼び出し元から受け渡す乱数生成器.
    area_size_m : float
        エリア一辺 [m].
    intensity : float
        送信機密度 [TX/m^2].

    Returns
    -------
    tx_locations : ndarray of shape (T, 2)
        送信機の (x, y) 座標 [m]. ローカル座標系.
    """
    expected_n = intensity * area_size_m**2
    n_tx = max(int(rng.poisson(expected_n)), 1)
    tx_locations: np.ndarray = rng.uniform(0.0, area_size_m, size=(n_tx, 2))
    return tx_locations


# ---------------------------------------------------------------------------
# Step 5: Sionna RT で電波マップを計算
# ---------------------------------------------------------------------------


def run_radio_map(
    scene_xml: Path,
    tx_position_local: Vec3,
    frequency_hz: float,
    tx_power_dbm: float,
    building_data: BuildingData,
    rx_height_m: float,
    max_depth: int,
    cell_size_m: float,
    samples_per_tx: int,
) -> np.ndarray:
    """
    Sionna RT で RSS マップ [dBm] を計算して返す.

    Parameters
    ----------
    scene_xml : Path
        Mitsuba XML シーンファイルのパス.
    tx_position_local : Vec3
        ローカル座標系での送信機位置 (x, y, z) [m].
        bbox_m の左下を原点とする.
    frequency_hz : float
        搬送波周波数 [Hz].
    tx_power_dbm : float
        送信電力 [dBm].
    building_data : BuildingData
        AreaSpec から cell_size_m / area_size_m を取得する.
    rx_height_m : float
        受信機高さ [m] (RadioMapSolver の測定面高さ).
    max_depth : int
        最大反射回数.
    samples_per_tx : int
        TX あたりのレイサンプル数.

    Returns
    -------
    rss_dbm : np.ndarray of shape (H, W)
        受信電力 [dBm]. [row, col] = [y_idx, x_idx], [0,0] は左下.
    """
    # Sionna はインポート時に GPU/CPU を自動選択するため関数内でインポート.
    # 型チェック時は TYPE_CHECKING ブロックの mitsuba を参照する.
    import mitsuba as mi
    from sionna.rt import (
        PlanarArray,
        RadioMapSolver,
        Transmitter,
        load_scene,
    )

    spec = building_data.area_spec
    half: float = spec.area_size_m / 2.0

    # --- シーン読み込み ---
    scene = load_scene(
        str(scene_xml),
        remove_duplicate_vertices=False,
    )
    scene.frequency = frequency_hz

    # --- アンテナアレイ (等方性単素子) ---
    # PlanarArray.__init__ は Python int/float を受け付け,
    # 内部で mi.UInt / mi.Float に変換する.
    iso_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.tx_array = iso_array
    scene.rx_array = iso_array

    # --- 送信機 ---
    tx = Transmitter(
        name="tx",
        position=mi.Point3f(list(tx_position_local)),
        orientation=mi.Point3f([0.0, 0.0, 0.0]),
        power_dbm=int(tx_power_dbm),
    )
    scene.add(tx)

    # --- RadioMapSolver ---
    rm_solver = RadioMapSolver()
    radio_map = rm_solver(
        scene=scene,
        max_depth=max_depth,
        cell_size=mi.Point2f([cell_size_m, cell_size_m]),
        samples_per_tx=samples_per_tx,
        center=mi.Point3f([half, half, rx_height_m]),
        orientation=mi.Point3f([0.0, 0.0, 0.0]),
        size=mi.Point2f([spec.area_size_m, spec.area_size_m]),
        diffraction=True,
    )

    rss_np = radio_map.rss.numpy()  # (num_tx=1, H, W) [W]
    rss_w = rss_np[0]

    # 0.0 セル (sionna rt の仕様でレイ未到達点は0.0になる) を nan に置換してから dBm 変換
    rss_w_safe = np.where(rss_w > 0.0, rss_w, np.nan)
    # print(f"rss raw: min={np.nanmin(rss_w_safe):.3e}, max={np.nanmax(rss_w_safe):.3e}")
    with np.errstate(invalid="ignore"):
        rss_dbm = 10.0 * np.log10(rss_w_safe) + 30.0  # W → dBm
    return rss_dbm


# ---------------------------------------------------------------------------
# Step 6: best-server マップの合成
# ---------------------------------------------------------------------------


def compute_best_server_map(
    rss_dbm_list: list[np.ndarray],  # path_gain_db_list → rss_dbm_list
) -> np.ndarray:
    """
    複数 TX の RSS [dBm] マップから best-server マップを合成する.

    各グリッドセルで最大 RSS を持つ TX を選択する.

    Parameters
    ----------
    rss_dbm_list : list of ndarray, each shape (H, W)
        TX ごとの RSS マップ [dBm].

    Returns
    -------
    rss_best : ndarray of shape (H, W)
        best-server の RSS [dBm].
    """
    # 最大値を取る
    stacked: np.ndarray = np.stack(rss_dbm_list, axis=0)  # (T, H, W)
    rss_best: np.ndarray = np.nanmax(stacked, axis=0)  # (H, W)
    return rss_best


# ---------------------------------------------------------------------------
# Step 7: 電波マップの可視化
# ---------------------------------------------------------------------------


def plot_radio_map(
    rss_dbm: np.ndarray,
    building_mask: np.ndarray,
    area_size_m: float,
    tx_locations: np.ndarray,
    frequency_hz: float,
    save_path: Path,
) -> None:
    """
    RSS マップ [dBm] と建物マスク・全TX位置を重ねて可視化・保存する.

    Parameters
    ----------
    rss_dbm : ndarray of shape (H, W)
        RSS マップ [dBm].
    building_mask : ndarray of shape (H, W), dtype bool
        建物セルのマスク.
    area_size_m : float
        エリア一辺 [m].
    tx_locations : ndarray of shape (T, 2)
        TX の (x, y) 座標 [m].
    frequency_hz : float
        搬送波周波数 [Hz].
    save_path : Path
        保存先パス.
    """
    _, ax = plt.subplots(figsize=(7, 6))
    extent = (0.0, area_size_m, 0.0, area_size_m)

    rss_masked = np.ma.masked_invalid(rss_dbm)

    vmin = float(np.nanmin(rss_dbm))
    vmax = float(np.nanmax(rss_dbm))

    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(color=cmap(0.0))

    im = ax.imshow(
        rss_masked,
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
        zorder=1,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label="RSS [dBm]")

    # 建物マスクを完全不透明グレーでオーバーレイ (zorder=2 で RSS マップの上)
    overlay: np.ndarray = np.zeros((*building_mask.shape, 4), dtype=np.float32)
    overlay[building_mask] = [0.7, 0.7, 0.7, 1.0]
    ax.imshow(overlay, origin="lower", extent=extent, zorder=2)

    # 全 TX をプロット (zorder=3 で建物マスクの上)
    ax.scatter(
        tx_locations[:, 0],
        tx_locations[:, 1],
        c="red",
        s=120,
        marker="*",
        zorder=3,
        label=f"TX (n={len(tx_locations)})",
    )
    ax.legend(loc="upper right")

    freq_ghz = frequency_hz / 1e9
    ax.set_title(
        f"Radio Map (Best-Server RSS)\nf={freq_ghz:.2f} GHz, {area_size_m:.0f}m x {area_size_m:.0f}m"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] saved → {save_path}")
