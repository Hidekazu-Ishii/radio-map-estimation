# type: ignore
# ruff: noqa: F821
"""
Sionna RT RadioMapSolver のフェージング除去パッチ (インコヒーレント電力和への差し替え)

Sionna RT の RadioMapSolver はデフォルトでコヒーレント和を計算する:

    P_coherent = |Σ_l (aw_l @ e_l)|²   ← 各パスのフィールドを先に加算

これは位相干渉 (瞬時フェージング) を含む instantaneous power である
本モジュールは add_paths をインコヒーレント和に差し替える:

    P_incoherent = Σ_l |aw_l @ e_l|²   ← 各パスの電力を個別に加算

これは shadowing のみを反映した平均電力 (= Σ_l |h_l|²) に相当し、
フェージングが完全に除去される

使い方 (RadioMapSolver より前に1回だけ呼ぶ) :
    from radio_map_estimation.sionna.incoherent_patch import apply_incoherent_patch
    apply_incoherent_patch()

注意:
    - uv sync / pip install で sionna-rt が再インストールされてもパッチは維持される
       (インストール済みファイルを変更しないため)
    - パッチ適用後は num_seeds ループは不要 (seed による位相ランダム化が無意味になる)
    - PlanarRadioMap と MeshRadioMap の両方に適用する
"""

from __future__ import annotations

import logging

import drjit as dr
import mitsuba as mi
from sionna.rt.utils import WedgeGeometry, wedge_interior_angle

logger = logging.getLogger(__name__)


def _planar_add_paths_incoherent(
    self,
    e_fields: List[mi.Vector4f],
    array_w: List[mi.Float],
    si: mi.SurfaceInteraction3f,
    k_world: mi.Vector3f,
    tx_indices: mi.UInt,
    active: mi.Bool,
    diffracted_paths: bool,
    solid_angle: mi.Float | None = None,
    tx_positions: mi.Point3f | None = None,
    wedges: WedgeGeometry | None = None,
    diff_point: mi.Point3f | None = None,
    wedges_samples_cnt: mi.UInt | None = None,
) -> None:
    """PlanarRadioMap.add_paths のインコヒーレント版

    変更箇所 (元実装 planar_radio_map.py L279-281) :
        # 変更前: コヒーレント和
        a = dr.zeros(mi.Vector4f, 1)
        for e_field, aw in zip(e_fields, array_w):
            a += aw @ e_field
        a = dr.squared_norm(a)                  # |Σ E_l|²

        # 変更後: インコヒーレント和
        a = mi.Float(0)
        for e_field, aw in zip(e_fields, array_w):
            a += dr.squared_norm(aw @ e_field)  # Σ|E_l|²
    """
    # セルインデックスとテンソルインデックス (元実装と同じ)
    cell_ind = self._local_to_cell_ind(si.uv)
    tensor_ind = tx_indices * self.cells_count + cell_ind

    # インコヒーレント和: 各パスの電力を個別に加算
    a = mi.Float(0)
    for e_field, aw in zip(e_fields, array_w, strict=False):
        a += dr.squared_norm(aw @ e_field)

    # レイ重み (元実装と同じ)
    if not diffracted_paths:
        k_local = si.to_local(k_world)
        cos_theta = dr.abs(k_local.z)
        w = solid_angle * dr.rcp(cos_theta)
    else:
        tx_positions_ = dr.gather(mi.Point3f, tx_positions, tx_indices, active=active)
        w = self._diffraction_integration_weight(wedges, tx_positions_, diff_point, k_world, si)
        w *= wedges.length * (dr.two_pi - wedge_interior_angle(wedges.n0, wedges.nn))
        w /= wedges_samples_cnt

    a *= w * self._normalization_factor

    dr.scatter_reduce(
        dr.ReduceOp.Add,
        self._pathgain_map.array,
        value=a,
        index=tensor_ind,
        active=active,
    )


def _mesh_add_paths_incoherent(
    self,
    e_fields: List[mi.Vector4f],
    array_w: List[mi.Float],
    si: mi.SurfaceInteraction3f,
    k_world: mi.Vector3f,
    tx_indices: mi.UInt,
    active: mi.Bool,
    diffracted_paths: bool,
    solid_angle: mi.Float | None = None,
    tx_positions: mi.Point3f | None = None,
    wedges: WedgeGeometry | None = None,
    diff_point: mi.Point3f | None = None,
    wedges_samples_cnt: mi.UInt | None = None,
) -> None:
    """MeshRadioMap.add_paths のインコヒーレント版

    変更箇所 (元実装 mesh_radio_map.py L132-134) :
        # 変更前: コヒーレント和
        a = dr.zeros(mi.Vector4f, 1)
        for e_field, aw in zip(e_fields, array_w):
            a += aw @ e_field
        a = dr.squared_norm(a)                  # |Σ E_l|²

        # 変更後: インコヒーレント和
        a = mi.Float(0)
        for e_field, aw in zip(e_fields, array_w):
            a += dr.squared_norm(aw @ e_field)  # Σ|E_l|²
    """
    tensor_ind = tx_indices * self.cells_count + si.prim_index

    # インコヒーレント和: 各パスの電力を個別に加算
    a = mi.Float(0)
    for e_field, aw in zip(e_fields, array_w, strict=False):
        a += dr.squared_norm(aw @ e_field)

    # レイ重み (元実装と同じ)
    if not diffracted_paths:
        cos_theta = dr.abs(dr.dot(si.n, k_world))
        w = solid_angle * dr.rcp(cos_theta)
    else:
        tx_positions_ = dr.gather(mi.Point3f, tx_positions, tx_indices, active=active)
        w = self._diffraction_integration_weight(wedges, tx_positions_, diff_point, k_world, si)
        w *= wedges.length * (dr.two_pi - wedge_interior_angle(wedges.n0, wedges.nn))
        w /= wedges_samples_cnt

    # セル面積による正規化 (元実装と同じ)
    # |v1 x v2| = sqrt(|v1|²|v2|² - (v1·v2)²)  ← ラグランジュ恒等式
    meas_surface = self.measurement_surface
    prim_index = meas_surface.face_indices(si.prim_index, active=active)
    v0 = meas_surface.vertex_position(prim_index[0])
    v1 = meas_surface.vertex_position(prim_index[1])
    v2 = meas_surface.vertex_position(prim_index[2])
    v1 = v1 - v0
    v2 = v2 - v0
    v1_sq_norm = dr.squared_norm(v1)
    v2_sq_norm = dr.squared_norm(v2)
    cell_area = 0.5 * dr.sqrt(v1_sq_norm * v2_sq_norm - dr.square(dr.dot(v1, v2)))
    w *= dr.rcp(cell_area)

    a *= w * self._normalization_factor

    dr.scatter_reduce(
        dr.ReduceOp.Add,
        self._pathgain_map.array,
        value=a,
        index=tensor_ind,
        active=active,
    )


def apply_incoherent_patch() -> None:
    """PlanarRadioMap と MeshRadioMap の add_paths をインコヒーレント版に差し替える

    RadioMapSolver() を呼ぶより前に、モジュールレベルで1回だけ呼び出すこと
    インストール済みファイルは変更しない (monkey-patch)

    Example
    -------
    >>> from radio_map_estimation.sionna.incoherent_patch import apply_incoherent_patch
    >>> apply_incoherent_patch()
    >>> from sionna.rt import RadioMapSolver
    >>> solver = RadioMapSolver()
    """
    from sionna.rt.radio_map_solvers.mesh_radio_map import MeshRadioMap
    from sionna.rt.radio_map_solvers.planar_radio_map import PlanarRadioMap

    PlanarRadioMap.add_paths = _planar_add_paths_incoherent
    MeshRadioMap.add_paths = _mesh_add_paths_incoherent

    logger.info(
        "incoherent_patch: PlanarRadioMap.add_paths and MeshRadioMap.add_paths "
        "replaced with incoherent power sum (Σ|E_l|²)."
    )
