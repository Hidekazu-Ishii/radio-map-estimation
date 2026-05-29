"""
PLY ファイル群から Plotly 3D プレビュー HTML を生成する

役割
----
data/processed/<city_dir>/<mesh_code>/ の PLY ファイルを読み込み、
同ディレクトリに scene.html として保存する

設計方針
--------
- combined.ply はスキップ (個別 PLY を材質ごとに色分けして表示)
- 面数が多い場合は simplify_quadric_decimation で間引く (インタラクティブ性を確保)
- I/O のみを担う (PLY 生成・XML 生成は mesh_builder / mitsuba_xml へ)
"""

from __future__ import annotations

import logging
from pathlib import Path

import plotly.graph_objects as go
import trimesh

logger = logging.getLogger(__name__)

# 材質ごとの色・透明度
_MESH_CONFIGS: dict[str, tuple[str, float]] = {
    "bldg": ("rgb(200,200,200)", 1.0),
    "dem": ("rgb(139,115,85)", 0.8),
    "tran": ("rgb(80,80,80)", 1.0),
    "wtr": ("rgb(64,164,223)", 0.9),
}

# 材質ごとの最大面数 (超過時に間引く)
_MAX_FACES: dict[str, int] = {
    "bldg": 10000,
    "dem": 10000,
    "tran": 1000,
    "wtr": 1000,
}


def save_scene_preview(
    scene_dir: Path,
    html_path: Path,
    title: str,
) -> None:
    """
    scene_dir 内の PLY ファイルから Plotly 3D プレビュー HTML を生成する

    combined.ply はスキップし、bldg / dem / tran / wtr を材質ごとに色分けして表示する

    Parameters
    ----------
    scene_dir : PLY ファイルが格納されたディレクトリ
                 (data/processed/<city_dir>/<mesh_code>/)
    html_path : 出力 HTML のパス (通常 scene_dir / "scene.html")
    title     : プレビューのタイトル
    """
    fig = go.Figure()
    n_traces = 0

    for ply_path in sorted(scene_dir.glob("*.ply")):
        mesh = trimesh.load(str(ply_path), force="mesh")
        assert isinstance(mesh, trimesh.Trimesh)

        # 面数が上限を超える場合は間引く
        key = next((k for k in _MAX_FACES if k in ply_path.name), None)
        if key is not None:
            limit = _MAX_FACES[key]
            if len(mesh.faces) > limit:
                reduction = 1.0 - limit / len(mesh.faces)
                mesh = mesh.simplify_quadric_decimation(reduction)

        v, f = mesh.vertices, mesh.faces
        color, opacity = next(
            (v2 for k, v2 in _MESH_CONFIGS.items() if k in ply_path.name),
            ("rgb(200,200,200)", 1.0),
        )

        fig.add_trace(
            go.Mesh3d(
                x=v[:, 0].tolist(),
                y=v[:, 1].tolist(),
                z=v[:, 2].tolist(),
                i=f[:, 0].tolist(),
                j=f[:, 1].tolist(),
                k=f[:, 2].tolist(),
                color=color,
                opacity=opacity,
                name=ply_path.stem,
            )
        )
        n_traces += 1
        logger.info("Preview: added %s (%d faces)", ply_path.stem, len(f))

    if n_traces == 0:
        logger.warning("No PLY files found in %s, preview not generated.", scene_dir)
        return

    fig.update_layout(
        scene={"aspectmode": "data"},
        title=title,
    )

    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path))
    logger.info("Saved scene preview: %s", html_path)
