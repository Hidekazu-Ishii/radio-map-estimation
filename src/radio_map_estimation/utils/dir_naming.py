from pathlib import Path


def build_trial_output_dir(
    root: Path,
    city_dir: str,
    mesh_code: str,
    freq_ghz: str,
    train_size: int,
    n_test_prod: int,
    trial_idx: int,
    pathloss_model: str,
    shadowing_model: str | None,
    kernel: str | None,
) -> Path:
    """trial 1回分の出力ディレクトリパスを構築する (純粋関数、mkdirは呼び出し元の責務)
    shadowing_model, kernel は該当モデルを使わないケース (パスロスのみの推定等) を
    表現するため Optional. None の要素はディレクトリ名から除外する.
    """
    model_parts = [pathloss_model, shadowing_model, kernel]
    model_dir_name = "_".join(part for part in model_parts if part is not None)

    return (
        root
        / "outputs"
        / "scratch"
        / city_dir
        / mesh_code
        / freq_ghz
        / model_dir_name
        / f"train{train_size}_test{n_test_prod}"
        / f"trial{trial_idx}"
    )


def freq_dir_name(freq_hz: float) -> str:
    """周波数 [Hz] をディレクトリ名用の文字列に変換する

    整数GHzの場合は小数点以下1桁、非整数の場合は有効数字10桁で表記する

    Args:
        freq_hz: 周波数 [Hz]

    Returns:
        ディレクトリ名として使える文字列 (例: 2.0e9 -> "2.0GHz")
    """
    ghz = freq_hz / 1e9
    return f"{ghz:.10g}GHz" if ghz != int(ghz) else f"{ghz:.1f}GHz"
