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
