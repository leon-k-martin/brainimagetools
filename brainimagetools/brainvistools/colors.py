from matplotlib.colors import to_hex
import numpy as np
import hcp_utils as hcp



def hex_to_rgb(value):
    """Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values

    :param value: hexcode
    :type value: str
    :return: rgba value
    :rtype: tuple
    """
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """Converts rgb to decimal colours (i.e. divides each value by 256)

    :param value: rgb values
    :type value: tuple
    :return: decimal rgb values
    :rtype: tuple
    """
    return [v / 256 for v in value]


def rgb_to_hex(c, keep_alpha=True):
    if isinstance(c, list) or isinstance(c, np.ndarray):
        return [to_hex(ci, keep_alpha=keep_alpha) for ci in c]

    else:
        return to_hex(c, keep_alpha=keep_alpha)


nature_colors = ["#e64b35", "#4dbbd5", "#00a087", "#3c5488", "#f39b7f"]

mmp1_colors = rgb_to_hex(list(hcp.mmp["rgba"].values()))
