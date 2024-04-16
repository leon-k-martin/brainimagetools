import numpy as np


def norm(x):
    """Normalise array betweeen 0-1

    :param x: _description_
    :type x: _type_
    :return: _description_
    :rtype: _type_
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))
