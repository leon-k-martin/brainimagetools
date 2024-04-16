def normalize_network_matrix(a):
    """Normalizes connectivity matrix by rowsums.

    :param a: NxN matrix
    :type a: numpy.ndarray, pd.DataFrame
    :return: Normalized matrix
    :rtype: numpy.ndarray, pd.DataFrame
    """
    return a / a.sum(axis=1, keepdims=True)
