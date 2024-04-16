# matrix.py
"""
Tools for working with and visualizing connectivity matrices.

Authors: Leon Martin
"""


def min_max_scaler(df, column_wise=False):
    """_summary_

    Parameters
    ----------
    df : pandas.DataFrame
        _description_
    column_wise : bool, optional
        _description_, by default False

    Returns
    -------
    pandas.DataFrame
        normalised matrix
    """

    if column_wise:
        df_norm = (df - df.min()) / (df.max() - df.min())
    else:
        df_norm = (df - df.min().min()) / (df.max().max() - df.min().min())

    return df_norm


def matrix2triplets(df):
    return df.stack().reset_index().rename({0: "value"}, axis=1)
