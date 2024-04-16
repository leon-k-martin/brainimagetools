import numpy as np
import pandas as pd

def upper(df):
    '''Returns the upper triangle of a correlation matrix.
    You can use scipy.spatial.distance.squareform to recreate matrix from upper triangle.
    Args:
      df: pandas or numpy correlation matrix
    Returns:
      list of values from upper triangle
    '''
    try:
        assert(type(df)==np.ndarray)
    except:
        if type(df)==pd.DataFrame:
            df = df.values
        else:
            raise TypeError('Must be np.ndarray or pd.DataFrame')
    mask = np.triu_indices(df.shape[0], k=1)
    return df[mask]


def fractional_rescale(matrix):
        '''
        applying fractional rescaling of connectivity matrix
        following Rosen and Halgren 2021 eNeuro
        F(DTI(i,j)) := DTI(i,j) / sum(DTI(i,x))+sum(DTI(y,i)) with x!=i,y!=j
        '''
        colsum = np.nansum(matrix, axis = 0)
        _temp1 = np.tile(colsum,(colsum.shape[0],1))
        colsum = np.nansum(matrix, axis = 1)
        _temp2 = np.tile(colsum,(colsum.shape[0],1))
        return( matrix / ( (_temp1 + _temp2.T ) - (2*matrix)) )