
import pandas as pd
import numpy as np
from numpy.linalg import svd

def make_labeled_svd(df, reduce=None, round=None):
    U, sigma, VT = svd(df)
    if round:
        U, sigma, VT = (np.around(x, round) for x in (U, sigma, VT))
    U_df = pd.DataFrame(U, index=df.index)
    VT_df = pd.DataFrame(VT, columns=df.columns)
    sigma_reduced = np.zeros(df.shape)
    np.fill_diagonal(sigma_reduced, sigma)
    if reduce: 
        sigma_reduced[:, reduce:] = 0
    return U, sigma_reduced, VT, U_df, VT_df
U, sigma_reduced, VT, U_df, VT_df = make_labeled_svd(ratings_df, reduce=None, round=None)