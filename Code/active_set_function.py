import numpy as np
import time
import pytest

from scipy.linalg import toeplitz
from numba import njit
from numba.typed import List
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import Lasso as sklearn_Lasso
from sklearn.utils import check_random_state
from cd_solver_lasso_numba import Lasso, sparse_cd
from SPP import simu, compute_inner_prod, SPP 


def active_set_matrix(X_binned, active_set_keys):
    """
    Parameters
    ----------
    X_binned: np.ndarray(n_samples, n_features * n_bins)
        sparse matrix containing the new discrete features (one-hot vectors)

    active_set_keys: np.ndarray(n_active_features, )
        list of lists containing the keys of the active features

    Returns
    -------
    X_active_set: np.ndarray(n_samples, n_active_features)
        sparse matrix containing only the discrete active features
    """

    n_active_features = len(active_set_keys)
    n_samples = np.shape(X_binned)[0]

    X_active_set = np.zeros((n_samples, n_active_features))

    count = 0
    for key in active_set_keys:
        X_active_set[:, count] = X_binned[:, key]  
        count += 1

    return X_active_set


def active_set_matrix_2(dict_active_set):
    """
    Parameters
    ----------
    X_binned: np.ndarray(n_samples, n_features * n_bins)
        sparse matrix containing the new discrete features (one-hot vectors)

    dict_active_set: dictionary
        the keys of the dictionary are the keys of the active features 
        (list of indices) and the corresponding values are the binned features
        (one-hot encoded vectors)

    Returns
    -------
    X_active_set: np.ndarray(n_samples, n_active_features)
        sparse matrix containing only the discrete active features
    """

    X_active_set = []

    for ind, key in enumerate(dict_active_set):
        X_active_set.append(dict_active_set[key])

    X_active_set = np.array(X_active_set)

    return X_active_set


def main():

    X_binned = np.array([[1, 0, 0, 0, 0], 
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]])

    active_set_keys = {
        
    }

    active_set_keys = [1, 3, 4]

    X_active_set = active_set_matrix(X_binned=X_binned, 
                                     active_set_keys=active_set_keys)

    print("X_active_set = ", X_active_set)

    dict_active_set = {'[1, 2]': [0, 1, 0, 1, 0], 
                       '[0, 3]': [1, 0, 1, 0, 1],
                       '[1, 4]': [1, 1, 0, 0, 1]
                       }

    X_active_set_2 = active_set_matrix_2(dict_active_set=dict_active_set)

    print("X_active_set_2 = ", X_active_set_2)

if __name__ == "__main__":
    main()
