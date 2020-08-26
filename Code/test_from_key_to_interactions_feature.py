import numpy as np
import time
import pytest

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import check_random_state
from SPP import simu, max_val
from numba.typed import List
from numba import njit
from scipy.sparse import csc_matrix


def from_key_to_interactions_feature(csc_data, csc_ind, csc_indptr, key):
    """
    Parameters
    ----------
    key: list(int)
        list of integers containing the indices of the binned features taking
        part in the interaction

    csc_data: list(int)
        list of integers containing the non-zero elements of the sparse matrix

    csc_ind: list(int)
        list of integers containing the row indices of the non-zero elements
        of the sparse matrix

    csc_indptr: list(int)
        list of integers containing the indices of each first non-zero elements
        of each feature in the csc_data vector

    Returns
    -------
    interfeat_data: list(int)
        list of integers containing the non-zero elements of the 
        feature of interactions
    
    interfeat_ind: list(int)
        list of integers containing the row indices of the non-zero elements
        of the feature of interactions

    """
    
    X_csc = csc_matrix((csc_data, csc_ind, csc_indptr), 
                       shape=(4, 3)).toarray()

    interfeat = np.ones(np.shape(X_csc)[0])     
    for idx in key:
        interfeat = np.multiply(X_csc[:, idx], interfeat)

    interfeat_data = []
    interfeat_ind = []
    ind = 0
    for data in interfeat:
        if data != 0:
            interfeat_data.append(data)
            interfeat_ind.append(ind)
        ind += 1

    return interfeat_data, interfeat_ind


def main():
    csc_data = [1, 1, 1, 1, 1]
    csc_ind = [0, 3, 1, 2, 2]
    csc_indptr = [0, 2, 4, 5]
    key = [1, 2]

    (interfeat_data, interfeat_ind) = \
        from_key_to_interactions_feature(csc_data=csc_data, csc_ind=csc_ind, 
                                         csc_indptr=csc_indptr, key=key)

    print('interfeat_data = ', interfeat_data)
    print('interfeat_ind = ', interfeat_ind)


if __name__ == "__main__":
    main()