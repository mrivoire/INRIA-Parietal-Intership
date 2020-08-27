import numpy as np
import time
import pytest

from SPP import simu, max_val
from scipy.sparse import csc_matrix
from SPP import from_key_to_interactions_feature
from numba import njit
from numba.typed import List
from SPP import compute_interactions


# @njit
def from_key_to_interactions_feature(csc_data, csc_ind, csc_indptr, 
                                     key, n_samples, n_features):
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
    
    # X_csc = csc_matrix((csc_data, csc_ind, csc_indptr), 
    #                    shape=(n_samples, n_features)).toarray()

    # interfeat = np.ones(np.shape(X_csc)[0])    

    n_features = len(csc_indptr) - 1
    n_samples = len(set(csc_ind)) 

    interfeat_data = List(np.ones(n_samples))
    print('interfeat_data_init = ', interfeat_data)
    interfeat_ind = List(np.arange(n_samples))
    print('interfeat_ind_init = ', interfeat_ind)
    csc_data = List(csc_data)
    print('csc_data = ', csc_data)
    csc_ind = List(csc_ind)
    print('csc_ind = ', csc_ind)
    csc_indptr = List(csc_indptr)
    print('csc_indptr = ', csc_indptr)

    for idx in key:
        print('idx = ', idx)
        print('idx - 1 =', idx - 1)
        print('idx + 1 = ', idx + 1)
        start, end = csc_indptr[idx - 1: idx + 1]
        start = np.int64(start)
        end = np.int64(end)
        print('start = ', start)
        print('end = ', end)
        data2 = csc_data[start: end]
        ind2 = csc_ind[start: end]
        print('data2 = ', data2)
        print('ind2 = ', ind2)
        interfeat_data, interfeat_ind = \
            compute_interactions(data1=interfeat_data, 
                                 ind1=interfeat_ind, 
                                 data2=data2, 
                                 ind2=ind2)

    return interfeat_data, interfeat_ind


# @njit
def test_from_key_to_interactions_feature():
    csc_data = [1, 1, 1, 1, 1]
    csc_ind = [0, 3, 1, 2, 2]
    csc_indptr = [0, 2, 4, 5]
    key = [1, 2]

    n_samples = 4
    n_features = len(csc_indptr) - 1

    (interfeat_data, interfeat_ind) = \
        from_key_to_interactions_feature(csc_data=csc_data, csc_ind=csc_ind, 
                                         csc_indptr=csc_indptr, key=key, 
                                         n_samples=n_samples, 
                                         n_features=n_features)

    # np.testing.assert_array_equal(interfeat_data, [1.0])
    # np.testing.assert_array_equal(interfeat_ind, [2])
    print('interfeat_data = ', interfeat_data)
    print('interfeat_ind = ', interfeat_ind)


def main():
    csc_data = [1, 1, 1, 1, 1]
    csc_ind = [0, 3, 1, 2, 2]
    csc_indptr = [0, 2, 4, 5]
    key = [1, 2]

    n_samples = 4
    n_features = len(csc_indptr) - 1

    interfeat_data, interfeat_ind = \
        from_key_to_interactions_feature(csc_data=csc_data, 
                                         csc_ind=csc_ind, 
                                         csc_indptr=csc_indptr, 
                                         key=key, 
                                         n_samples=n_samples, 
                                         n_features=n_features)

    print("interfeat_data = ", interfeat_data)
    print("interfeat_ind = ", interfeat_ind)

    # test_from_key_to_interactions_feature()

if __name__ == "__main__":
    main()
