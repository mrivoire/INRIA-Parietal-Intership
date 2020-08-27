import numpy as np
import time
import pytest

from SPP import simu, max_val
from scipy.sparse import csc_matrix
from SPP import from_key_to_interactions_feature, compute_interactions
from numba import njit
from numba.typed import List


# @njit
def test_from_key_to_interactions_feature():
    csc_data = [1, 1, 1, 1, 1]
    csc_ind = [0, 3, 1, 2, 2]
    csc_indptr = [0, 2, 4, 5]
    key = [2, 3]

    n_samples = 4
    n_features = len(csc_indptr) - 1

    # Compute interactions by hand 
    start_feat1, end_feat1 = csc_indptr[key[0] - 1: key[0] + 1]
    start_feat2, end_feat2 = csc_indptr[key[1] - 1: key[1] + 1]
    feat1_data = csc_data[start_feat1: end_feat1]
    feat1_ind = csc_ind[start_feat1: end_feat1]
    feat2_data = csc_data[start_feat2: end_feat2]
    feat2_ind = csc_ind[start_feat2: end_feat2]
    
    count1 = 0
    count2 = 0
    interfeat_test_data = []
    interfeat_test_ind = []

    while count1 < len(feat1_ind) and count2 < len(feat2_ind):
        if feat1_ind[count1] == feat2_ind[count2]:
            prod = feat1_data[count1] * feat2_data[count2]
            interfeat_test_ind.append(feat1_ind[count1])
            interfeat_test_data.append(prod)
            count1 += 1
            count2 += 1
        elif feat1_ind[count1] < feat2_ind[count2]:
            count1 += 1
        else:
            count2 += 1


    (interfeat_data, interfeat_ind) = \
        from_key_to_interactions_feature(csc_data=csc_data, csc_ind=csc_ind, 
                                         csc_indptr=csc_indptr, key=key, 
                                         n_samples=n_samples, 
                                         n_features=n_features)

    # np.testing.assert_array_equal(interfeat_data, [1.0])
    # np.testing.assert_array_equal(interfeat_ind, [2])

    np.testing.assert_array_equal(interfeat_data, interfeat_test_data)
    np.testing.assert_array_equal(interfeat_ind, interfeat_test_ind)


def main():
    test_from_key_to_interactions_feature()


if __name__ == "__main__":
    main()