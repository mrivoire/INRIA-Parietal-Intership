import numpy as np
import time
import pytest
import numba

from SPP import simu, max_val
from scipy.sparse import csc_matrix
from SPP import from_key_to_interactions_feature, compute_interactions
from numba import njit
from numba.typed import List
from numba import generated_jit


def test_from_key_to_interactions_feature():
    csc_data = [1, 1, 1, 1, 1]
    csc_ind = [0, 3, 1, 2, 2]
    csc_indptr = [0, 2, 4, 5]
    key = [1, 2]

    n_samples = 4
    n_features = len(csc_indptr) - 1

    # Compute interactions by hand
    start_feat1, end_feat1 = csc_indptr[key[0]: key[0] + 2]
    start_feat2, end_feat2 = csc_indptr[key[1]: key[1] + 2]
    feat1_data = csc_data[start_feat1: end_feat1]
    feat1_ind = csc_ind[start_feat1: end_feat1]
    feat2_data = csc_data[start_feat2: end_feat2]
    feat2_ind = csc_ind[start_feat2: end_feat2]

    interfeat_test_data, interfeat_test_ind = \
        compute_interactions(data1=feat1_data, ind1=feat1_ind,
                             data2=feat2_data, ind2=feat2_ind)

    (interfeat_data, interfeat_ind) = \
        from_key_to_interactions_feature(csc_data=csc_data, csc_ind=csc_ind,
                                         csc_indptr=csc_indptr, key=key,
                                         n_samples=n_samples,
                                         n_features=n_features)

    # np.testing.assert_array_equal(interfeat_data, [1.0])
    # np.testing.assert_array_equal(interfeat_ind, [2])

    np.testing.assert_array_equal(interfeat_data, interfeat_test_data)
    np.testing.assert_array_equal(interfeat_ind, interfeat_test_ind)