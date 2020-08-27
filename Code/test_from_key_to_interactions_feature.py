import numpy as np
import time
import pytest

from SPP import simu, max_val
from scipy.sparse import csc_matrix
from SPP import from_key_to_interactions_feature
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

    (interfeat_data, interfeat_ind) = \
        from_key_to_interactions_feature(csc_data=csc_data, csc_ind=csc_ind, 
                                         csc_indptr=csc_indptr, key=key, 
                                         n_samples=n_samples, 
                                         n_features=n_features)

    np.testing.assert_array_equal(interfeat_data, [1.0])
    np.testing.assert_array_equal(interfeat_ind, [2])
    print('interfeat_data = ', interfeat_data)
    print('interfeat_ind = ', interfeat_ind)


if __name__ == "__main__":
    main()
