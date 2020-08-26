import numpy as np
import time
import pytest

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import check_random_state
from SPP import simu, max_val
from numba.typed import List
from numba import njit


@njit
def from_numbalists_tocsc(numbalist_data, numbalist_ind):
    """
    Parameters
    ----------
    numbalist_data:

    numbalist_ind:

    Returns
    -------
    csc_data:

    csc_ind:

    csc_indptr:

    """
    csc_data = []
    csc_ind = []
    csc_indptr = [0]

    indptr = 0
    for feat in numbalist_data:
        indptr += len(feat)
        csc_indptr.append(indptr)
        for data in feat:
            csc_data.append(data)

    for ind_list in numbalist_ind:
        for ind in ind_list:
            csc_ind.append(ind)

    return csc_data, csc_ind, csc_indptr


@njit
def main():

    numbalist_data = List([[1, 2], [4, 5], [6]])
    numbalist_ind = List([[0, 3], [1, 2], [2]])

    print("numbalist_data = ", numbalist_data)
    print("numbalist_ind = ", numbalist_ind)

    csc_data, csc_ind, csc_indptr = \
        from_numbalists_tocsc(numbalist_data=numbalist_data, 
                              numbalist_ind=numbalist_ind)

    print("csc_data = ", csc_data)
    print("csc_ind = ", csc_ind)
    print("csc_indptr = ", csc_indptr)

    
if __name__ == "__main__":
    main()