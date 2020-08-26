import numpy as np

from SPP import from_numbalists_tocsc

from numba.typed import List
from numba import njit


@njit
def main():

    numbalist_data = List([[1, 2], [4, 5], [6]])
    numbalist_ind = List([[0, 3], [1, 2], [2]])

    print("numbalist_data = ", numbalist_data)
    print("numbalist_ind = ", numbalist_ind)

    csc_data, csc_ind, csc_indptr = \
        from_numbalists_tocsc(numbalist_data=numbalist_data,
                              numbalist_ind=numbalist_ind)

    return csc_data, csc_ind, csc_indptr


def test_from_numbalists_tocsc():
    csc_data, csc_ind, csc_indptr = main()

    print("csc_data = ", csc_data)
    print("csc_ind = ", csc_ind)
    print("csc_indptr = ", csc_indptr)

    np.testing.assert_array_equal(csc_data, [1, 2, 4, 5, 6])
    np.testing.assert_array_equal(csc_ind, [0, 3, 1, 2, 2])
    np.testing.assert_array_equal(csc_indptr, [0, 2, 4, 5])


if __name__ == "__main__":
    main()
