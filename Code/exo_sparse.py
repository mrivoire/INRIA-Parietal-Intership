import numpy as np
from scipy.sparse import csc_matrix


########################################################################
#                    Print column of sparse matrix
########################################################################


def column_sparse_matrix(data, indices, indptr, j):
    """Return the wished column of a sparse matrix

    Parameters
    ----------
    data: numpy.array, shape = (number of non-zero elements, )
        contains the non-zero elements of the sparse matrix

    indices: csc format index array, shape = (number of non-zero elements, )
        contains the indices of the rows to which belong the non-zero elements
        of the sparse matrix

    indptr: csc format index pointer array, shape = (number of columns + 1, )
        contains the indices of the columns which have at least a non-zero
        element

    j: int
        number of the column to print

    Returns
    -------
    col: numpy.array, shape = (n_rows, )
    """

    # To select the indices of the non-zero elements of the wished column
    start, end = indptr[j:j+2]
    n_rows = max(indices) + 1

    col = np.zeros(n_rows)

    # We scan the elements of the data array corresponding to the non-zero
    # whose the indices correspond to the one selected for the wished column
    for ind in range(start, end):
        col[indices[ind]] = data[ind]
        # So that the non-zero elements might be properly placed in the
        # original  sparse matrix we have to take into account the indices
        # of the rows to which they belong that is the reason why we use
        # indices[ind] to obtain the proper row index

    return col


############################################################################
#               Product vector - matrix with a sparse matrix
############################################################################


def sparse_matrix_product(data, indices, indptr, vect):
    """Perform the matrix product between a given vector and a sparse matrix

    Parameters
    ----------
    data: numpy.array, shape = (number of non-zero elements, )
        contains the non-zero elements of the sparse matrix

    indices: csc format index array, shape = (number of non-zero elements, )
        contains the indices of the rows to which belong the non-zero elements
        of the sparse matrix

    indptr: csc format index pointer array, shape = (n_col + 1, )
        contains the indices of the columns which have at least a non-zero
        element

    vect: numpy.array, shape = (, n_rows)

    Returns
    -------
    prod: float
        matrix product between the sparse matrix
    """

    n_rows = max(indices) + 1
    n_cols = len(indptr) - 1

    prod = np.zeros(n_cols)

    col = np.zeros(n_rows)

    for j in range(n_cols):
        col = np.zeros(n_rows)
        start, end = indptr[j:j+2]
        for ind in range(start, end):
            col[indices[ind]] = data[ind] 

        tmp_prod = 0.

        for i in range(n_rows):
            tmp_prod += col[i] * vect[i]
        prod[j] = tmp_prod

    return prod


def main():

    A = np.array([[0, 1, 0], [2, 0, 0], [3, 0, 4]])
    print("A = ", A)
    A = csc_matrix(A)
    A_data = A.data
    A_indices = A.indices
    A_indptr = A.indptr

    col = column_sparse_matrix(A_data, A_indices, A_indptr, 2)
    print("col = ", col)

    # row = row_sparse_matrix(A_data, A_indices, A_indptr, 1)
    # print("row = ", row)

    vect = np.array([1, 2, 3])

    prod = sparse_matrix_product(A_data, A_indices, A_indptr, vect)
    print("prod = ", prod)

    A.toarray()

    print("result = ", vect.dot(A))

    

if __name__ == "__main__":
    main()