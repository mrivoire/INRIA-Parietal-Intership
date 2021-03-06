import numpy as np
# import time
import pytest

from scipy.sparse import csc_matrix, hstack

from numba.typed import List
from numba import njit

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import Lasso as sklearn_Lasso
from sklearn.utils import check_random_state
# from cd_solver_lasso_numba import Lasso, sparse_cd

from SPP import simu, spp_solver, compute_interactions, max_val
from SPP import from_numbalists_tocsc


def _compute_all_interactions(X_binned):
    n_samples = X_binned.shape[0]
    X_binned_data = X_binned.data
    X_binned_indices = X_binned.indices
    X_binned_indptr = X_binned.indptr

    n_features = X_binned.shape[1]
    X_inter_feat_data = []
    X_inter_feat_ind = []
    X_inter_feat_indptr = []
    indptr = 0
    X_inter_feat_indptr.append(indptr)
    n_inter_feats = 0

    # Computation of the features of interactions
    X_tilde_keys = []

    # Why testing with sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False) ?
    for i in range(n_features):
        start_feat1, end_feat1 = X_binned_indptr[i: i + 2]
        for j in range(i, n_features):
            n_inter_feats += 1
            start_feat2, end_feat2 = X_binned_indptr[j: j + 2]

            inter_feat_data, inter_feat_ind = \
                compute_interactions(
                    data1=X_binned_data[start_feat1: end_feat1],
                    ind1=X_binned_indices[start_feat1: end_feat1],
                    data2=X_binned_data[start_feat2: end_feat2],
                    ind2=X_binned_indices[start_feat2: end_feat2])

            if i != j:
                X_tilde_keys.append([i, j])
            else:
                X_tilde_keys.append([i])
            indptr += len(inter_feat_ind)

            X_inter_feat_data.extend(inter_feat_data)
            X_inter_feat_ind.extend(inter_feat_ind)
            X_inter_feat_indptr.append(indptr)

    X_interfeats = csc_matrix(
        (X_inter_feat_data, X_inter_feat_ind, X_inter_feat_indptr),
        shape=(n_samples, n_inter_feats))

    return X_interfeats, X_tilde_keys


# @pytest.mark.parametrize("seed", [0, 1, 2, 5, 10])
def test_SPP():

    # Definition of the parameters
    rng = check_random_state(4)
    n_samples, n_features = 100, 10
    beta = rng.randn(n_features)
    lmbda = 1.
    f = 10
    epsilon = 1e-14
    n_epochs = 100000
    screening = True
    store_history = True
    encode = 'onehot'
    strategy = 'quantile'
    n_bins = 3
    max_depth = 2
    n_lambda = 10
    tol = 1e-08
    lambda_max_ratio = 0.5
    n_active_max = 100

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False,
                random_state=rng)

    # Discretization by binning strategy
    enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    X_binned = enc.fit_transform(X)
    X_binned = X_binned.tocsc()

    # Lasso
    # We have to run the lasso of sklearn on the matrix containing the whole
    # set of features i.e. both the discrete features provided thanks to the
    # binning process of the original continuous features and the features of
    # interactions until the maximum order

    X_tilde, X_tilde_keys = _compute_all_interactions(X_binned)

    X_binned_data = X_binned.data
    X_binned_indices = X_binned.indices
    X_binned_indptr = X_binned.indptr
    lambda_max, max_key = max_val(X_binned_data=X_binned_data,
                                  X_binned_indices=X_binned_indices,
                                  X_binned_indptr=X_binned_indptr,
                                  residuals=y, max_depth=max_depth)

    lmbda = lambda_max / 2

    sparse_lasso_sklearn = sklearn_Lasso(alpha=(lmbda / X_tilde.shape[0]),
                                         fit_intercept=False,
                                         normalize=False,
                                         max_iter=n_epochs,
                                         tol=tol).fit(X_tilde, y)

    beta_star_lasso = []
    active_set_keys_lasso = []
    for i, coef in enumerate(sparse_lasso_sklearn.coef_):
        if coef != 0:
            active_set_keys_lasso.append(X_tilde_keys[i])
            beta_star_lasso.append(coef)

    lambdas=[lmbda]
    solutions = \
        spp_solver(X_binned=X_binned, y=y, n_lambda=n_lambda, lambdas=lambdas,
                   max_depth=max_depth, epsilon=epsilon, f=f,
                   n_epochs=n_epochs,
                   tol=tol, lambda_max_ratio=lambda_max_ratio, 
                   n_active_max=n_active_max, screening=screening, 
                   store_history=store_history)

    beta_star_spp = solutions[0]['spp_lasso_slopes']
    active_set_keys_spp = solutions[0]['keys']

    assert len(active_set_keys_spp) == len(beta_star_spp)
    assert len(active_set_keys_spp) == len(active_set_keys_lasso)
    assert len(beta_star_spp) == len(beta_star_lasso)

    print('length active_set_keys_spp = ', len(active_set_keys_spp))
    print('length beta_star_spp = ', len(beta_star_spp))
    print('NNZ beta_star_lasso = ', np.count_nonzero(beta_star_lasso))
    print('beta_star_spp = ', beta_star_spp)
    print('beta_star_lasso = ', beta_star_lasso)
    print('active_set_keys_spp = ', active_set_keys_spp)
    print('active_set_keys_lasso = ', active_set_keys_lasso)

    equality_test = [False] * len(active_set_keys_spp)

    for i, key_spp in enumerate(active_set_keys_spp):
        for j, key_lasso in enumerate(active_set_keys_lasso):
            key_lasso = list(key_lasso)
            key_spp = list(key_spp)

            if (key_lasso == key_spp) or (key_lasso.reverse() == key_spp):
                np.testing.assert_allclose(beta_star_spp[i],
                                           beta_star_lasso[j],
                                           rtol=1e-05)

                equality_test[i] = True

                pass

    if False not in equality_test:
        print('right test : solutions are equal')
    else:
        print('wrong test : solutions are not equal')


@njit
def _aux_from_numbalists_tocscmain():

    numbalist_data = List([[1, 2], [4, 5], [6]])
    numbalist_ind = List([[0, 3], [1, 2], [2]])

    print("numbalist_data = ", numbalist_data)
    print("numbalist_ind = ", numbalist_ind)

    csc_data, csc_ind, csc_indptr = \
        from_numbalists_tocsc(numbalist_data=numbalist_data,
                              numbalist_ind=numbalist_ind)

    return csc_data, csc_ind, csc_indptr


def test_from_numbalists_tocsc():
    csc_data, csc_ind, csc_indptr = _aux_from_numbalists_tocscmain()

    print("csc_data = ", csc_data)
    print("csc_ind = ", csc_ind)
    print("csc_indptr = ", csc_indptr)

    np.testing.assert_array_equal(csc_data, [1, 2, 4, 5, 6])
    np.testing.assert_array_equal(csc_ind, [0, 3, 1, 2, 2])
    np.testing.assert_array_equal(csc_indptr, [0, 2, 4, 5])


def main():

    test_SPP()


if __name__ == "__main__":
    main()
