import numpy as np
# import time
import pytest

# from scipy.linalg import toeplitz
# from numba import njit
# from numba.typed import List
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import Lasso as sklearn_Lasso
from sklearn.utils import check_random_state
# from cd_solver_lasso_numba import Lasso, sparse_cd
from SPP import simu, SPP, compute_interactions, max_val
from scipy.sparse import csc_matrix, hstack


@pytest.mark.parametrize("seed", [0, 1, 2, 5, 10])
def test_SPP(seed):

    # Definition of the parameters
    rng = check_random_state(1)
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
    n_val_gs = 10
    tol = 1e-08

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False,
                random_state=rng)

    # Discretization by binning strategy
    enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    X_binned = enc.fit_transform(X)
    X_binned = X_binned.tocsc()
    X_binned_data = X_binned.data
    X_binned_indices = X_binned.indices
    X_binned_indptr = X_binned.indptr

    # Definition of the dimensions
    n_features = len(X_binned_indptr) - 1
    n_samples = max(X_binned_indices) + 1
    print('n_features = ', n_features)
    # Lasso
    # We have to run the lasso of sklearn on the matrix containing the whole
    # set of features i.e. both the discrete features provided thanks to the
    # binning process of the original continuous features and the features of
    # interactions until the maximum order

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
        X_tilde_keys.append([i])
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

            X_tilde_keys.append([i, j])
            indptr += len(inter_feat_ind)

            X_inter_feat_data.extend(inter_feat_data)
            X_inter_feat_ind.extend(inter_feat_ind)
            X_inter_feat_indptr.append(indptr)

    X_interfeats = csc_matrix(
        (X_inter_feat_data, X_inter_feat_ind, X_inter_feat_indptr),
        shape=(n_samples, n_inter_feats))

    X_tilde = hstack([X_binned, X_interfeats])
    X_tilde_data = X_tilde.data
    X_tilde_indptr = X_tilde.indptr
    X_tilde_ind = X_tilde.indices

    lambda_max, max_key = max_val(X_binned_data=X_binned_data,
                                  X_binned_indices=X_binned_indices,
                                  X_binned_indptr=X_binned_indptr,
                                  residuals=y, max_depth=max_depth)

    lmbda = lambda_max / 2

    n_tilde_feats = n_features + n_inter_feats
    print('n_tilde_feats = ', n_tilde_feats)

    # (beta_star_lasso, residuals, primal_hist_sparse, dual_hist_sparse,
    #  gap_hist_sparse, r_list_sparse, n_active_features_true_sparse,
    #  theta_hat_cyclic_cd_sparse, P_lmbda_sparse, D_lmbda_sparse,
    #  G_lmbda_sparse, safe_set_sparse) = \
    #     sparse_cd(X_data=X_tilde_data, X_indices=X_tilde_ind,
    #               X_indptr=X_tilde_indptr, y=y, lmbda=lmbda,
    #               epsilon=epsilon, f=f, n_epochs=n_epochs,
    #               screening=screening, store_history=store_history)

    sparse_lasso_sklearn = sklearn_Lasso(alpha=(lmbda / X_tilde.shape[0]),
                                         fit_intercept=False,
                                         normalize=False,
                                         max_iter=n_epochs,
                                         tol=1e-14).fit(X_tilde, y)

    beta_star_lasso = sparse_lasso_sklearn.coef_
    active_set_keys_lasso = []
    for i in range(n_tilde_feats):
        if beta_star_lasso[i] != 0:
            active_set_keys_lasso.append(X_tilde_keys[i])

    solutions_dict = \
        SPP(X_binned_data=X_binned_data, X_binned_indices=X_binned_indices,
            X_binned_indptr=X_binned_indptr, y=y, n_val_gs=n_val_gs,
            max_depth=max_depth, epsilon=epsilon, f=f, n_epochs=n_epochs,
            tol=tol, screening=screening, store_history=store_history)

    beta_star_spp = solutions_dict['spp_lasso_slopes']
    print('beta_star_spp = ', beta_star_spp)
    print('beta_star_lasso = ', beta_star_lasso)
    print('length beta_star_spp = ', len(beta_star_spp))
    print('length beta_star_lasso = ', np.count_nonzero(beta_star_lasso))

    active_set_data = solutions_dict['data']
    print('length active_set_data = ', len(active_set_data))

    print('length X_tilde_data = ', len(X_tilde_data))

    active_set_ind = solutions_dict['ind']
    print('length active_set_ind = ', len(active_set_ind))

    print('length X_tilde_ind = ', len(X_tilde_ind))

    active_set_indptr = solutions_dict['indptr']
    print('length active_set_indptr = ', len(active_set_indptr))

    print('length X_tilde_indptr = ', len(X_tilde_indptr))

    active_set_keys = solutions_dict['keys']
    print('length active_set_keys = ', len(active_set_keys))
    print('length X_tilde_keys = ', len(X_tilde_keys))
    print('active_set_keys_lasso = ', active_set_keys_lasso)
    print('length active_set_keys_lasso = ', len(active_set_keys_lasso))

    print('X_tilde shape 0 = ', X_tilde.shape[0])

    assert len(active_set_data) == len(active_set_ind)
    assert len(active_set_data) == len(active_set_keys)
    assert len(active_set_data) == np.count_nonzero(beta_star_spp)
    # assert np.count_nonzero(beta_star_spp) == np.count_nonzero(beta_star_lasso)


def main():
    seed = 1
    test_SPP(seed)


if __name__ == "__main__":
    main()
