import numpy as np
import time
import pytest

from scipy.linalg import toeplitz
from numba import njit
from numba.typed import List
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import Lasso as sklearn_Lasso
from sklearn.utils import check_random_state
from cd_solver_lasso_numba import Lasso, sparse_cd
from SPP import simu, safe_prune, compute_inner_prod


@pytest.mark.parametrize("intersect", [True])
def test_safe_prune(intersect):

    # Parameters definition
    rng = check_random_state(0)
    n_samples, n_features = 30, 4
    beta = rng.randn(n_features)
    encode = 'onehot'
    strategy = 'quantile'
    lmbda = 1.
    n_epochs = 1000
    n_bins = 3
    max_depth = 2

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False,
                random_state=rng)

    # Discretization of the features by binning strategy
    enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    X_binned = enc.fit_transform(X)
    X_binned = X_binned.tocsc()
    X_binned_data = X_binned.data
    X_binned_indices = X_binned.indices
    X_binned_indptr = X_binned.indptr

    # Definition of the dimensions in dense format
    n_features = len(X_binned_indptr) - 1
    n_samples = max(X_binned_indices) + 1

    # Computation of the hyper-parameters : 
    # safe_sphere_radius, safe_sphere_center

    sparse_lasso_sklearn = sklearn_Lasso(alpha=(lmbda / X_binned.shape[0]),
                                         fit_intercept=False,
                                         normalize=False,
                                         max_iter=n_epochs,
                                         tol=1e-14).fit(X_binned, y)

    beta_star = sparse_lasso_sklearn.coef_
    residuals = y - X_binned.dot(beta_star)
    XTR_absmax = np.max(np.abs(X_binned.T @ residuals))

    theta = residuals / max(XTR_absmax, lmbda)

    P_lmbda = 0.5 * residuals.dot(residuals)
    P_lmbda += lmbda * np.linalg.norm(beta_star, 1)

    D_lmbda = 0.5 * np.linalg.norm(y, ord=2) ** 2
    D_lmbda -= (((lmbda ** 2) / 2) * np.linalg.norm(theta - y / lmbda, ord=2)
                ** 2)

    # Computation of the dual gap
    G_lmbda = P_lmbda - D_lmbda
    safe_sphere_radius = np.sqrt(2 * G_lmbda) / lmbda
    safe_sphere_radius = 1
    safe_sphere_center = theta

    # Computation of the active set with recursive function

    start_sp = time.time()
    (safe_set_data, safe_set_ind, safe_set_key, flatten_safe_set_data,
        flatten_safe_set_ind, safe_set_indptr) = safe_prune(
        X_binned_data=X_binned_data, X_binned_indices=X_binned_indices,
        X_binned_indptr=X_binned_indptr, safe_sphere_center=safe_sphere_center,
        safe_sphere_radius=safe_sphere_radius, max_depth=max_depth)

    end_sp = time.time()
    delay_sp = end_sp - start_sp
    print("delay safe prune = ", delay_sp)

    safe_set_key = list(safe_set_key)
    for key in safe_set_key:
        key.sort()

    flat_safe_set_key = []

    for item in safe_set_key:
        new_item = []
        for it in item:
            new_item.append(it)
        flat_safe_set_key.append(tuple(new_item))

    print("flat_safe_set_key = ", flat_safe_set_key)

    # Computation of the active set with nested loop
    # X_binned = X_binned.toarray()
    safe_set_key_test = []
    safe_set_data_test = []
    safe_set_data_card_test = []
    start_sp_test = time.time()
    for j in range(n_features):
        for k in range(j, n_features):
            # for i in range(k, n_features):
            print('shapes =', (X_binned[:, j].shape, X_binned[:, k].shape))
            inter_feat = X_binned[:, j].multiply(X_binned[:, k])
            card = inter_feat.nnz
            safe_set_data_card_test.append(card)
            u_t = inter_feat.T @ safe_sphere_center
            v_t = (inter_feat.data ** 2).sum()
            sppc_t = abs(u_t) + safe_sphere_radius * np.sqrt(v_t)

            if sppc_t >= 1:
                safe_set_data_test.append(inter_feat)
                if j == k:
                    safe_set_key_test.append([j])
                else:
                    safe_set_key_test.append([j, k])
                # if j == k == i:
                #     safe_set_key_test.append([j, ])
                # elif j == k and j != i:
                #     safe_set_key_test.append([j, i, ])
                # elif j == i and j != k:
                #     safe_set_key_test.append([j, k, ])
                # # elif j == i and i != k:
                # #     safe_set_key_test.append([k, i, ])
                # elif j != i and j != k:
                #     safe_set_key_test.append([j, k, i])

    end_sp_test = time.time()
    delay_sp_test = end_sp_test - start_sp_test
    print("delay safe pruning test = ", delay_sp_test)

    safe_set_key_test = list(safe_set_key_test)
    for key in safe_set_key_test:
        key.sort()

    # safe_set_data_test = tuple([data] for data in safe_set_data_test)
    # safe_set_key_test = tuple([key] for key in safe_set_key_test)

    for ind in range(len(safe_set_key_test)):
        safe_set_key_test[ind] = tuple(safe_set_key_test[ind])

    print('safe_set_key_test = ', safe_set_key_test)

    assert safe_set_key_test == flat_safe_set_key

    if intersect:
        intersect_key = [ele1 for ele1 in flat_safe_set_key if ele1 in
                         safe_set_key_test]
        # print("intersect_key = ", intersect_key)

        if len(intersect_key) == len(flat_safe_set_key):
            print("same length")

        # print("flat safe set key = ", flat_safe_set_key[:15])
        # print("safe set key test = ", safe_set_key_test[:15])

        print("length intersect_key = ", len(intersect_key))
        print("length safe set key : ", len(flat_safe_set_key))
        print("length safe set key test = ", len(safe_set_key_test))

        assert len(intersect_key) == len(flat_safe_set_key)
        assert len(intersect_key) == len(safe_set_key_test)