import numpy as np
import time
import pytest

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import Lasso
from sklearn.utils import check_random_state
from SPP import simu, safe_prune


def test_safe_prune():

    # Parameters definition
    rng = check_random_state(0)
    n_samples, n_features = 30, 4
    beta = rng.randn(n_features)
    encode = 'onehot'
    strategy = 'quantile'
    lmbda = 1.
    n_epochs = 1000
    n_bins = 3
    max_depth = 3

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False,
                random_state=rng)

    # Discretization of the features by binning strategy
    enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    X_binned = enc.fit_transform(X).tocsc()
    del X

    # Computation of the hyper-parameters : 
    sparse_lasso_sklearn = Lasso(alpha=(lmbda / X_binned.shape[0]),
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
    safe_sphere_center = theta

    # Computation of the safe set with recursive function
    X_binned_data = X_binned.data
    X_binned_indices = X_binned.indices
    X_binned_indptr = X_binned.indptr
    start_sp = time.time()
    (safe_set_data, safe_set_ind, safe_set_key, flatten_safe_set_data,
        flatten_safe_set_ind, safe_set_indptr) = safe_prune(
        X_binned_data=X_binned_data, X_binned_indices=X_binned_indices,
        X_binned_indptr=X_binned_indptr, safe_sphere_center=safe_sphere_center,
        safe_sphere_radius=safe_sphere_radius, max_depth=max_depth)

    end_sp = time.time()
    delay_sp = end_sp - start_sp
    print("delay safe prune = ", delay_sp)

    safe_set_key = list(map(tuple, map(sorted, safe_set_key)))

    # Computation of the active set with nested loop
    safe_set_key_test = []
    start_sp_test = time.time()
    n_features = X_binned.shape[1]
    for i in range(n_features):
        for j in range(i, n_features):
            for k in range(j, n_features):
                inter_feat = X_binned[:, i].multiply(
                    X_binned[:, j]).multiply(X_binned[:, k])
                u_t = inter_feat.T @ safe_sphere_center
                v_t = (inter_feat.data ** 2).sum()
                sppc_t = abs(u_t) + safe_sphere_radius * np.sqrt(v_t)
                key = tuple(sorted(set([i, j, k])))
                if sppc_t >= 1 and key not in safe_set_key_test:
                    safe_set_key_test.append(key)

    end_sp_test = time.time()
    delay_sp_test = end_sp_test - start_sp_test
    print("delay safe pruning test = ", delay_sp_test)

    assert len(safe_set_key_test) == len(safe_set_key)
    assert set(safe_set_key_test) == set(safe_set_key)
