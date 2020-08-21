import numpy as np
import time

from scipy.linalg import toeplitz
from numba import njit
from numba.typed import List
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import Lasso as sklearn_Lasso
from sklearn.utils import check_random_state
from cd_solver_lasso_numba import Lasso, sparse_cd
from SPP import simu, max_val


def test_max_val():

    # Parameters definition
    rng = check_random_state(0)
    n_samples, n_features = 100, 40
    beta = rng.randn(n_features)
    encode = 'onehot'
    strategy = 'quantile'
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

    # Computation of the residuals
    residuals = rng.randn(n_samples)

    # Computation of the max_val with nested loop
    max_val_test = 0
    X_binned = X_binned.toarray()
    start1 = time.time()
    for j in range(n_features):
        for k in range(j, n_features):
            for l in range(k, n_features):
                inter_feat = (X_binned[:, j]
                              * X_binned[:, k]
                              * X_binned[:, l])

                inner_prod = inter_feat.dot(residuals)
                if abs(inner_prod) > max_val_test:
                    max_val_test = abs(inner_prod)
                    key = [j, k, l]
                    print("key = ", key)

    end1 = time.time()
    delay1 = end1 - start1
    print("delay 1 = ", delay1)
    print("max val test = ", max_val_test)

    # Computation of the max_val with recursive function
    start2 = time.time()
    max_inner_prod, max_key = max_val(X_binned_data=X_binned_data,
                                      X_binned_indices=X_binned_indices,
                                      X_binned_indptr=X_binned_indptr,
                                      residuals=residuals,
                                      max_depth=max_depth)

    max_key = list(max_key)
    end2 = time.time()
    delay2 = end2 - start2
    print("delay 2 = ", delay2)
    print("max inner prod = ", max_inner_prod)
    print("max key= ", max_key)

    np.testing.assert_allclose(max_val_test, max_inner_prod, rtol=1e-14)
    if key[0] == key[1] and key[0] == key[2]:
        assert key[0] == max_key[0]
    else:
        assert key == max_key