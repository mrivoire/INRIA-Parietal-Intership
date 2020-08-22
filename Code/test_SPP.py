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
from SPP import simu, compute_inner_prod, SPP 


def test_SPP():

    # Definition of the parameters 
    rng = check_random_state(0)
    n_samples, n_features = 20, 2
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

    # Lasso 
    sparse_lasso_sklearn = sklearn_Lasso(alpha=(lmbda / X_binned.shape[0]),
                                         fit_intercept=False,
                                         normalize=False,
                                         max_iter=n_epochs,
                                         tol=1e-14).fit(X_binned, y)

    beta_star = sparse_lasso_sklearn.coef_
    # residuals = y - X_binned.dot(beta_star)

    beta_hat_t, safe_set_data, safe_set_ind, safe_set_key = SPP(
        X_binned=X_binned, X_binned_data=X_binned_data,
        X_binned_indices=X_binned_indices, X_binned_indptr=X_binned_indptr,
        y=y,
        n_val_gs=n_val_gs, max_depth=max_depth, epsilon=epsilon, f=f,
        n_epochs=n_epochs, screening=screening, store_history=store_history)

    print("beta_hat_t =", beta_hat_t)

    assert beta_hat_t == beta_star
    assert len(beta_hat_t) == len(beta_star)