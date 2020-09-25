import numpy as np
from cd_solver_lasso_numba import sparse_cd
from dataset import load_nyc_taxi
from grid_search import get_models
from sklearn.preprocessing import KBinsDiscretizer


lmbda = 0.1
epsilon = 1e-7
f = 10
n_splits = 2
screening = True
store_history = True
n_epochs = 10000
n_jobs = 1
encode = "onehot"
strategy = "quantile"
n_bins = 3
max_depth = 2
tol = 1e-08
n_lambda = 10
lambda_max_ratio = (1 / 20)
# lambdas = [1, 0.5, 0.2, 0.1, 0.01]
lambdas = None
# lambdas = [0.1]
n_active_max = 100

X, y = load_nyc_taxi()

enc = KBinsDiscretizer(n_bins=3, encode=encode, strategy=strategy)
X_binned = enc.fit_transform(X)
X_binned = X_binned.tocsc()
X_binned_data = X_binned.data
X_binned_ind = X_binned.ind
X_binned_indptr = X_binned.indptr

X = X[:1000]
y = y[:1000]

(beta_hat_t, residuals, primal_hist_sparse, dual_hist_sparse,
 gap_hist_sparse, r_list_sparse, n_active_features_true_sparse,
 theta_hat_cyclic_cd_sparse, P_lmbda_sparse, D_lmbda_sparse,
 G_lmbda_sparse, safe_set_sparse) = \
    sparse_cd(X_data=X_binned_data,
              X_indices=X_binned_ind,
              X_indptr=X_binned_indptr, y=y, lmbda=lmbda,
              epsilon=epsilon, f=f, n_epochs=n_epochs,
              screening=screening, store_history=store_history)
