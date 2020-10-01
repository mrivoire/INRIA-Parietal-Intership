import numpy as np
from scipy import sparse
# from cd_solver_lasso_numba import sparse_cd
from SPP import spp_solver

epsilon = 1e-7
f = 10
screening = True
store_history = True
n_epochs = 10000
n_jobs = 1
max_depth = 2
tol = 1e-08
n_lambda = 10
lambda_max_ratio = (1 / 20)
n_active_max = 100

X_binned = sparse.load_npz('X_binned.npz')
y = np.load('y.npy')

spp_solver(X_binned, y,
           n_lambda, lambdas=None, max_depth=max_depth,
           epsilon=epsilon, f=f, n_epochs=n_epochs, tol=tol,
           lambda_max_ratio=lambda_max_ratio,
           n_active_max=n_active_max,
           screening=True,
           store_history=True)

# (beta_hat_t, residuals, primal_hist_sparse, dual_hist_sparse,
#  gap_hist_sparse, r_list_sparse, n_active_features_true_sparse,
#  theta_hat_cyclic_cd_sparse, P_lmbda_sparse, D_lmbda_sparse,
#  G_lmbda_sparse, safe_set_sparse) = \
#     sparse_cd(X_data=X_binned_data,
#               X_indices=X_binned_ind,
#               X_indptr=X_binned_indptr, y=y, lmbda=lmbda,
#               epsilon=epsilon, f=f, n_epochs=n_epochs,
#               screening=screening, store_history=store_history)
