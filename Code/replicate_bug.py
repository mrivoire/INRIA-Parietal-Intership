import numpy as np
from scipy import sparse
from SPP import spp_solver

X_binned = sparse.load_npz('X_binned.npz')
y = np.load('y.npy')
n_lambda = 10
lambdas = None
max_depth = 2
epsilon = 1e-7
f = 10
n_epochs = 10000
tol = 1e-8
lambda_max_ratio = 1 / 20.
n_active_max = 100

spp_solver(X_binned, y,
           n_lambda, lambdas, max_depth, epsilon, f, n_epochs, tol,
           lambda_max_ratio, n_active_max, screening=True,
           store_history=True)
