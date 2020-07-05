import numpy as np
import time

from cd_solver_lasso_numba import simu, Lasso
from sklearn.linear_model import Lasso as sklearn_Lasso
from scipy.sparse import csc_matrix


# Data Simulation
rng = np.random.RandomState(0)
n_samples, n_features = 500, 3000
beta = rng.randn(n_features)
beta[np.sort(np.abs(beta))[-30] > np.abs(beta)] = 0.0
lmbda = 0.1

X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False, random_state=42)

y /= np.linalg.norm(y)

f = 10
n_epochs = 1000
sparse = False
tol = 1e-15

if sparse:
    X = csc_matrix(X)

lasso = Lasso(
    lmbda=lmbda,
    epsilon=tol,
    f=f,
    n_epochs=n_epochs,
    screening=True,
    store_history=False,
)
lasso.fit(X[:, :2], y)  # compile numba code

start1 = time.time()
lasso.fit(X, y)
end1 = time.time()

delay1 = end1 - start1
print(delay1)

start2 = time.time()
sklasso = sklearn_Lasso(
    alpha=lmbda / X.shape[0],
    fit_intercept=False,
    normalize=False,
    max_iter=n_epochs,
    tol=tol,
).fit(X, y)

end2 = time.time()
delay2 = end2 - start2
print(delay2)


primal_function_sklearn = 0.5 * np.linalg.norm(
    y - X.dot(sklasso.coef_.T), 2
) ** 2 + lmbda * np.linalg.norm(sklasso.coef_, 1)
primal_function_numba = 0.5 * np.linalg.norm(
    y - X.dot(lasso.slopes.T), 2
) ** 2 + lmbda * np.linalg.norm(lasso.slopes, 1)
print(primal_function_sklearn - primal_function_numba)
