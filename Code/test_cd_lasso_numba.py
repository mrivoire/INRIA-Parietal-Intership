import pytest
import numpy as np

from cd_solver_lasso_numba import simu, cyclic_coordinate_descent
from sklearn.linear_model import Lasso as sklearn_Lasso


@pytest.mark.parametrize('screening', [True, False])
def test_cd_lasso(screening):
    # Data Simulation
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 30
    beta = rng.randn(n_features)
    lmbda = 1.

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False)

    epsilon = 1e-14
    f = 10
    n_epochs = 1000000

    beta_hat, theta_hat, all_objs = \
        cyclic_coordinate_descent(X,
                                  y,
                                  lmbda,
                                  epsilon,
                                  f,
                                  n_epochs,
                                  screening=True)

    # KKT conditions
    residuals = y - np.dot(X, beta_hat)
    kkt = np.abs(np.dot(X.T, residuals))
    # Sklearn's Lasso
    lasso = sklearn_Lasso(alpha=lmbda / len(X), fit_intercept=False,
                          normalize=False,
                          max_iter=n_epochs, tol=1e-15).fit(X, y)

    P_lmbda = 0.5*residuals.dot(residuals)
    P_lmbda += lmbda * np.linalg.norm(beta_hat, 1)
    D_lmbda = 0.5*np.linalg.norm(y, ord=2)**2
    D_lmbda -= (((lmbda**2) / 2)
                * np.linalg.norm(theta_hat - y / lmbda, ord=2)**2)
    G_lmbda = P_lmbda - D_lmbda

    assert G_lmbda < 1e-11
    assert kkt.all() <= 1
    if screening:
        r = np.sqrt(2*np.abs(G_lmbda))/lmbda
        assert r < 1e-5

    np.testing.assert_allclose(beta_hat, lasso.coef_, rtol=1)
