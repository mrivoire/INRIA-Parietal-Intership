import pytest
import numpy as np

from cd_solver_lasso_numba import simu, cyclic_coordinate_descent
from sklearn.linear_model import Lasso as sklearn_Lasso


@pytest.mark.parametrize('screening', [True, False])
# @pytest.mark.parametrize('store_history', [True, False])
def test_cd_lasso(screening, store_history=True):
    # Data Simulation
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 30
    beta = rng.randn(n_features)
    lmbda = 1.

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False)

    epsilon = 1e-14
    f = 10
    n_epochs = 100000

    (beta_hat_cyclic_cd_true,
        primal_hist,
        dual_hist,
        gap_hist,
        r_list,
        n_active_features_true,
        theta_hat_cyclic_cd,
        P_lmbda,
        D_lmbda,
        G_lmbda) = cyclic_coordinate_descent(X,
                                             y,
                                             lmbda,
                                             epsilon,
                                             f,
                                             n_epochs=n_epochs,
                                             screening=screening,
                                             store_history=True)

    # KKT conditions
    kkt = np.abs(np.dot(X.T, y - np.dot(X, beta_hat_cyclic_cd_true)))
    # Sklearn's Lasso
    lasso = sklearn_Lasso(alpha=lmbda / len(X), fit_intercept=False,
                          normalize=False,
                          max_iter=n_epochs, tol=1e-15).fit(X, y)
    # Tests 
    assert G_lmbda < 1e-11
    assert kkt.all() <= 1
    if screening:
        assert r_list[-1] < 1e-5

    np.testing.assert_allclose(beta_hat_cyclic_cd_true, lasso.coef_, rtol=1)
