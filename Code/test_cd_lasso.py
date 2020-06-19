import pytest
import numpy as np

from cd_solver_lasso_numba import simu, Lasso
from sklearn.linear_model import Lasso as sklearn_Lasso
from scipy.sparse import csc_matrix


@pytest.mark.parametrize('screening', [True, False])
@pytest.mark.parametrize('store_history', [True, False])
@pytest.mark.parametrize('sparse', [True, False])
def test_cd_lasso(screening, store_history, sparse):
    # Data Simulation
    rng = np.random.RandomState(0)
    n_samples, n_features = 20, 30
    beta = rng.randn(n_features)
    lmbda = 1.

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False)

    epsilon = 1e-14
    f = 10
    n_epochs = 100000

    if sparse:
        X = csc_matrix(X)

    lasso = Lasso(lmbda=lmbda, epsilon=epsilon, f=f, n_epochs=n_epochs,
                  screening=True, store_history=True)

    lasso.fit(X, y)

    # (beta_hat, primal_hist, dual_hist, gap_hist, r_list,
    #  n_active_features_true, theta_hat_cyclic_cd, P_lmbda, D_lmbda,
    #  G_lmbda) = out

    # KKT conditions
    kkt = np.abs(X.T.dot(y - X.dot(lasso.slopes)))
    # Sklearn's Lasso
    sklasso = sklearn_Lasso(alpha=lmbda / X.shape[0], fit_intercept=False,
                            normalize=False,
                            max_iter=n_epochs, tol=1e-15).fit(X, y)
    # Tests
    assert lasso.G_lmbda < 1e-11
    assert kkt.all() <= 1
    if screening and store_history:
        assert lasso.r_list[-1] < 1e-5

    np.testing.assert_allclose(lasso.slopes, sklasso.coef_, rtol=1)
