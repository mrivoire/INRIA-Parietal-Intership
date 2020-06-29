import pytest
import numpy as np
import time 

from cd_solver_lasso_numba import simu, Lasso
from sklearn.linear_model import Lasso as sklearn_Lasso
from scipy.sparse import csc_matrix


@pytest.mark.parametrize('screening', [True, False])
@pytest.mark.parametrize('store_history', [True, False])
@pytest.mark.parametrize('sparse', [True, False])
def test_cd_lasso(screening, store_history, sparse):
    # Data Simulation
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 30
    beta = rng.randn(n_features)
    lmbda = 1.

    X, y = simu(beta, n_samples=n_samples, corr=0.5,
                for_logreg=False, random_state=42)

    epsilon = 1e-14
    f = 10
    n_epochs = 100000

    if sparse:
        X = csc_matrix(X)

    start1 = time.time()

    lasso = Lasso(lmbda=lmbda, epsilon=epsilon, f=f, n_epochs=n_epochs,
                  screening=True, store_history=True)

    lasso.fit(X, y)

    end1 = time.time()
    delay1 = end1 - start1

    # KKT conditions
    kkt = np.abs(X.T.dot(y - X.dot(lasso.slopes)))
    # Sklearn's Lasso

    start2 = time.time()
    sklasso = sklearn_Lasso(alpha=lmbda / X.shape[0], fit_intercept=False,
                            normalize=False,
                            max_iter=n_epochs, tol=1e-15).fit(X, y)

    end2 = time.time()
    delay2 = end2 - start2

    primal_function_sklearn = ((1 / (2 * n_samples)) 
                               * np.linalg.norm(y - X.dot(sklasso.coef_.T), 
                                                2)**2 
                               + lmbda * np.linalg.norm(sklasso.coef_, 1))

    # Tests
    assert lasso.G_lmbda < 1e-11
    assert kkt.all() <= 1
    if screening and store_history:
        assert lasso.r_list[-1] < 1e-5

    assert len(lasso.A_c) == np.count_nonzero(lasso.slopes)

    np.testing.assert_allclose(lasso.slopes, sklasso.coef_, rtol=1)
    np.testing.assert_allclose(lasso.P_lmbda, primal_function_sklearn, 
                               rtol=1e-14)
    np.testing.assert_allclose(delay1, delay2, rtol=1e-3)

    print("beta hat : ", lasso.slopes)
    print("coef sklearn : ", sklasso.coef_)
