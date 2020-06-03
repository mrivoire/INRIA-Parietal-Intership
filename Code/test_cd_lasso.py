import numpy as np

from cd_solver_lasso import simu, cyclic_coordinate_descent
from sklearn.linear_model import Lasso as sklearn_Lasso


def test_cd_lasso():
    # Data Simulation
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 30
    beta = rng.randn(n_features)
    lmbda = 0.1

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False)

    epsilon = 1e-14
    f = 10

    (beta_hat_cyclic_cd_true,
        primal_hist,
        dual_hist,
        gap_hist,
        r_list,
        n_active_features_true,
        objs_cyclic_cd,
        theta_hat_cyclic_cd,
        P_lmbda,
        D_lmbda,
        G_lmbda) = cyclic_coordinate_descent(X,
                                             y,
                                             lmbda,
                                             epsilon,
                                             f,
                                             n_epochs=100000,
                                             screening=True)

    # KKT conditions
    kkt = np.abs(np.dot(X.T, y - np.dot(X, beta_hat_cyclic_cd_true)))
    # Sklearn's Lasso
    lasso = sklearn_Lasso(alpha=lmbda / len(X), fit_intercept=False,
                          normalize=False, tol=1e-10).fit(X, y)
    assert G_lmbda < 1e-11
    assert kkt.all() <= 1
    assert r_list[-1] <= 1
    np.testing.assert_allclose(beta_hat_cyclic_cd_true, lasso.coef_, rtol=1)
