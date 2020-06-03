import numpy as np

from cd_solver_lasso import simu, cyclic_coordinate_descent
from sklearn.linear_model import Lasso as sklearn_Lasso


def test_cd_lasso_gap():
    # Data Simulation
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 30
    beta = rng.randn(n_features)
    lmbda = 0.1

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False)

    epsilon = 1e-14
    f = 10

    (beta_hat_cyclic_cd_false,
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
                                             n_epochs=1000000,
                                             screening=False)

    assert G_lmbda < 1e-11


def test_KKT_conditions():
    """Test whether the KKT conditions are satisfied or not

    Parameters
    ----------
    parameters are given by the @pytest

    Returns
    -------
    None
    """

    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 30
    beta = rng.randn(n_features)
    epsilon = 10**(-14)
    lmbda = 0.1
    f = 10

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False)

    (beta_hat_cyclic_cd_true, _, _, _, _, _, _, _, _, _, _) = \
        cyclic_coordinate_descent(X, y, lmbda, epsilon, f, n_epochs=1000000,
                                  screening=True)

    kkt = abs(np.dot(X.T, y - np.dot(X, beta_hat_cyclic_cd_true)))

    np.testing.assert_allclose(kkt, lmbda, rtol=1)


def test_radius_convergence():
    """Test whether the radius converges towards 0

    Parameters
    ----------
    parameters are given by the @pytest

    Returns
    -------
    None
    """
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 30
    beta = rng.randn(n_features)
    epsilon = 10**(-14)
    lmbda = 0.1
    f = 10

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False)

    (_, _, _, _, r_list, _, _, _, _, _, G_lmbda) = \
        cyclic_coordinate_descent(X, y, lmbda, epsilon, f, n_epochs=1000000,
                                  screening=True)

    assert r_list[-1] <= 1


def test_lasso():
    """Test that our Lasso solver behaves as sklearn's Lasso

    Parameters
    ----------
    parameters are given thanks to @pytest

    Returns
    -------
    None
    """
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 30
    beta = rng.randn(n_features)
    epsilon = 10**(-14)
    lmbda = 0.1
    f = 10

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False)

    # Sklearn's Lasso
    lasso = sklearn_Lasso(alpha=lmbda / len(X), fit_intercept=False,
                          normalize=False, tol=1e-10).fit(X, y)

    # Our Lasso
    (beta_hat_cyclic_cd_true, _, _, _, _, _, _, _, _, _, _) = \
        cyclic_coordinate_descent(X, y, lmbda, epsilon, f, n_epochs=1000000,
                                  screening=True)

    np.testing.assert_allclose(beta_hat_cyclic_cd_true, lasso.coef_, rtol=1)
