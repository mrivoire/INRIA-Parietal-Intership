import numpy as np

from cd_solver_lasso import simu, cyclic_coordinate_descent


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
        cyclic_coordinate_descent(X, y, lmbda, epsilon, f, n_epochs=1000,
                                  screening=True)

    kkt = abs(np.dot(X.T, y - np.dot(X, beta_hat_cyclic_cd_true)))

    np.testing.assert_allclose(kkt, lmbda, rtol=1)
