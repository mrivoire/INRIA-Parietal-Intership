import numpy as np

from cd_solver_lasso import simu, cyclic_coordinate_descent, radius


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
        cyclic_coordinate_descent(X, y, lmbda, epsilon, f, n_epochs=1000,
                                  screening=True)

    assert r_list[-1] < 1
