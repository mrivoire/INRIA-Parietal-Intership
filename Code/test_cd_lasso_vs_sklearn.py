import numpy as np

from cd_solver_lasso import simu, cyclic_coordinate_descent
from sklearn.linear_model import Lasso as sklearn_Lasso


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
        cyclic_coordinate_descent(X, y, lmbda, epsilon, f, n_epochs=10000,
                                  screening=True)

    np.testing.assert_allclose(beta_hat_cyclic_cd_true, lasso.coef_, rtol=1e-05)
