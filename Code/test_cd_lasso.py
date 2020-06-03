import numpy as np

from cd_solver_lasso import simu, cyclic_coordinate_descent


def test_cd_lasso_kkt():
    # Data Simulation
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 30
    beta = rng.randn(n_features)
    lmbda = 0.1

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False)

    epsilon = 1e-14
    f = 10

    (beta_hat_cyclic_cd_false,
        A_C_hist_false,
        primal_hist,
        dual_hist,
        gap_hist,
        theta_hist_false,
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
                                             n_epochs=4000,
                                             screening=False)

    assert gap_hist[-1] < 1e-12
