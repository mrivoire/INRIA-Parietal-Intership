import numpy as np
from numpy import random

from scipy.linalg import toeplitz
from numba import njit
from numba.typed import List
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.utils import check_random_state
from cd_solver_lasso_numba import sparse_cd
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import BaggingRegressor
from SPP import SPPRegressor, simu


class SPPBaggingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_lambda, lambdas, max_depth, epsilon, f, n_epochs, tol,
                 lambda_max_ratio, n_active_max, screening, store_history,
                 n_estimators, random_state):

        self.n_lambda = n_lambda
        self.lambdas = lambdas
        self.max_depth = max_depth
        self.epsilon = epsilon
        self.f = f
        self.n_epochs = n_epochs
        self.tol = tol
        self.lambda_max_ratio = lambda_max_ratio
        self.n_active_max = n_active_max
        self.screening = screening
        self.store_history = store_history
        self.n_estimators = n_estimators
        self.random_state = random_state

        assert epsilon > 0
        assert tol > 0 and tol < 1e-01

    def fit(self, X_binned, y):
        """
        """

        y = np.array(y, dtype=float)
        spp_reg = SPPRegressor(n_lambda=self.n_lambda,
                               lambdas=self.lambdas,
                               max_depth=self.max_depth,
                               epsilon=self.epsilon, f=self.f,
                               n_epochs=self.n_epochs, tol=self.tol,
                               lambda_max_ratio=self.lambda_max_ratio,
                               n_active_max=self.n_active_max,
                               screening=self.screening,
                               store_history=self.store_history)

        bagging_reg = BaggingRegressor(
            base_estimator=spp_reg, n_estimators=self.n_estimators,
            random_state=self.random_state).fit(X_binned, y)

        self.bagging_reg_ = bagging_reg

        return self

    def predict(self, X_binned):
        """
        """
        # y_pred = np.mean([est.predict(X)
        #                   for est in self.baggingregressor.estimators_], axis=0)
        y_hats = np.mean([est.predict(X_binned)
                          for est in
                          self.bagging_reg_.estimators_], axis=0)

        return y_hats

    def score(self, X, y):
        """
        """
        y_hats = self.predict(X)
        if y_hats.ndim == 1:
            y_hats = y_hats[:, None]

        scores = []
        for y_hat in y_hats.T:
            scores.append(-mean_squared_error(y, y_hat))

        if len(scores) == 1:
            scores = scores[0]

        return scores


def main():
    # Parameters
    rng = check_random_state(2)
    n_samples, n_features = 100, 10
    beta = rng.randn(n_features)
    f = 10
    epsilon = 1e-14
    n_epochs = 100000
    screening = True
    store_history = True
    encode = 'onehot'
    strategy = 'quantile'
    n_bins = 3
    max_depth = 2
    n_lambda = 5
    # if n_lmbda is none otherwise list lmbdas
    tol = 1e-08
    lambda_max_ratio = 0.5
    n_active_max = 100
    store_history = False
    lambdas = None
    random_state = 0
    n_estimators = 10

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False,
                random_state=rng)

    # Discretization by binning strategy
    enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    X_binned = enc.fit_transform(X)
    X_binned = X_binned.tocsc()

    reg = SPPBaggingRegressor(n_lambda=n_lambda,
                              lambdas=lambdas,
                              max_depth=max_depth,
                              epsilon=epsilon, f=f,
                              n_epochs=n_epochs, tol=tol,
                              lambda_max_ratio=lambda_max_ratio,
                              n_active_max=n_active_max,
                              screening=screening,
                              store_history=store_history,
                              n_estimators=n_estimators,
                              random_state=random_state)

    spp_bagging_reg = reg.fit(X_binned=X_binned, y=y)

    print('sol = ', spp_bagging_reg)

    y_hats = spp_bagging_reg.predict(X_binned=X_binned)

    print('pred = ', y_hats)

    scores = spp_bagging_reg.score(X=X_binned, y=y)
    print('scores = ', scores)


if __name__ == "__main__":
    main()
