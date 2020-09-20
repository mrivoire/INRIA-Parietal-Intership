import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import KBinsDiscretizer

from checkerboard import checkerboard
from tests_real_data import get_models, compute_gs


def test_crossval_spp():
    # Paramters
    dim1 = 10
    dim2 = 10
    n_bins = 3
    n_samples = 3000

    lmbda = 0.1
    epsilon = 1e-7
    f = 10
    n_splits = 2

    screening = True
    store_history = True
    n_epochs = 1000
    n_jobs = 1
    max_depth = 2
    tol = 1e-8
    n_lambda = 100
    lambda_max_ratio = 0.5
    lambdas = [1, 0.5, 0.2, 0.1, 0.01]
    n_active_max = 100

    X, y = checkerboard(dim1=dim1, dim2=dim2, n_samples=n_samples,
                        n_bins=n_bins)
    X = pd.DataFrame(X)

    models, tuned_params = get_models(X=X,
                                      n_lambda=n_lambda,
                                      lambdas=lambdas,
                                      lambda_lasso=lmbda,
                                      n_bins=n_bins,
                                      max_depth=max_depth,
                                      epsilon=epsilon,
                                      f=f,
                                      n_epochs=n_epochs,
                                      tol=tol,
                                      lambda_max_ratio=lambda_max_ratio,
                                      n_active_max=n_active_max,
                                      screening=screening,
                                      store_history=store_history)

    # Save time
    del models['lasso']
    del models['lasso_cv']
    del models['ridge_cv']
    del models['rf']
    del models['xgb']

    print('models = ', models)
    print('tuned_params = ', tuned_params)
    # How to use the best max_depth to perform the binning of X_train whereas
    # it is returned by the compute_gs function ?
    # We need to use get_models befor gs_models but we need to have the
    # best_max_depth to run get_models to perform a good binning process
    # How to handle that ?
    gs_models = compute_gs(X=X,
                           y=y,
                           models=models,
                           tuned_parameters=tuned_params,
                           n_splits=n_splits,
                           n_lambda=n_lambda,
                           lambdas=lambdas,
                           max_depth=max_depth,
                           epsilon=epsilon,
                           f=f,
                           n_epochs=n_epochs,
                           tol=tol,
                           lambda_max_ratio=lambda_max_ratio,
                           n_active_max=n_active_max,
                           screening=screening,
                           store_history=store_history,
                           n_jobs=n_jobs)

    best_score_spp = gs_models['spp_reg']['best_score']
    best_params_spp = gs_models['spp_reg']['best_params']

    best_n_bins = best_params_spp['n_bins']
    best_max_depth = best_params_spp['max_depth']
    best_lambda_spp = best_params_spp['lambda']

    # order = max_depth
    # To compare the lambda values we have to run the polynomial features on a
    # number of bins and of max_depth equal to the best_n_bins and
    # best_max_depth provided by the grid search of spp
    enc = KBinsDiscretizer(
        n_bins=best_n_bins, encode='onehot', strategy='quantile')
    X_binned = enc.fit_transform(X)

    poly = PolynomialFeatures(order=best_max_depth,
                              include_bias=False,
                              interaction_only=True)
    X_poly = poly.fit_transform(X_binned)

    n_samples = X_binned.shape[0]
    alphas_lassoCV = np.array(lambdas) / n_samples

    reg_lassoCV = LassoCV(alphas=alphas_lassoCV,
                          fit_intercept=False,
                          normalize=False,
                          max_iter=n_epochs,
                          tol=tol,
                          cv=n_splits,
                          verbose=False).fit(X_poly, y)

    best_alpha_lassoCV = reg_lassoCV.alpha_
    best_lambda_lassoCV = best_alpha_lassoCV * n_samples

    mse_per_lambda_lassoCV = reg_lassoCV.mse_path_.mean(axis=1)
    best_mse_LassoCV = -mse_per_lambda_lassoCV.min()

    print('best_lambda_lassoCV = ', best_lambda_lassoCV)
    print('best_lambda_spp = ', best_lambda_spp)
    print('score_lassoCV = ', best_mse_LassoCV)
    print('best_score_spp = ', best_score_spp)

    assert best_lambda_lassoCV == best_lambda_spp
    assert n_bins == best_n_bins
    assert best_mse_LassoCV == pytest.approx(best_score_spp, abs=1e-2)


def main():

    # # Paramters
    # dim1 = 10
    # dim2 = 10
    # n_bins = 5
    # n_samples = 1000
    # lmbda = 0.1
    # epsilon = 1e-7
    # f = 10
    # n_splits = 5
    # screening = True
    # store_history = True
    # n_epochs = 1000
    # n_jobs = 1
    # encode = "onehot"
    # strategy = "quantile"
    # n_bins = 3
    # max_depth = 2
    # n_val_gs = 100
    # tol = 1e-08
    # n_lambda = 100
    # lambda_max_ratio = 0.5
    # lambdas = [0.5]
    # n_active_max = 100

    # X, y = checkerboard(dim1=dim1, dim2=dim2, n_samples=n_samples,
    #                     n_bins=n_bins)

    # X_train, X_test, y_train, y_test = \
    #     train_test_split(X, y, test_size=.4, random_state=42)

    # X_train = pd.DataFrame(X_train)
    # X_test = pd.DataFrame(X_test)
    # print('X_train = ', X_train)
    # print('X_test = ', X_test)

    # cat_feats = categorical_features(X_train)
    # print('cat_feats = ', cat_feats)

    # num_feats = numeric_features(X_train)
    # print('num_feats = ', num_feats)
    # num_feats_test = numeric_features(X_test)
    # print('num_feats_test = ', num_feats_test)

    test_crossval_spp()


if __name__ == "__main__":
    main()
