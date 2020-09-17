import numpy as np
import pandas as pd 

from checkerboard import checkerboard
from sklearn.model_selection import train_test_split
from tests_real_data import get_models, compute_gs
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV


def test_crossval_spp():
    # Paramters 
    dim1 = 10
    dim2 = 10
    n_bins = 5
    n_samples = 1000
    lmbda = 0.1
    epsilon = 1e-7
    f = 10
    n_splits = 5
    screening = True
    store_history = True
    n_epochs = 1000
    n_jobs = 1
    encode = "onehot"
    strategy = "quantile"
    n_bins = 3
    max_depth = 2
    n_val_gs = 100
    tol = 1e-08
    n_lambda = 100
    lambda_max_ratio = 0.5
    lambdas = [0.5]
    n_active_max = 100

    X, y = checkerboard(dim1=dim1, dim2=dim2, n_samples=n_samples,
                        n_bins=n_bins)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    models, tuned_params = get_models(X=X_train, 
                                      n_lambda=n_lambda, 
                                      lambdas=None, 
                                      lambda_lasso=lmbda, 
                                      max_depth=max_depth, 
                                      epsilon=epsilon, 
                                      f=f, 
                                      n_epochs=n_epochs, 
                                      tol=tol, 
                                      lambda_max_ratio=lambda_max_ratio, 
                                      n_active_max=n_active_max, 
                                      screening=screening, 
                                      store_history=store_history)

    print('models = ', models)
    print('tuned_params = ', tuned_params)

    gs_models = compute_gs(X=X_train,
                           y=y_train, 
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
    best_coefs_spp = gs_models['spp_reg']['best_coefs']
    pred_spp = gs_models['spp_reg']['best_pred']

    poly = PolynomialFeatures(order=2, 
                              include_bias=False, 
                              interaction_only=True)

    poly_train = poly.fit_transform(X_train)
    poly_test = poly.fit_transform(X_test)

    reg_lassoCV = LassoCV(eps=0.001, 
                          n_alphas=100, 
                          alphas=None, 
                          fit_intercept=False, 
                          normalize=False, 
                          precompute='auto', 
                          max_iter=n_epochs, 
                          tol=tol, 
                          copy_X=True, 
                          cv=n_splits, 
                          verbose=False, 
                          n_jobs=None, 
                          positive=False, 
                          random_state=None, 
                          selection='cyclic').fit(poly_train, y_train)

    score_lassoCV = reg_lassoCV.score(X=poly_test, y=y_test, sample_weight=None)
    pred_lassoCV = reg_lassoCV.predict(X=poly_test)
    best_coefs_lassoCV = reg_lassoCV.coef_
    best_alpha_lassoCV = reg_lassoCV.alpha_

    np.testing.assert_allclose(best_alpha_lassoCV, best_params_spp[2], rtol=1e-8)
    np.testing.assert_allclose(best_score_spp, score_lassoCV, rtol=1e-8)
    np.testing.assert_allclose(best_coefs_lassoCV, best_coefs_spp, rtol=1e-8)
    np.testing.assert_allclose(pred_lassoCV, pred_spp, rtol=1e-8)


def main():

    test_crossval_spp()


if __name__ == "__main__":
    main()