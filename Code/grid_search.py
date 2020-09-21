import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.core import numeric
import pandas as pd

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from cd_solver_lasso_numba import Lasso
from dataset import (
    load_auto_prices,
    load_lacrimes,
    load_black_friday,
    load_nyc_taxi,
    load_housing_prices,
)
from SPP import SPPRegressor
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


def numeric_features(X):
    # numeric_feats = X.dtypes[(dtype.kind in "if" for dtype in X.dtypes)].index
    numeric_feats = X.dtypes[(X.dtypes == np.float64)
                             | (X.dtypes == np.int64)].index
    return numeric_feats


def categorical_features(X):
    categorical_feats = X.dtypes[
        ((X.dtypes == "object") | (X.dtypes == "category"))
    ].index
    return categorical_feats


def time_features(X):
    time_feats = X.dtypes[(X.dtypes == "datetime64[ns]")].index
    return time_feats


def time_convert(X):
    X = X.copy()

    for col, dtype in zip(X.columns, X.dtypes):
        if X[col].dtypes == "datetime64[ns]":
            col_name = str(col) + "_"
            X[col_name + "year"] = X[col].dt.year
            X[col_name + "weekday"] = X[col].dt.dayofweek
            X[col_name + "yearday"] = X[col].dt.dayofyear
            X[col_name + "time"] = X[col].dt.minute
            X = X.drop(columns=[col], axis=1)

    return X


def get_models(
    X,
    n_bins,
    kwargs_lasso,
    kwargs_spp
):
    """
    X: numpy.ndarray(), shape = (n_samples, n_features)
        features matrix

    n_lambda: int
        number of lambda parameters in the grid search

    lambdas: list of floats
        list of lambda parameters
        if it is None, then we create a log-scale grid of lambdas

    lambda_lasso: float
        regularization parameter for the Lasso

    n_bins: int
        number of bins in the binning process

    max_depth: int
        maximum order of interactions

    epsilon: float
        tolerance for the dual gap

    f: int
        frequency

    n_epochs: int
        number of iterations until convergence of the sparse_cd algorithm

    tol: float
        tolerance

    lambda_max_ratio: float
        lambda_max_ratio * lambda_max is the min value of the grid search


    n_active_max: int
        defines the maximum number of active features that we accept in the
        active set so that the latter might not be too large and that the
        resolution of the optimization problem might be tractable

    screening: bool, default = True
        defines whether or not one adds screening to the solver

    store_history: bool, default = True
        defines whether or not one stores the values of the parameters
        while the solver is running

    """

    # Pipeline

    time_feats = time_features(X)
    time_transformer = FunctionTransformer(time_convert)

    numeric_feats = numeric_features(X)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("binning",
                KBinsDiscretizer(n_bins=n_bins, encode="onehot",
                                 strategy="quantile"),
             ),
        ]
    )

    rf_numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_feats = categorical_features(X)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing"),),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_feats),
            ("cat", categorical_transformer, categorical_feats),
            ("time", time_transformer, time_feats),
        ]
    )

    rf_preprocessor = ColumnTransformer(
        transformers=[
            ("num", rf_numeric_transformer, numeric_feats),
            ("cat", categorical_transformer, categorical_feats),
            ("time", time_transformer, time_feats),
        ]
    )

    models = {}
    tuned_parameters = {}

    # Lasso
    lasso = Lasso(**kwargs_lasso)
    models["lasso"] = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", lasso)]
    )
    lmbdas = np.logspace(-4, -0.5, 3)
    tuned_parameters["lasso"] = {
        "regressor__lmbda": lmbdas,
        "preprocessor__num__binning__n_bins": [2, 3, 4],
    }

    # LassoCV
    models["lasso_cv"] = Pipeline(
        steps=[("preprocessor", preprocessor),
               ("regressor", linear_model.LassoCV()), ]
    )
    tuned_parameters["lasso_cv"] = {
        "preprocessor__num__binning__n_bins": [2, 3, 4]}

    # RidgeCV
    models["ridge_cv"] = Pipeline(
        steps=[("preprocessor", preprocessor),
               ("regressor", linear_model.RidgeCV()), ]
    )
    tuned_parameters["ridge_cv"] = {
        "preprocessor__num__binning__n_bins": [2, 3, 4]}

    # XGBoost
    xgb = XGBRegressor()
    models["xgb"] = Pipeline(
        steps=[("preprocessor", rf_preprocessor), ("regressor", xgb)]
    )
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    tuned_parameters["xgb"] = {
        "regressor__alpha": alphas,
        "regressor__n_estimators": [30, 100],
    }

    # Random Forest
    rf = RandomForestRegressor()
    models["rf"] = Pipeline(
        steps=[("preprocessor", rf_preprocessor), ("regressor", rf)]
    )
    tuned_parameters["rf"] = {"regressor__max_depth": [2, 3, 4, 5]}

    # SPP Regressor
    spp_reg = SPPRegressor(**kwargs_spp)
    models["spp_reg"] = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", spp_reg)]
    )

    tuned_parameters["spp_reg"] = {
        "preprocessor__num__binning__n_bins": [2, 3, 4],
        "regressor__max_depth": [2, 3],
    }

    return models, tuned_parameters


def compute_gs(
    X,
    y,
    models,
    tuned_parameters,
    n_splits,
    n_lambda,
    lambdas,
    max_depth,
    epsilon,
    f,
    n_epochs,
    tol,
    lambda_max_ratio,
    n_active_max,
    screening,
    store_history,
    n_jobs=1,
):
    """
    Parameters
    ----------
    X: numpy.ndarray(), shape = (n_samples, n_features)
        features matrix

    y: numpy.array(), shape = '(n_samples, )
        target vector

    models : dict
        dict of models

    tuned_parameters : dict
        dict of parameters to tune with grid-search

    n_splits: int
        number of folds

    n_lambda: int 
        number of lambda parameters in the grid search

    lambdas: list of floats
        list of lambda parameters 
        if it is None, then we create a log-scale grid of lambdas 

    max_depth: int 
        maximum order of interactions

    epsilon: float 
        dual gap tolerance 

    f: int
        frequency

    n_epochs: int
        number of iterations until convergence of the sparse_cd algorithm

    tol: float 
        tolerance

    lambda_max_ratio: float 
        lambda_max_ratio * lambda_max is the min value of the grid search

    n_active_max: int
        defines the maximum number of active features that we accept in the 
        active set so that the latter might not be too large and that the 
        resolution of the optimization problem might be tractable

    screening: bool, default = True
        defines whether or not one adds screening to the solver

    store_history: bool, default = True
        defines whether or not one stores the values of the parameters
        while the solver is running

    n_jobs: int
        number of jobs in parallel

    Returns
    -------
    cv_scores: dict
        cross validation scores for different models
    """
    y = np.array(y).astype("float")
    gs_models = {}

    for name, model in models.items():
        if name != "spp_reg":
            gs = GridSearchCV(
                model, cv=n_splits, param_grid=tuned_parameters[name], n_jobs=n_jobs,
                scoring='neg_mean_squared_error',
            )

            gs.fit(X, y)

            results_gs = {
                "best_score": gs.best_score_,
                "best_params": gs.best_params_,
            }

        else:
            kf = KFold(n_splits=n_splits, random_state=None, shuffle=False)

            gs_list = []

            for fold_num, (train_index, test_index) in enumerate(kf.split(X), 1):
                # print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]

                n_bins_list = tuned_parameters['spp_reg']['preprocessor__num__binning__n_bins']
                max_depth_list = tuned_parameters['spp_reg']['regressor__max_depth']

                for n_bins in n_bins_list:
                    for max_depth in max_depth_list:
                        spp_reg = model.set_params(preprocessor__num__binning__n_bins=n_bins, regressor__max_depth=max_depth)

                        # enc = KBinsDiscretizer(
                        #     n_bins=n_bins, encode='onehot', strategy='quantile')
                        # X_binned_train = enc.fit_transform(X_train)
                        # X_binned_test = enc.transform(X_test)

                        # solutions = spp_reg.fit(
                        #     X_binned_train, y_train).solutions_
                        # cv_scores = spp_reg.score(X_binned_test, y_test)

                        # solutions = spp_reg.fit(
                        #     X_train, y_train).solutions_
                        spp_reg.fit(X_train, y_train)
                        solutions = ...
                        cv_scores = spp_reg.score(X_test, y_test)

                        lambda_list = []
                        slopes_list = []
                        for this_sol in solutions:
                            lambda_list.append(this_sol["lambda"])
                            slopes_list.append(this_sol["spp_lasso_slopes"])

                        if type(cv_scores) is np.float64:
                            cv_scores = [cv_scores]

                        # import ipdb
                        # ipdb.set_trace()

                        results = {
                            "n_bins": [n_bins] * len(cv_scores),
                            "max_depth": [max_depth] * len(cv_scores),
                            "lambda": lambda_list,
                            "score": cv_scores,
                            "fold_number": [fold_num] * len(cv_scores),
                        }

                        gs_list.append(
                            pd.DataFrame(
                                results
                            )
                        )

            gs_dataframe = pd.concat(gs_list)

            gs_groupby_params = (
                gs_dataframe.groupby(
                    by=["n_bins", "max_depth", "lambda"])["score"]
                .mean()
                .reset_index()
            )

            best_params = gs_groupby_params.loc[
                gs_groupby_params["score"] == gs_groupby_params["score"].max(
                ), ["n_bins", "max_depth", "lambda"]]

            print('fold_num = ', fold_num)
            print('best_params = ', best_params)

            best_score = gs_groupby_params["score"].max()

            print('best_score = ', best_score)

            results_gs = {
                "best_score": best_score,
                "best_params": {'n_bins': best_params.iloc[0, 0].values,
                                'max_depth': best_params.iloc[0, 1].values,
                                'lambda': best_params.iloc[0, 2].values
                                },
            }

        gs_models[name] = results_gs

    return gs_models


def main():

    lmbda_lasso = 1.0
    epsilon = 1e-7
    f = 10
    n_splits = 2
    screening = True
    store_history = True
    n_epochs = 10000
    n_jobs = 1
    # encode = "onehot"
    # strategy = "quantile"
    n_bins = 3
    max_depth = 2
    tol = 1e-08
    n_lambda = 100
    lambda_max_ratio = 0.5
    lambdas = [1, 0.5, 0.2, 0.1, 0.01]
    n_active_max = 100

    kwargs_spp = {
        "n_lambda": n_lambda,
        "lambdas": lambdas,
        "max_depth": max_depth,
        "epsilon": epsilon,
        "f": f,
        "n_epochs": n_epochs,
        "tol": tol,
        "lambda_max_ratio": lambda_max_ratio,
        "n_active_max": n_active_max,
        "screening": screening,
        "store_history": store_history,
    }

    kwargs_lasso = {
        "lmbda": lmbda_lasso,
        "epsilon": epsilon,
        "f": f,
        "n_epochs": n_epochs,
        "screening": screening,
        "store_history": store_history,
    }

    datasets = [
        "auto_prices",
        "lacrimes",
        "black_friday",
        "nyc_taxi",
        "housing_prices",
    ]

    # for i in range(len(datasets)):
    #     if datasets[i] == "auto_prices":
    #         X, y = load_auto_prices()
    #         data_name = "auto_prices"
    #     if datasets[i] == "lacrimes":
    #         continue
    #         X, y = load_lacrimes()
    #         data_name = "lacrimes"
    #     if datasets[i] == "black_friday":
    #         continue
    #         X, y = load_black_friday()
    #         data_name = "black_friday"
    #     if datasets[i] == "nyc_taxi":
    #         continue
    #         X, y = load_nyc_taxi()
    #         data_name = "nyc_taxi"
    #     if datasets[i] == "housing_prices":
    #         continue
    #         load_housing_prices()
    #         data_name = "housing_prices"

    X, y = load_auto_prices()

    X = X[:1000]
    y = y[:1000]

    # models, tuned_parameters = get_models(
    #     X=X,
    #     n_lambda=n_lambda,
    #     lambdas=lambdas,
    #     lambda_lasso=lmbda_lasso,
    #     n_bins=n_bins,
    #     max_depth=max_depth,
    #     epsilon=epsilon,
    #     f=f,
    #     n_epochs=n_epochs,
    #     tol=tol,
    #     lambda_max_ratio=lambda_max_ratio,
    #     n_active_max=n_active_max,
    #     screening=screening,
    #     store_history=store_history,
    # )

    models, tuned_parameters = get_models(
        X=X,
        n_bins=n_bins,
        kwargs_lasso=kwargs_lasso,
        kwargs_spp=kwargs_spp
    )

    del models['lasso']
    del models['lasso_cv']
    del models['ridge_cv']
    del models['rf']
    del models['xgb']

    del tuned_parameters['lasso']
    del tuned_parameters['lasso_cv']
    del tuned_parameters['ridge_cv']
    del tuned_parameters['rf']
    del tuned_parameters['xgb']

    gs_models = compute_gs(
        X=X,
        y=y,
        models=models,
        tuned_parameters=tuned_parameters,
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
        n_jobs=n_jobs,
    )

    print("gs_models = ", gs_models)
    best_score_spp = gs_models["spp_reg"]["best_score"]
    best_params_spp = gs_models["spp_reg"]["best_params"]

    print('best_score_spp = ', best_score_spp)
    print('best_params = ', best_params_spp)


if __name__ == "__main__":
    main()
