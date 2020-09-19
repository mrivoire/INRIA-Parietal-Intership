"""
Experiments on real data with categorical variables

"""
import numpy as np
import time
import matplotlib.pyplot as plt
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

# from pandas_profiling import ProfileReport

#################################################
#          Preprocess Numeric Features
#################################################


def numeric_features(X):
    numeric_feats = X.dtypes[(dtype.kind in "if" for dtype in X.dtypes)].index
    return numeric_feats


##################################################
#          Preprocess Categorical Features
##################################################


def categorical_features(X):
    categorical_feats = X.dtypes[
        ((X.dtypes == "object") | (X.dtypes == "category"))
    ].index
    return categorical_feats


##################################################
#          Preprocess Time Features
##################################################


def time_features(X):
    time_feats = X.dtypes[(X.dtypes == "datetime64[ns]")].index
    return time_feats


#######################################################
#                   Time Conversion
#######################################################


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


################################################################
#                       Grid Search
################################################################


def get_models(
    X,
    n_lambda,
    lambdas,
    lambda_lasso,
    max_depth,
    epsilon,
    f,
    n_epochs,
    tol,
    lambda_max_ratio,
    n_active_max,
    screening,
    store_history,
):
    # Pipeline
    # **kwargs_lasso, **kwargs_spp
    # kwargs_lasso = dict, kwargs_spp = dict
    # call **kwargs_lasso, **kwargs_spp

    time_feats = time_features(X)
    time_transformer = FunctionTransformer(time_convert)

    numeric_feats = numeric_features(X)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "binning",
                KBinsDiscretizer(n_bins=3, encode="onehot",
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
    lasso = Lasso(
        lmbda=lambda_lasso,
        epsilon=epsilon,
        f=f,
        n_epochs=n_epochs,
        screening=screening,
        store_history=store_history,
    )
    models["lasso"] = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", lasso)]
    )
    lmbdas = np.logspace(-4, -0.5, 3)
    tuned_parameters["lasso"] = {
        "regressor__lmbda": lmbdas,
        "preprocessor__num__binning__n_bins": [2, 3, 4, 5],
    }

    # LassoCV
    models["lasso_cv"] = Pipeline(
        steps=[("preprocessor", preprocessor),
               ("regressor", linear_model.LassoCV()), ]
    )
    tuned_parameters["lasso_cv"] = {
        "preprocessor__num__binning__n_bins": [2, 3, 4, 5]}

    # RidgeCV
    models["ridge_cv"] = Pipeline(
        steps=[("preprocessor", preprocessor),
               ("regressor", linear_model.RidgeCV()), ]
    )
    tuned_parameters["ridge_cv"] = {
        "preprocessor__num__binning__n_bins": [2, 3, 4, 5]}

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
    spp_reg = SPPRegressor(
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
    )
    models["spp_reg"] = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", spp_reg)]
    )

    tuned_parameters["spp_reg"] = {
        "preprocessor__num__binning__n_bins": [2, 3, 4, 5],
        "regressor__max_depth": [2, 3, 4, 5],
    }

    return models, tuned_parameters


def compute_cv(X, y, models, n_splits, n_jobs=1):
    """
    Parameters
    ----------
    X: numpy.ndarray(), shape = (n_samples, n_features)
        features matrix

    y: numpy.array(), shape = '(n_samples, )
        target vector

    models : dict
        dict of models

    n_splits: intprint("X before = ", X)
        number of folds

    n_jobs: int
        number of jobs in parallel

    Returns
    -------
    cv_scores: dict
        cross validation scores for different models
    """
    # X = np.array(X.rename_axis('ID'))
    y = y.to_numpy().astype("float")
    cv_scores = {}

    for name, model in models.items():
        cv_scores[name] = cross_val_score(
            model, X, y, cv=n_splits, n_jobs=n_jobs
        ).mean()
    return cv_scores


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

    n_jobs: int
        number of jobs in parallel

    Returns
    -------
    cv_scores: dict
        cross validation scores for different models
    """
    # X = np.array(X.rename_axis('ID'))
    y = y.to_numpy().astype("float")
    gs_models = {}

    for name, model in models.items():
        if name != "spp_reg":
            gs = GridSearchCV(
                model, cv=n_splits, param_grid=tuned_parameters[name], n_jobs=n_jobs,
            )

            gs.fit(X, y)

            results_gs = {
                "best_score": gs.best_score_,
                "best_params": gs.best_params_,
            }

        else:
            kf = KFold(n_splits=n_splits, random_state=None, shuffle=False)

            gs_list = []

            # X = pd.DataFrame(X, index=X[:, 0])
            # y = pd.DataFrame(y)

            # change the order of the for loop
            fold_num = 0
            for train_index, test_index in kf.split(X):
                fold_num += 1
                print("TRAIN:", train_index, "TEST:", test_index)
                # shuffled_ind_train = np.array(shuffle(train_index))
                # shuffled_ind_test = np.array(shuffle(test_index))
                X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
                y_train, y_test = y[train_index], y[test_index]

                # X_train, X_test = X.iloc[shuffled_ind_train], X.iloc[shuffled_ind_test]
                # y_train, y_test = y.iloc[shuffled_ind_train], y.iloc[shuffled_ind_test]

                # n_bins_list = tuned_parameters['spp_reg']['preprocessor__num__binning__n_bins']
                # max_depth_list = tuned_parameters['spp_reg']['regressor__max_depth']
                n_bins_list = [2, 3, 4, 5]
                max_depth_list = [2, 3, 4, 5]

                for n_bins in n_bins_list:
                    for max_depth in max_depth_list:
                        spp_reg = SPPRegressor(
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
                        )
                        # spp_reg = model.set_params(preprocessor__num__binning__n_bins=n_bins, regressor__max_depth=max_depth)
                        enc = KBinsDiscretizer(
                            n_bins=n_bins, encode='onehot', strategy='quantile')
                        X_binned_train = enc.fit_transform(X_train)
                        X_binned_test = enc.transform(X_test)

                        solutions = spp_reg.fit(
                            X_binned_train, y_train).solutions_
                        cv_scores = spp_reg.score(X_binned_test, y_test)

                        lambda_list = []
                        slopes_list = []
                        for idx in range(len(solutions)):
                            lambda_list.append(solutions[idx]["lambda"])
                            slopes_list.append(
                                solutions[idx]["spp_lasso_slopes"])

                        if type(cv_scores) is np.float64:
                            cv_scores = [cv_scores]

                        # import ipdb
                        # ipdb.set_trace()

                        results = {
                            "n_bins": [
                                n_bins for i in range(len(cv_scores))],
                            "max_depth": [max_depth for i in range(len(cv_scores))],
                            "lambda": lambda_list,
                            "score": cv_scores,
                            "fold_number": [
                                fold_num for i in range(len(cv_scores))
                            ],
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
                )]["n_bins", "max_depth", "lambda"]

            results_gs = {
                "best_score": gs_groupby_params["score"].max(),
                "best_params": {'n_bins': best_params.iloc[0, 0],
                                'max_depth': best_params.iloc[0, 1],
                                'lambda': best_params.iloc[0, 2]
                                },
            }

        gs_models[name] = results_gs

        # test on a synthetic dataset such as checkerboard and compare optimal
        # params with the ones obtained with lasso cv on polynomial features
        # (function already implemented)

    return gs_models


def main():

    ####################################################################
    #                           Parameters
    ####################################################################

    lmbda = 1.0
    epsilon = 1e-7
    f = 10
    n_splits = 5
    screening = True
    store_history = True
    n_epochs = 10000
    n_jobs = 4
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
        "lmbda": lmbda,
        "epsilon": epsilon,
        "f": f,
        "n_epochs": n_epochs,
        "screening": screening,
        "store_history": store_history,
    }

    ######################################################################
    #                  Auto Label Function For Bar Plots
    ######################################################################

    def autolabel(rects, scale):
        """Attach a text label above each bar in *rects*, displaying its
        height.
        """

        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(round(height * scale, 0) / scale),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    #########################################################################
    #                       Auto Prices Dataset
    #########################################################################

    datasets = [
        "auto_prices",
        "lacrimes",
        "black_friday",
        "nyc_taxi",
        "housing_prices",
    ]

    for i in range(len(datasets)):
        if datasets[i] == "auto_prices":
            X, y = load_auto_prices()
            data_name = "auto_prices"
        if datasets[i] == "lacrimes":
            continue
            X, y = load_lacrimes()
            data_name = "lacrimes"
        if datasets[i] == "black_friday":
            continue
            X, y = load_black_friday()
            data_name = "black_friday"
        if datasets[i] == "nyc_taxi":
            continue
            X, y = load_nyc_taxi()
            data_name = "nyc_taxi"
        if datasets[i] == "housing_prices":
            continue
            load_housing_prices()
            data_name = "housing_prices"

        X = X[:10]
        y = y[:10]
        models, tuned_parameters = get_models(
            X=X,
            n_lambda=n_lambda,
            lambdas=lambdas,
            lambda_lasso=lmbda,
            max_depth=max_depth,
            epsilon=epsilon,
            f=f,
            n_epochs=n_epochs,
            tol=tol,
            lambda_max_ratio=lambda_max_ratio,
            n_active_max=n_active_max,
            screening=screening,
            store_history=store_history,
        )

        print("models = ", models)
        print("tuned_params = ", tuned_parameters)

        # cv_scores = compute_cv(
        #     X=X, y=y, models=models, n_splits=n_splits, n_jobs=n_jobs
        # )

        # print("cv_scores = ", cv_scores)

        # list_cv_scores = []

        # for k, v in cv_scores.items():
        #     print(f"{k}: {v}")
        #     list_cv_scores.append(v)

        # print("cv_scores without tuning params = ", list_cv_scores)

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
            n_jobs=1,
        )

        print("gs_models = ", gs_models)
        best_score_spp = gs_models["spp_reg"]["best_score"]
        best_params_spp = gs_models["spp_reg"]["best_params"]

        print('best_score_spp = ', best_score_spp)
        print('best_params = ', best_params_spp)
        # list_gs_scores = []
        # scores = pd.DataFrame(
        #     {"model": [], "best_cv_score": [], "best_param": []}
        # )

        # for k, v in gs_scores.items():
        #     print(f"{k} -- best params = {v.best_params_}")
        #     print(f"{k} -- cv scores = {v.best_score_}")
        #     list_gs_scores.append(v.best_score_)
        #     scores = scores.append(
        #         pd.DataFrame(
        #             {
        #                 "model": [k],
        #                 "best_cv_score": [v.best_score_],
        #                 "best_param": [v.best_params_],
        #             }
        #         )
        #     )

        # print("Housing Prices Dataset with 100 samples")
        # print("cv_score with tuning params = ", list_gs_scores)

        # print(scores)
        # scores.to_csv(
        #     "/home/mrivoire/Documents/M2DS_Polytechnique/INRIA-Parietal-Intership/Code/"
        #     + data_name
        #     + ".csv",
        #     index=False,
        # )

        #######################################################################
        #                         Bar Plots CV Scores
        #######################################################################

        # labels = ["Lasso", "Lasso_cv", "Ridge_cv", "XGB", "RF"]

        # x = np.arange(len(labels))  # the label locations
        # width = 0.35  # the width of the bars

        # fig, ax = plt.subplots()
        # rects1 = ax.bar(x, list_gs_scores, width)
        # # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax.set_ylabel("CV Scores")
        # ax.set_title("Crossval Scores By Predictive Model With Tuning For 100 Samples")
        # ax.set_xticks(x)
        # ax.set_xticklabels(labels)
        # ax.legend()

        # autolabel(rects1, 1000)

        # fig.tight_layout()

        # plt.show()

        #######################################################################
        #                         Bar Plots CV Time
        #######################################################################

        # labels = ["Lasso", "Lasso_cv", "Ridge_cv", "XGB", "RF"]

        # x = np.arange(len(labels))  # the label locations
        # width = 0.35  # the width of the bars

        # fig, ax = plt.subplots()
        # rects1 = ax.bar(x, execution_time_list, width)
        # # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax.set_ylabel("Running Time")
        # ax.set_title("Running Time By Predictive Model With Tuning For 100 Samples")
        # ax.set_xticks(x)
        # ax.set_xticklabels(labels)
        # ax.legend()

        # autolabel(rects1, 1000)

        # fig.tight_layout()

        # plt.show()


if __name__ == "__main__":
    main()
