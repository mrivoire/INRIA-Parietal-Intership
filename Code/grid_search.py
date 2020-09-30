"""
Experiments on real data with categorical variables

"""
import numpy as np
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
)
from SPP import SPPRegressor
from sklearn.model_selection import KFold

# from pandas_profiling import ProfileReport

#################################################
#          Preprocess Numeric Features
#################################################


def numeric_features(X):
    # numeric_feats = X.dtypes[(dtype.kind in "if" for dtype in X.dtypes)].index
    numeric_feats = X.dtypes[(X.dtypes == np.float64)
                             | (X.dtypes == np.int64)].index
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


################################################################
#                       Grid Search
################################################################


def get_models(
    X,
    n_bins,
    n_bins_less_bins,
    max_depth_less_bins,
    n_bins_more_bins,
    max_depth_more_bins,
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
        ],
        sparse_threshold=1  # we must have sparse output
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

    # Random Forest With A Single Tree (finite depth)
    dt = RandomForestRegressor(n_estimators=1)
    models["dt"] = Pipeline(
        steps=[("preprocessor", rf_preprocessor), ("regressor", dt)]
    )
    tuned_parameters["dt"] = {"regressor__max_depth": max_depth_less_bins}

    # Random Forest With A Single Tree (infinite depth)
    infinite_dt = RandomForestRegressor(n_estimators=1)
    models["infinite_dt"] = Pipeline(
        steps=[("preprocessor", rf_preprocessor), ("regressor", infinite_dt)]
    )
    tuned_parameters["infinite_dt"] = {"regressor__max_depth": [None]}

    # Random Forest With 100 Trees (default value)(finite depth)
    rf = RandomForestRegressor()
    models["rf"] = Pipeline(
        steps=[("preprocessor", rf_preprocessor), ("regressor", rf)]
    )
    tuned_parameters["rf"] = {"regressor__max_depth": max_depth_less_bins}

    # Random Forest With 100 Trees (default value)(infinite depth)
    infinite_rf = RandomForestRegressor()
    models["infinite_rf"] = Pipeline(
        steps=[("preprocessor", rf_preprocessor), ("regressor", infinite_rf)]
    )
    tuned_parameters["infinite_rf"] = {"regressor__max_depth": [None]}

    # SPP Regressor less bins
    spp_reg_less_bins = SPPRegressor(**kwargs_spp)
    models["spp_reg_less_bins"] = Pipeline(
        steps=[("preprocessor", preprocessor),
               ("regressor", spp_reg_less_bins)]
    )

    tuned_parameters["spp_reg_less_bins"] = {
        "preprocessor__num__binning__n_bins": n_bins_less_bins,
        "regressor__max_depth": max_depth_less_bins,
    }

    # SPP Regressor less bins
    spp_reg_more_bins = SPPRegressor(**kwargs_spp)
    models["spp_reg_more_bins"] = Pipeline(
        steps=[("preprocessor", preprocessor),
               ("regressor", spp_reg_more_bins)]
    )

    tuned_parameters["spp_reg_more_bins"] = {
        "preprocessor__num__binning__n_bins": n_bins_more_bins,
        "regressor__max_depth": max_depth_more_bins,
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
        print('name = ', name)
        print('model = ', model)
        if 'spp' not in name:
            gs = GridSearchCV(
                model, cv=n_splits, param_grid=tuned_parameters[name], n_jobs=n_jobs,
                scoring='neg_mean_squared_error',
            )

            gs.fit(X, y)

            results_gs = {
                "best_score": gs.best_score_,
                "best_params": gs.best_params_,
            }
            if (name == 'lasso_cv') | (name == 'ridge_cv'):
                best_slopes = gs.best_estimator_['regressor'].coef_
                n_active_features = np.count_nonzero(best_slopes)
                results_gs = {
                    "best_score": gs.best_score_,
                    "best_params": gs.best_params_,
                    "n_active_features": n_active_features
                }

        else:
            kf = KFold(n_splits=n_splits, random_state=None, shuffle=False)

            gs_list = []

            for fold_num, (train_index, test_index) in enumerate(kf.split(X), 1):
                # print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]

                n_bins_list = tuned_parameters[name]['preprocessor__num__binning__n_bins']
                max_depth_list = tuned_parameters[name]['regressor__max_depth']
                for n_bins in n_bins_list:
                    for max_depth in max_depth_list:
                        spp_reg = model.set_params(
                            preprocessor__num__binning__n_bins=n_bins, regressor__max_depth=max_depth)

                        spp_reg.fit(X_train, y_train)
                        solutions = spp_reg.steps[1][1].solutions_

                        # import ipdb
                        # ipdb.set_trace()

                        print('solutions = ', solutions)

                        cv_scores = spp_reg.score(X_test, y_test)

                        lambda_list = []
                        slopes_list = []

                        for this_sol in solutions:
                            lambda_list.append(this_sol["lambda"])
                            slopes_list.append(this_sol["spp_lasso_slopes"])

                        if type(cv_scores) is np.float64:
                            cv_scores = [cv_scores]

                        results = {
                            "n_bins": [n_bins] * len(cv_scores),
                            "max_depth": [max_depth] * len(cv_scores),
                            "lambda": lambda_list,
                            "score": cv_scores,
                            "fold_number": [fold_num] * len(cv_scores),
                            "slopes": slopes_list
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

            best_n_bins = best_params.iloc[0, 0]
            best_max_depth = best_params.iloc[0, 1]
            best_lambda = best_params.iloc[0, 2]

            print('best_n_bins = ', best_n_bins)
            print('best_max_depth = ', best_max_depth)
            print('best_lambda = ', best_lambda)
            print('lambda_list = ', len(lambda_list))

            idx_best_lambda = list(gs_dataframe['lambda']).index(best_lambda)
            best_slopes = list(gs_dataframe['slopes'])[idx_best_lambda]
            n_active_features = len(best_slopes)

            results_gs = {
                "best_score": best_score,
                "best_params": {'n_bins': best_params.iloc[0, 0],
                                'max_depth': best_params.iloc[0, 1],
                                'lambda': best_params.iloc[0, 2]
                                },
                "n_active_features": n_active_features,
            }

        gs_models[name] = results_gs

    return gs_models


def main():

    ####################################################################
    #                           Parameters
    ####################################################################

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

    X = X[:100]
    y = y[:100]

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
    n_bins_less_bins = [3]
    n_bins_more_bins = [10]
    max_depth_less_bins = [2, 3, 4, 5, 6]
    max_depth_more_bins = [2, 3, 4]

    models, tuned_parameters = get_models(
        X=X,
        n_bins=n_bins,
        n_bins_less_bins=n_bins_less_bins,
        max_depth_less_bins=max_depth_less_bins,
        n_bins_more_bins=n_bins_more_bins,
        max_depth_more_bins=max_depth_more_bins,
        kwargs_lasso=kwargs_lasso,
        kwargs_spp=kwargs_spp
    )

    # del models['lasso']
    # del models['lasso_cv']
    # del models['ridge_cv']
    # del models['rf']
    # del models['xgb']

    # del tuned_parameters['lasso']
    # del tuned_parameters['lasso_cv']
    # del tuned_parameters['ridge_cv']
    # del tuned_parameters['rf']
    # del tuned_parameters['xgb']

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
    # best_score_spp = gs_models["spp_reg"]["best_score"]
    # best_params_spp = gs_models["spp_reg"]["best_params"]

    # print('best_score_spp = ', best_score_spp)
    # print('best_params = ', best_params_spp)

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

    ######################################################################
    #                  Auto Label Function For Bar Plots
    ######################################################################

    # def autolabel(rects, scale):
    #     """Attach a text label above each bar in *rects*, displaying its
    #     height.
    #     """

    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate(
    #             "{}".format(round(height * scale, 0) / scale),
    #             xy=(rect.get_x() + rect.get_width() / 2, height),
    #             xytext=(0, 3),  # 3 points vertical offset
    #             textcoords="offset points",
    #             ha="center",
    #             va="bottom",
    #         )

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
