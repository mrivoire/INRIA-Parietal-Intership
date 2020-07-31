"""
Experiments on real data with categorical variables

"""
import numpy as np
import pandas as pd
from sklearn import datasets
import joblib
import faulthandler
import time
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew
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
    load_auto_prices, load_lacrimes, load_black_friday, load_nyc_taxi,
    load_housing_prices)
from pandas_profiling import ProfileReport

#################################################
#          Preprocess Numeric Features
#################################################


def numeric_features(X):
    numeric_feats = X.dtypes[~((X.dtypes == "object")
                                | (X.dtypes == "category"))].index

    X[numeric_feats] = X[numeric_feats].astype(np.float64)

    return X, numeric_feats


##################################################
#          Preprocess Categorical Features
##################################################


def categorical_features(X):
    categorical_feats = X.dtypes[((X.dtypes == "object")
                                  | (X.dtypes == "category"))].index

    X[categorical_feats] = X[categorical_feats].astype('category')

    return X, categorical_feats


#######################################################
#                   Time Conversion
#######################################################


def time_convert(X):
    if hasattr(X, 'columns'):
        X = X.copy()
    columns_kept_time = []
    for col, dtype in zip(X.columns, X.dtypes):
        if X[col].dtypes == "datetime64[ns]":
            col_name = str(col)
            datetime = pd.to_datetime(X[col])
            X[col_name + 'year'] = datetime.dt.year
            X[col_name + 'weekday'] = datetime.dt.dayofweek
            X[col_name + 'yearday'] = datetime.dt.dayofyear
            X[col_name + 'time'] = datetime.dt.minute
            columns_kept_time.extend([col_name + 'year',
                                      col_name + 'weekday',
                                      col_name + 'yearday',
                                      col_name + 'time'])
            X = X.drop([col_name], axis=1)

    return X, columns_kept_time


##########################################################
#     One-hot encoding of the categorical features
##########################################################


def onehot_encoding(dataset):
    onehot_data = pd.get_dummies(dataset)

    return onehot_data


################################################################
#                       Grid Search
################################################################


def get_models(X, **kwargs):
    # Pipeline

    X, time_feats = time_convert(X)
    print("time_feats = ", time_feats)
    time_transformer = FunctionTransformer(time_convert)
    time_transformer.transform(X)

    numeric_feats = numeric_features(X)
    print("numeric_feats = ", numeric_feats)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('binning', KBinsDiscretizer(n_bins=3, encode='onehot',
                                     strategy='quantile'))])

    print("numeric_transformer = ", numeric_transformer)

    rf_numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    print("rf_numeric_transformer = ", rf_numeric_transformer)

    categorical_feats = categorical_features(X)
    print("categorical_feats = ", categorical_feats)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    print("categorical_transformer = ", categorical_transformer)

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats),
        ('time', time_transformer, time_feats)])

    print("preprocessor = ", preprocessor)

    rf_preprocessor = ColumnTransformer(transformers=[
        ('num', rf_numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)])

    print("rf_preprocessor = ", rf_preprocessor)
    models = {}
    tuned_parameters = {}

    # Lasso
    lasso = Lasso(**kwargs)
    models['lasso'] = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', lasso)])
    lmbdas = np.logspace(-4, -0.5, 3)
    tuned_parameters['lasso'] = \
        {'regressor__lmbda': lmbdas, 'preprocessor__num__binning__n_bins': [2, 3, 5, 7, 10, 12, 15, 20]}
    print("lasso = ", models['lasso'])
    print("tuned_params_lasso = ", tuned_parameters['lasso'])
    # LassoCV
    models['lasso_cv'] = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', linear_model.LassoCV())])
    tuned_parameters['lasso_cv'] = {'preprocessor__num__binning__n_bins': [2, 3, 5, 7, 10, 12, 15, 20]}
    print("lasso_cv = ", models['lasso_cv'])
    print("tuned_parameters_lasso_cv = ", tuned_parameters['lasso_cv'])
    # RidgeCV
    models['ridge_cv'] = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', linear_model.RidgeCV())])
    tuned_parameters['ridge_cv'] = {'preprocessor__num__binning__n_bins': [2, 3, 5, 7, 10, 12, 15, 20]}
    print("models_ridge_cv = ", models['ridge_cv'])
    print("tuned_params_ridge_cv = ", tuned_parameters['ridge_cv'])
    # XGBoost
    xgb = XGBRegressor()
    models['xgb'] = Pipeline(steps=[('preprocessor', rf_preprocessor),
                                    ('regressor', xgb)])
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    tuned_parameters['xgb'] = {'regressor__alpha': alphas,
                               'regressor__n_estimators': [30, 100]}

    # tuned_parameters['xgb'] = {'regressor__n_estimators': [30, 100]}
    print("models_xgb = ", models['xgb'])
    print("tuned_params_xgb = ", tuned_parameters['xgb'])
    # Random Forest
    rf = RandomForestRegressor()
    models['rf'] = Pipeline(steps=[('preprocessor', rf_preprocessor),
                                   ('regressor', rf)])
    tuned_parameters['rf'] = {'regressor__max_depth': [3, 5]}

    print("models_rf = ", type(models['rf']))
    print("tuned_params_rf = ", type(tuned_parameters['rf']))

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
    y = y.to_numpy().astype('float')
    cv_scores = {}

    for name, model in models.items():
        print("name = ", name)
        print("model = ", model)
        cv_scores[name] = \
            cross_val_score(model, X, y, cv=n_splits, n_jobs=n_jobs).mean()
        print("cv = ", cv_scores[name])
    return cv_scores


def compute_gs(X, y, models, tuned_parameters, n_splits, n_jobs=1):
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
    y = y.to_numpy().astype('float')
    gs_models = {}

    for name, model in models.items():
        gs = \
            GridSearchCV(model, cv=n_splits,
                         param_grid=tuned_parameters[name], n_jobs=n_jobs)
        gs.fit(X, y)
        gs_models[name] = gs

    return gs_models


def main():

    ####################################################################
    #                           Parameters
    ####################################################################

    lmbda = 1.
    epsilon = 1e-7
    f = 10
    n_splits = 5
    screening = True
    store_history = True
    n_epochs = 10000
    n_jobs = 4

    ######################################################################
    #                  Auto Label Function For Bar Plots
    ######################################################################

    def autolabel(rects, scale):
        """Attach a text label above each bar in *rects*, displaying its
        height.
        """

        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height * scale, 0)/scale),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    #########################################################################
    #                       Auto Prices Dataset
    #########################################################################

    start1 = time.time()

    X, y = load_auto_prices()
    print("X = ", X.dtypes)
    # profile = ProfileReport(X, title='Pandas Profiling Report')
    # profile.to_file(output_file='output_black_friday.html')
   
    # X, y = load_lacrimes()
    # X, y = load_black_friday()
    # X, y = load_housing_prices()
    # X, y = load_nyc_taxi()
    # print("X = ", type(X))  # pandas class

    models, tuned_parameters = get_models(
            X[:20],
            lmbda=lmbda,
            epsilon=epsilon,
            f=f, n_epochs=n_epochs,
            screening=screening,
            store_history=store_history)

    print("params = ", params)
    print("models = ", models)
    X = X.to_numpy()
    print("X = ", X)
    print("X = ", type(X))  # numpy ndarray
    X = np.array(X.rename_axis('ID'))

    cv_scores = compute_cv(X=X[:20].values,
                           y=y[:20],
                           models=models, n_splits=n_splits, n_jobs=n_jobs)

    print("cv_scores = ", cv_scores)

    end1 = time.time()
    delay1 = end1 - start1

    list_cv_scores = []

    for k, v in cv_scores.items():
        print(f'{k}: {v}')
        list_cv_scores.append(v)

    print("cv_scores without tuning params = ", list_cv_scores)

    start2 = time.time()
    gs_scores = compute_gs(X=X[:20],
                           y=y[:20],
                           models=models, n_splits=n_splits,
                           tuned_parameters=tuned_parameters, n_jobs=n_jobs)

    end2 = time.time()
    delay2 = end2 - start2
    delay = delay1 + delay2
    list_gs_scores = []
    for k, v in gs_scores.items():
        print(f'{k} -- best params = {v.best_params_}')
        print(f'{k} -- cv scores = {v.best_score_}')
        list_gs_scores.append(v.best_score_)

    print("cv_score with tuning params = ", list_gs_scores)

    print("delay_auto_prices = ", delay)

    #######################################################################
    #                     Bar Plots Auto Prices Dataset
    #######################################################################

    labels = ['Lasso', 'Lasso_cv', 'Ridge_cv', 'XGB', 'RF']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, list_gs_scores, width)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('CV Scores')
    ax.set_title('Crossval Scores By Predictive Model With Tuning For 1000 Samples')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1, 1000)

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()