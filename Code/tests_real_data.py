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


#################################################
#               Read Datasets
#################################################


def read_datasets():
    faulthandler.enable()
    mem = joblib.Memory(location='cache')

    raw_data = dict()

    openml_ids = {
    # https://www.openml.org/d/1189 BNG(auto_price)
    # No high-cardinality categories
    'BNG(auto_price)': 1189,
    ## https://www.openml.org/d/42160
    ## A few high-cardinality strings and some dates
    'la_crimes': 42160,
    # https://www.openml.org/d/42208 nyc-taxi-green-dec-2016
    # No high cardinality strings, a few datetimes
    # Skipping because I cannot get it to encode right
    'nyc-taxi-green-dec-2016': 42208,
    # No high-cardinality categories
    # https://www.openml.org/d/41540
    'black_friday': 41540,
    }

    for name, openml_id in openml_ids.items():
        X, y = mem.cache(datasets.fetch_openml)(
                                    data_id=openml_id, return_X_y=True,
                                    as_frame=True)
        raw_data[name] = X, y

    auto_price_rawdata = raw_data['BNG(auto_price)']
    crimes_rawdata = raw_data['la_crimes']
    nyc_taxi_rawdata = raw_data['nyc-taxi-green-dec-2016']
    black_friday_rawdata = raw_data['black_friday']

    X_auto_raw = auto_price_rawdata[0]
    y_auto_raw = auto_price_rawdata[1]
    X_crimes_raw = crimes_rawdata[0]
    y_crimes_raw = crimes_rawdata[1]
    X_nyc_taxi_raw = nyc_taxi_rawdata[0]
    y_nyc_taxi_raw = nyc_taxi_rawdata[1]
    X_black_friday_raw = black_friday_rawdata[0]
    y_black_friday_raw = black_friday_rawdata[1]

    return (
        X_auto_raw, y_auto_raw, X_crimes_raw, y_crimes_raw, X_nyc_taxi_raw,
        y_nyc_taxi_raw, X_black_friday_raw, y_black_friday_raw, 
        auto_price_rawdata, crimes_rawdata, nyc_taxi_rawdata, 
        black_friday_rawdata)


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

    time_transformer = FunctionTransformer(time_convert)
    time_transformer.transform(X)

    numeric_feats = numeric_features(X)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('binning', KBinsDiscretizer(n_bins=3, encode='onehot',
                                     strategy='quantile'))])

    rf_numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_feats = categorical_features(X)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)])

    rf_preprocessor = ColumnTransformer(transformers=[
        ('num', rf_numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)])

    models = {}
    tuned_parameters = {}

    # Lasso
    lasso = Lasso(**kwargs)
    models['lasso'] = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', lasso)])
    lmbdas = np.logspace(-4, -0.5, 3)
    tuned_parameters['lasso'] = \
        {'regressor__lmbda': lmbdas, 'preprocessor__num__binning__n_bins': [2, 3, 5, 7, 10, 12, 15, 20]}

    # LassoCV
    models['lasso_cv'] = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', linear_model.LassoCV())])
    tuned_parameters['lasso_cv'] = {'preprocessor__num__binning__n_bins': [2, 3, 5, 7, 10, 12, 15, 20]}

    # RidgeCV
    models['ridge_cv'] = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', linear_model.RidgeCV())])
    tuned_parameters['ridge_cv'] = {'preprocessor__num__binning__n_bins': [2, 3, 5, 7, 10, 12, 15, 20]}

    # XGBoost
    xgb = XGBRegressor()
    models['xgb'] = Pipeline(steps=[('preprocessor', rf_preprocessor),
                                    ('regressor', xgb)])
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    tuned_parameters['xgb'] = {'regressor__alpha': alphas,
                               'regressor__n_estimators': [30, 100]}

    # tuned_parameters['xgb'] = {'regressor__n_estimators': [30, 100]}

    # Random Forest
    rf = RandomForestRegressor()
    models['rf'] = Pipeline(steps=[('preprocessor', rf_preprocessor),
                                   ('regressor', rf)])
    tuned_parameters['rf'] = {'regressor__max_depth': [3, 5]}

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
    y = y.to_numpy().astype('float')
    cv_scores = {}

    for name, model in models.items():
        cv_scores[name] = \
            cross_val_score(model, X, y, cv=n_splits, n_jobs=n_jobs).mean()

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

    ######################################################################
    #                           Read CSV
    ######################################################################

    (X_auto_raw, y_auto_raw, X_crimes_raw, y_crimes_raw, X_nyc_taxi_raw,
        y_nyc_taxi_raw, X_black_friday_raw, y_black_friday_raw, 
        auto_price_rawdata, crimes_rawdata, nyc_taxi_rawdata, 
        black_friday_rawdata) = read_datasets()

    #########################################################################
    #                       Auto Prices Dataset
    #########################################################################

    start1 = time.time()

    # X, y = load_auto_prices()
    # X, y = load_lacrimes()
    # X, y = load_black_friday()
    X, y = load_housing_prices()
    # X, y = load_nyc_taxi()
    print("X = ", X.dtypes)

    params, models = models, tuned_parameters = get_models(
            X[:1000], 
            lmbda=lmbda,
            epsilon=epsilon,
            f=f, n_epochs=n_epochs,
            screening=screening,
            store_history=store_history)

    print("params = ", params)
    print("models = ", models)

    cv_scores = compute_cv(X=X[:1000], 
                           y=y[:1000],
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
    gs_scores = compute_gs(X=X[:1000], 
                           y=y[:1000],
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