"""
Experiments on real data with categorical variables

"""
import numpy as np
import pandas as pd
from sklearn import datasets
import joblib

import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
# from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split, GridSearchCV

from cd_solver_lasso_numba import Lasso


#################################################
#           Numeric Features
#################################################


def numeric_features(dataset):
    numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index

    return numeric_feats


##################################################
#           Categorical Features
##################################################


def categorical_features(dataset):
    categorical_feats = dataset.dtypes[dataset.dtypes == "object"].index

    return categorical_feats


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
        {'regressor__lmbda': lmbdas, 'preprocessor__num__binning__n_bins': [2, 3, 5, 7, 10, 12, 15]}

    # LassoCV
    models['lasso_cv'] = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', linear_model.LassoCV())])
    tuned_parameters['lasso_cv'] = {'preprocessor__num__binning__n_bins': [2, 3, 5, 7, 10, 12, 15]}

    # RidgeCV
    models['ridge_cv'] = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', linear_model.RidgeCV())])
    tuned_parameters['ridge_cv'] = {'preprocessor__num__binning__n_bins': [2, 3, 5, 7, 10, 12, 15]}

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
    #'nyc-taxi-green-dec-2016': 42208,
    # No high-cardinality categories
    # https://www.openml.org/d/41540
    'black_friday': 41540,
    }

    age_mapping = {
        '0-17': 15,
        '18-25': 22,
        '26-35': 30,
        '36-45': 40,
        '46-50': 48,
        '51-55': 53,
        '55+': 60
        }

    for name, openml_id in openml_ids.items():
        X, y = mem.cache(datasets.fetch_openml)(
                                    data_id=openml_id, return_X_y=True,
                                    as_frame=True)
        raw_data[name] = X, y
        
    # print("raw data = ", raw_data)
    auto_price_rawdata = raw_data['BNG(auto_price)']
    crimes_rawdata = raw_data['la_crimes']
    black_friday_rawdata = raw_data['black_friday']

    # print("auto prices dataset : ", auto_price_rawdata)
    # print("crimes dataset : ", crimes_rawdata)
    # print("black friday dataset : ", black_friday_rawdata)

    # Encode the data to numerical matrices

    clean_data = dict()

    for name, (X, y) in raw_data.items():
        print('\nBefore encoding: % 20s: n=%i, d=%i'
            % (name, X.shape[0], X.shape[1]))
        if hasattr(X, 'columns'):
            print(list(X.columns))
            X = X.copy()
            if 'Age' in X.columns:
                X['Age'] = X['Age'].replace(age_mapping)
            columns_kept = []
            for col, dtype in zip(X.columns, X.dtypes):
                if col.endswith('datetime'):
                    # Only useful for NYC taxi
                    col_name = col[:-8]
                    datetime = pd.to_datetime(X[col])
                    X[col_name + 'year'] = datetime.dt.year
                    X[col_name + 'weekday'] = datetime.dt.dayofweek
                    X[col_name + 'yearday'] = datetime.dt.dayofyear
                    X[col_name + 'time'] = datetime.dt.time
                    columns_kept.extend([col_name + 'year',
                                        col_name + 'weekday',
                                        col_name + 'yearday',
                                        col_name + 'time'])
                elif dtype.kind in 'if':
                    columns_kept.append(col)
                elif hasattr(dtype, 'categories'):
                    if len(dtype.categories) < 30:
                        columns_kept.append(col)
                elif dtype.kind == 'O':
                    if X[col].nunique() < 30:
                        columns_kept.append(col)
            X = X[columns_kept]

            X_array = pd.get_dummies(X).values
        else:
            X_array = X
        clean_data[name] = X_array, y
        print('After encoding: % 20s: n=%i, d=%i'
            % (name, X_array.shape[0], X_array.shape[1]))

    print("clean data = ", clean_data)

    lmbda = 1.
    epsilon = 1e-7
    f = 10
    n_splits = 5
    screening = True
    store_history = True
    n_epochs = 10000
    n_jobs = 4

    auto_price_data = clean_data['BNG(auto_price)']
    crimes_data = clean_data['la_crimes']
    black_friday_data = clean_data['black_friday']

    # print("auto price clean data : ", auto_price_data)
    # print("crimes clean data : ", crimes_data)
    # print("balck friday clean data : ", black_friday_data)
    X_auto = auto_price_data[0]
    y_auto = auto_price_data[1]
    X_crimes = crimes_data[0]
    y_crimes = crimes_data[1]
    X_black_friday = black_friday_data[0]
    y_black_friday = black_friday_data[1]


    models, tuned_parameters = get_models(X_auto, lmbda=lmbda, epsilon=epsilon, 
                                          f=f, n_epochs=n_epochs,
                                          screening=screening,
                                          store_history=store_history)

    # cv_scores = compute_cv(X=X, y=y, models=models, n_splits=n_splits,
    #                        n_jobs=n_jobs)
    # list_cv_scores = []

    # for k, v in cv_scores.items():
    #     print(f'{k}: {v}')
    #     list_cv_scores.append(v)

    # print("cv_scores without tuning params = ", list_cv_scores)

    # gs_scores = compute_gs(X=X, y=y, models=models, n_splits=n_splits,
    #                        tuned_parameters=tuned_parameters, n_jobs=n_jobs)

    # list_gs_scores = []
    # for k, v in gs_scores.items():
    #     print(f'{k} -- best params = {v.best_params_}')
    #     print(f'{k} -- cv scores = {v.best_score_}')
    #     list_gs_scores.append(v.best_score_)

    # print("cv_score with tuning params = ", list_gs_scores)


if __name__ == "__main__":
    main()