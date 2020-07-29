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
from sklearn.preprocessing import FunctionTransformer
# from sklearn.model_selection import train_test_split, GridSearchCV

from cd_solver_lasso_numba import Lasso


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


###################################################
#   Preprocess Time Features La Crimes Dataset
###################################################


def time_features_lacrimes(X):
    if hasattr(X, 'columns'):
        X = X.copy()
    columns_kept_time = []
    for col, dtype in zip(X.columns, X.dtypes):
        if ((col == 'Date_Reported') | (col == 'Date_Occurred')):
            X[col] = pd.to_datetime(X[col])
            columns_kept_time.append(col)
    return X, columns_kept_time


###############################################################
#       Preprocess Time Features NYC Taxi Dataset
###############################################################


def time_features_NYCTaxi(X):
    if hasattr(X, 'columns'):
        X = X.copy()
    columns_kept_time = []
    for col, dtype in zip(X.columns, X.dtypes):
        if col.endswith('datetime'):
            # Only useful for NYC taxi
            col_name = col[:-8]
            X[col] = pd.to_datetime(X[col])
            columns_kept_time.append(col_name)
            
    return X, columns_kept_time


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

######################################################
#               Preprocess Age Feature
######################################################


def age_map(X):

    age_mapping = {
        '0-17': 15,
        '18-25': 22,
        '26-35': 30,
        '36-45': 40,
        '46-50': 48,
        '51-55': 53,
        '55+': 60
    }

    if hasattr(X, 'columns'):
        X = X.copy()
        columns_kept_age = []
        if 'Age' in X.columns:
            X['Age'] = X['Age'].replace(age_mapping)
            columns_kept_age.append('Age')

    return X, columns_kept_age


######################################################
#          Preprocess Int-Float Features
######################################################


def preprocess_int_float(X):
    if hasattr(X, 'columns'):
        print(list(X.columns))
        X = X.copy()
        columns_kept_if = []
        for col, dtype in zip(X.columns, X.dtypes):
            if dtype.kind in 'if':
                columns_kept_if.append(col)

    return columns_kept_if


######################################################
#          Preprocess Category Features
######################################################


def preprocess_category_feats(X):
    if hasattr(X, 'columns'):
        print(list(X.columns))
        X = X.copy()
        columns_kept_category = []
        for col, dtype in zip(X.columns, X.dtypes):
            if hasattr(dtype, 'categories'):
                if len(dtype.categories) < 30:
                    columns_kept_category.append(col)

    return columns_kept_category


######################################################
#          Preprocess Object Features
######################################################


def preprocess_object_feats(X):
    if hasattr(X, 'columns'):
        print(list(X.columns))
        X = X.copy()
        columns_kept_object = []
        for col, dtype in zip(X.columns, X.dtypes):
            if dtype.kind == 'O':
                if X[col].nunique() < 30:
                    columns_kept_object.append(col)

    return columns_kept_object


######################################################
#     Create dataset with only the kept columns
######################################################


def preprocessed_dataset(X, columns_kept_list):
    flatten_columns_kept_list = []
    for columns_kept in columns_kept_list:
        for item in columns_kept:
            flatten_columns_kept_list.append(item)

    X = X[flatten_columns_kept_list]
    return X


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

    X, datetime_feats_lacrimes = time_features_lacrimes(X)
    X, datetime_feats_NYCTaxi = time_features_NYCTaxi(X)
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

    print("auto_price_rawdata = ", auto_price_rawdata)
    print("crimes_rawdata = ", crimes_rawdata)
    print("black_friday_rawdata = ", black_friday_rawdata)
    print("nyc_taxi_rawdata = ", nyc_taxi_rawdata)

    # print("X_auto_raw = ", X_auto_raw)
    # print("y_auto_raw = ", y_auto_raw)
    # print("X_crimes_raw = ", X_crimes_raw)
    # print("y_crimes_raw = ", y_crimes_raw)
    # print("X_black_friday_raw = ", X_black_friday_raw)
    # print("y_black_friday_raw = ", y_black_friday_raw)
    # print(X_nyc_taxi_raw.dtypes)

    ########################################################################
    #                       Tests Load Datasets
    ########################################################################

    X_lacrimes, columns_kept = time_features_lacrimes(X_crimes_raw)
    X_lacrimes2, list_time_features = time_convert(X_lacrimes)
    print('dtypes = ', X_lacrimes2.dtypes)
    print('columns_kept = ', columns_kept)

    X_NYCTaxi, columns_kept = time_features_NYCTaxi(X_nyc_taxi_raw)
    # print('dtypes = ', X_NYCTaxi.dtypes)
    # print('columns_kept = ', columns_kept)

    X_NYCTaxi2, list_time_features = time_convert(X_NYCTaxi)
    # print('X_NYCTaxi = ', X_NYCTaxi2)
    # print('dtypes = ', X_NYCTaxi2.dtypes)
    X_NYCTaxi, y_NYCTaxi = preprocess_NYCtaxi()
    #########################################################################
    #                       Auto Prices Dataset
    #########################################################################

    # start_auto_prices1 = time.time()

    # X_auto_prices, y_auto_prices_raw = preprocess_auto_prices()

    # params, models = models, tuned_parameters = get_models(
    #         X_auto_prices[:1000], 
    #         lmbda=lmbda,
    #         epsilon=epsilon,
    #         f=f, n_epochs=n_epochs,
    #         screening=screening,
    #         store_history=store_history)

    # cv_scores = compute_cv(X=X_auto_prices[:1000], 
    #                        y=y_auto_prices_raw[:1000],
    #                        models=models, n_splits=n_splits, n_jobs=n_jobs)

    # end_auto_prices1 = time.time()
    # delay_auto_prices1 = end_auto_prices1 - start_auto_prices1

    # list_cv_scores_auto = []

    # for k, v in cv_scores.items():
    #     print(f'{k}: {v}')
    #     list_cv_scores_auto.append(v)

    # print("cv_scores without tuning params = ", list_cv_scores_auto)

    # start_auto_prices2 = time.time()
    # gs_scores = compute_gs(X=X_auto_prices[:1000], 
    #                        y=y_auto_prices_raw[:1000],
    #                        models=models, n_splits=n_splits,
    #                        tuned_parameters=tuned_parameters, n_jobs=n_jobs)

    # end_auto_prices2 = time.time()
    # delay_auto_prices2 = end_auto_prices2 - start_auto_prices2
    # delay_auto_prices = delay_auto_prices1 + delay_auto_prices2
    # list_gs_scores_auto = []
    # for k, v in gs_scores.items():
    #     print(f'{k} -- best params = {v.best_params_}')
    #     print(f'{k} -- cv scores = {v.best_score_}')
    #     list_gs_scores_auto.append(v.best_score_)

    # print("cv_score with tuning params = ", list_gs_scores_auto)
    
    # print("delay_auto_prices = ", delay_auto_prices)

    #######################################################################
    #                     Bar Plots Auto Prices Dataset
    #######################################################################

    # labels = ['Lasso', 'Lasso_cv', 'Ridge_cv', 'XGB', 'RF']

    # x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x, list_gs_scores_auto, width)
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('CV Scores')
    # ax.set_title('Crossval Scores By Predictive Model With Tuning For 1000 Samples')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # ax.legend()

    # autolabel(rects1, 1000)

    # fig.tight_layout()

    # plt.show()

    #########################################################################
    #                           LA Crimes Dataset
    #########################################################################
    # start_lacrimes1 = time.time()

    # X_lacrimes, y_crimes_raw = preprocess_lacrimes()

    # params, models = models, tuned_parameters = get_models(
    #         X_lacrimes[:1000], 
    #         lmbda=lmbda,
    #         epsilon=epsilon,
    #         f=f, n_epochs=n_epochs,
    #         screening=screening,
    #         store_history=store_history)

    # cv_scores = compute_cv(X=X_lacrimes[:1000], 
    #                        y=y_crimes_raw[:1000],
    #                        models=models, n_splits=n_splits, n_jobs=n_jobs)

    # end_lacrimes1 = time.time()
    # delay_lacrimes1 = end_lacrimes1 - start_lacrimes1

    # list_cv_scores_lacrimes = []

    # for k, v in cv_scores.items():
    #     print(f'{k}: {v}')
    #     list_cv_scores_lacrimes.append(v)

    # print("cv_scores without tuning params = ", list_cv_scores_lacrimes)

    # start_lacrimes2 = time.time()
    # gs_scores = compute_gs(X=X_lacrimes[:1000], 
    #                        y=y_crimes_raw[:1000],
    #                        models=models, n_splits=n_splits,
    #                        tuned_parameters=tuned_parameters, n_jobs=n_jobs)

    # end_lacrimes2 = time.time()
    # delay_lacrimes2 = end_lacrimes2 - start_lacrimes2

    # delay_lacrimes = delay_lacrimes1 + delay_lacrimes2

    # list_gs_scores_lacrimes = []
    # for k, v in gs_scores.items():
    #     print(f'{k} -- best params = {v.best_params_}')
    #     print(f'{k} -- cv scores = {v.best_score_}')
    #     list_gs_scores_lacrimes.append(v.best_score_)

    # print("cv_score with tuning params = ", list_gs_scores_lacrimes)

    # print("delay_lacrimes = ", delay_lacrimes)

    #########################################################################
    #                     Bar Plots LA Crimes Dataset
    #########################################################################

    # labels = ['Lasso', 'Lasso_cv', 'Ridge_cv', 'XGB', 'RF']

    # x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x, list_gs_scores_lacrimes, width)
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('CV Scores')
    # ax.set_title('Crossval Scores By Predictive Model With Tuning For 1000 Samples')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # ax.legend()

    # autolabel(rects1, 1000)

    # fig.tight_layout()

    # plt.show()

    #########################################################################
    #                       Black Friday Dataset
    #########################################################################

    # start_black_friday1 = time.time()

    # X_black_friday, y_black_friday_raw = preprocess_black_friday() 

    # params, models = models, tuned_parameters = get_models(
    #         X_black_friday[:1000], 
    #         lmbda=lmbda,
    #         epsilon=epsilon,
    #         f=f, n_epochs=n_epochs,
    #         screening=screening,
    #         store_history=store_history)

    # cv_scores = compute_cv(X=X_black_friday[:1000], 
    #                        y=y_black_friday_raw[:1000],
    #                        models=models, n_splits=n_splits, n_jobs=n_jobs)

    # end_black_friday1 = time.time()
    # delay_black_friday1 = end_black_friday1 - start_black_friday1

    # list_cv_scores_black_friday = []

    # for k, v in cv_scores.items():
    #     print(f'{k}: {v}')
    #     list_cv_scores_black_friday.append(v)

    # print("cv_scores without tuning params = ", list_cv_scores_black_friday)

    # start_black_friday2 = time.time()

    # gs_scores = compute_gs(X=X_black_friday[:1000], 
    #                        y=y_black_friday_raw[:1000],
    #                        models=models, n_splits=n_splits,
    #                        tuned_parameters=tuned_parameters, n_jobs=n_jobs)

    # end_black_friday2 = time.time()
    # delay_black_friday2 = end_black_friday2 - start_black_friday2
    # delay_black_friday = delay_black_friday1 + delay_black_friday2

    # list_gs_scores_black_friday = []
    # for k, v in gs_scores.items():
    #     print(f'{k} -- best params = {v.best_params_}')
    #     print(f'{k} -- cv scores = {v.best_score_}')
    #     list_gs_scores_black_friday.append(v.best_score_)

    # print("cv_score with tuning params = ", list_gs_scores_black_friday)

    # print("delay_black_friday = ", delay_black_friday)

    ######################################################################
    #                   Bar Plots Black Friday Dataset
    ######################################################################

    # labels = ['Lasso', 'Lasso_cv', 'Ridge_cv', 'XGB', 'RF']

    # x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x, list_gs_scores_black_friday, width)
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('CV Scores')
    # ax.set_title('Crossval Scores By Predictive Model With Tuning For 1000 Samples')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # ax.legend()

    # autolabel(rects1, 1000)

    # fig.tight_layout()

    # plt.show()

    #########################################################################
    #                       NYC Taxi Dataset
    #########################################################################
    start_NYCTaxi1 = time.time()
    
    X_NYCTaxi, y_NYCTaxi = preprocess_NYCtaxi()

    params, models = models, tuned_parameters = get_models(
            X_NYCTaxi[:1000], 
            lmbda=lmbda,
            epsilon=epsilon,
            f=f, n_epochs=n_epochs,
            screening=screening,
            store_history=store_history)

    cv_scores = compute_cv(X=X_NYCTaxi[:1000], 
                           y=y_NYCTaxi[:1000],
                           models=models, n_splits=n_splits, n_jobs=n_jobs)

    end_NYCTaxi1 = time.time()
    delay_NYCTaxi1 = end_NYCTaxi1 - start_NYCTaxi1

    list_cv_scores_NYCTaxi = []

    for k, v in cv_scores.items():
        print(f'{k}: {v}')
        list_cv_scores_NYCTaxi.append(v)

    print("cv_scores without tuning params = ", list_cv_scores_NYCTaxi)

    start_NYCTaxi2 = time.time()

    gs_scores = compute_gs(X=X_NYCTaxi[:1000], 
                           y=y_NYCTaxi[:1000],
                           models=models, n_splits=n_splits,
                           tuned_parameters=tuned_parameters, n_jobs=n_jobs)

    end_NYCTaxi2 = time.time()
    delay_NYCTaxi2 = end_NYCTaxi2 - start_NYCTaxi2
    delay_NYCTaxi = delay_NYCTaxi1 + delay_NYCTaxi2

    list_gs_scores_NYCTaxi = []
    for k, v in gs_scores.items():
        print(f'{k} -- best params = {v.best_params_}')
        print(f'{k} -- cv scores = {v.best_score_}')
        list_gs_scores_NYCTaxi.append(v.best_score_)

    print("cv_score with tuning params = ", list_gs_scores_NYCTaxi)

    print("delay_NYCTaxi = ", delay_NYCTaxi)

    #########################################################################
    #                     Bar Plots NYC Taxi Dataset
    #########################################################################

    # labels = ['Lasso', 'Lasso_cv', 'Ridge_cv', 'XGB', 'RF']

    # x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x, list_gs_scores_NYCTaxi, width)
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('CV Scores')
    # ax.set_title('Crossval Scores By Predictive Model With Tuning For 1000 Samples')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # ax.legend()

    # autolabel(rects1, 1000)

    # fig.tight_layout()

    # plt.show()

    
if __name__ == "__main__":
    main()