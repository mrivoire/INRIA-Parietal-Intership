import numpy as np
import pandas as pd
from sklearn import datasets
import joblib
import faulthandler
import time
import datetime

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
from tests_real_data import (
    numeric_features, categorical_features, time_features_lacrimes,
    time_features_NYCTaxi, age_map)

######################################################
#           Load Auto Prices Dataset
######################################################


def load_auto_prices():
    faulthandler.enable()
    mem = joblib.Memory(location='cache')

    openml_id_auto_prices = {
        # https://www.openml.org/d/1189 BNG(auto_price)
        # No high-cardinality categories
        'BNG(auto_price)': 1189
        }

    raw_data = dict()
    for name, openml_id in openml_id_auto_prices.items():
        X, y = mem.cache(datasets.fetch_openml)(
                                data_id=openml_id, return_X_y=True,
                                as_frame=True)

        raw_data['BNG(auto_price)'] = X, y
        X = X.copy()

    auto_price_rawdata = raw_data['BNG(auto_price)']

    X_auto_raw = auto_price_rawdata[0]
    y_auto_raw = auto_price_rawdata[1]

    X_auto_prices_num, num_feats = numeric_features(X_auto_raw)
    X_auto_prices_num = X_auto_prices_num.select_dtypes(exclude=['category'])
    X_auto_prices_cat, cat_feats = categorical_features(X_auto_raw)
    X_auto_prices_cat = X_auto_prices_cat.select_dtypes(exclude=['float64'])

    X = pd.concat([X_auto_prices_cat, X_auto_prices_num], axis=1)

    return X, y_auto_raw


######################################################
#             Load Crimes Dataset
######################################################


def load_lacrimes():
    faulthandler.enable()
    mem = joblib.Memory(location='cache')

    raw_data = dict()

    openml_id_lacrimes = {
        # https://www.openml.org/d/42160
        # A few high-cardinality strings and some dates
        'la_crimes': 42160
    }

    for name, openml_id in openml_id_lacrimes.items():
        X, y = mem.cache(datasets.fetch_openml)(
                                    data_id=openml_id, return_X_y=True,
                                    as_frame=True)
        raw_data[name] = X, y
        X = X.copy()

    crimes_rawdata = raw_data['la_crimes']

    X_crimes_raw = crimes_rawdata[0]
    y_crimes_raw = crimes_rawdata[1]

    X_lacrimes_num, num_feats = numeric_features(X_crimes_raw)
    X_lacrimes_num = X_lacrimes_num.select_dtypes(exclude=['category',
                                                           'object'])

    X_lacrimes, time_feats = time_features_lacrimes(X_crimes_raw)
    X_lacrimes_time = X_lacrimes.select_dtypes(exclude=['category',
                                                        'object',
                                                        'float64'])

    X_lacrimes_cat, cat_feats = categorical_features(X_lacrimes)
    X_lacrimes_cat = X_lacrimes_cat.select_dtypes(exclude=['float64',
                                                           'datetime64[ns]'])

    X = pd.concat([X_lacrimes_num, X_lacrimes_cat, X_lacrimes_time], axis=1)

    return X, y_crimes_raw


######################################################
#          Load Black Friday Dataset
######################################################


def load_black_friday():
    faulthandler.enable()
    mem = joblib.Memory(location='cache')

    raw_data = dict()

    openml_id_black_friday = {
        # https://www.openml.org/d/41540
        'black_friday': 41540
    }

    for name, openml_id in openml_id_black_friday.items():
        X, y = mem.cache(datasets.fetch_openml)(
                                    data_id=openml_id, return_X_y=True,
                                    as_frame=True)
        raw_data[name] = X, y
        X = X.copy()

    black_friday_rawdata = raw_data['black_friday']

    X_black_friday_raw = black_friday_rawdata[0]
    y_black_friday_raw = black_friday_rawdata[1]

    X_blf, age_feat = age_map(X_black_friday_raw)
    X_blf_num, num_feats = numeric_features(X_blf)
    X_blf_num = X_blf_num.select_dtypes(exclude=['category', 'object'])
    X_blf_cat, cat_feats = categorical_features(X_blf)
    X_blf_cat = X_blf_cat.select_dtypes(exclude=['float64'])

    X = pd.concat([X_blf_num, X_blf_cat])

    return X, y_black_friday_raw


######################################################################
#                       Load NYC Taxi Dataset
######################################################################

def load_NYCtaxi():

    faulthandler.enable()
    mem = joblib.Memory(location='cache')

    raw_data = dict()

    openml_id_nyc_taxi = {
        # https://www.openml.org/d/42208 nyc-taxi-green-dec-2016
        # No high cardinality strings, a few datetimes
        # Skipping because I cannot get it to encode right
        'nyc-taxi-green-dec-2016': 42208
        }

    for name, openml_id in openml_id_nyc_taxi.items():
        X, y = mem.cache(datasets.fetch_openml)(
                                    data_id=openml_id, return_X_y=True,
                                    as_frame=True)
        raw_data[name] = X, y
        X = X.copy()

    nyc_taxi_rawdata = raw_data['nyc-taxi-green-dec-2016']

    X_NYCTaxi_raw = nyc_taxi_rawdata[0]
    y_NYCTaxi_raw = nyc_taxi_rawdata[1]

    X_NYCTaxi_num, num_feats = numeric_features(X_NYCTaxi_raw)
    X_NYCTaxi_num = X_NYCTaxi_num.select_dtypes(exclude=['category', 'object'])
    X_NYCTaxi, time_feats = time_features_NYCTaxi(X_NYCTaxi_raw)
    X_NYCTaxi_time = X_NYCTaxi.select_dtypes(exclude=['category',
                                                      'object',
                                                      'float64'])
    X_NYCTaxi_cat, cat_feats = categorical_features(X_NYCTaxi)
    X_NYCTaxi_cat = X_NYCTaxi_cat.select_dtypes(exclude=['float64',
                                                         'datetime64[ns]'])
    X = pd.concat([X_NYCTaxi_cat, X_NYCTaxi_num, X_NYCTaxi_time])
    X, cat_feats = categorical_features(X)

    return X, y_NYCTaxi_raw


def main():

    ########################################################
    #               Load Auto Prices Dataset
    ########################################################

    X_auto_prices, y_auto_prices_raw = load_auto_prices()

    ########################################################
    #               Load LA Crimes Dataset
    ########################################################

    X_lacrimes, y_lacrimes_raw = load_lacrimes()
    # print("X = ", X_lacrimes.dtypes)

    #########################################################
    #           Load Black Friday Dataset
    #########################################################

    X_black_friday, y_black_friday_raw = load_black_friday()
    # print("X_black_friday = ", X_black_friday.dtypes)

    ###########################################################
    #               Load NYC Taxi Dataset
    ###########################################################

    X_NYCTaxi, y_NYCTaxi_raw = load_NYCtaxi()
    # print("X_NYCTaxi Types = ", X_NYCTaxi.dtypes)

if __name__ == "__main__":
    main()