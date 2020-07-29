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
            # X = X['Age']
            columns_kept_age.append('Age')

    return X, columns_kept_age


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
    print("X = ", X.dtypes)

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

    #######################################################
    #               Read All CSVs          
    #######################################################
    
    (X_auto_raw, y_auto_raw, X_crimes_raw, y_crimes_raw, X_nyc_taxi_raw,
        y_nyc_taxi_raw, X_black_friday_raw, y_black_friday_raw, 
        auto_price_rawdata, crimes_rawdata, nyc_taxi_rawdata, 
        black_friday_rawdata) = read_datasets()

    #############################################################
    #              Preprocess Numeric Features 
    #############################################################

    X, num_feats = numeric_features(X_auto_raw)

    #############################################################
    #              Preprocess Numeric Features 
    #############################################################

    X, cat_feats = categorical_features(X_crimes_raw)
    # print("X = ", X.dtypes)

    #############################################################
    #              Preprocess Age Feature
    #############################################################

    # X, columns_kept_age = age_map(X_black_friday_raw)
    # print("X = ", X.dtypes)

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

    # X_black_friday, y_black_friday_raw = load_black_friday()
    # print("X_black_friday = ", X_black_friday.dtypes)

    ###########################################################
    #               Load NYC Taxi Dataset
    ###########################################################

    X_NYCTaxi, y_NYCTaxi_raw = load_NYCtaxi()
    # print("X_NYCTaxi Types = ", X_NYCTaxi.dtypes)

if __name__ == "__main__":
    main()