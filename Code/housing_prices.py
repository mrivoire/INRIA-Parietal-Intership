import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
# from sklearn.linear_model import Lasso
# from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import Lasso as sklearn_Lasso
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.model_selection import train_test_split, GridSearchCV

from cd_solver_lasso_numba import Lasso

#######################################
#              Read CSV
#######################################


def read_csv(filePath):
    data = pd.read_csv(filePath + ".csv")
    return data


########################################
#               All Data
########################################


def concat(X_train, X_test):

    all_data = pd.concat((X_train.loc[:, 'MSSubClass':'SaleCondition'],
                          X_test.loc[:, 'MSSubClass':'SaleCondition']))

    return all_data


#################################################
#           Log transform of the target
#################################################


def log_transform(feature):
    log_feat = np.log1p(feature)

    return log_feat


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


###################################################
#           Skewed Features
###################################################


def skewness(dataset):
    numeric_feats = numeric_features(dataset)
    skewed_feats = dataset[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    return skewed_feats


##########################################################
#     One-hot encoding of the categorical features
##########################################################


def onehot_encoding(dataset):
    onehot_data = pd.get_dummies(dataset)

    return onehot_data


###########################################################
#                       Nan Values
###########################################################


def proportion_NaN(dataset):
    prop = dataset.isna().mean()

    return prop


def fill_NaN(dataset):
    filled_data = dataset.fillna(dataset.mean())

    return filled_data


###############################################################
#                       Split Train Test
###############################################################


def split_train_test(dataset, train):
    X_train = dataset[:train.shape[0]]
    X_test = dataset[train.shape[0]:]
    y = train.SalePrice

    return X_train, X_test, y


################################################################
#                       Folds
################################################################


def compute_cv(X, y, n_splits, lmbda, epsilon, f, n_epochs,
               screening, store_history):
    """
    Parameters
    ----------
    X: numpy.ndarray(), shape = (n_samples, n_features)
        features matrix

    y: numpy.array(), shape = '(n_samples, )
        target vector

    n_splits: int
        number of folds

    lmbda: float
        regularization parameter

    epsilon: float
        tolerance

    f: int
        frequency

    n_epochs: int
        number of epochs

    screening: bool
        indicates if we run the solver with or without screening process

    store_history: bool
        indicates if we store the history variables when we run the solver

    Returns
    -------
    cv_lasso: float
        cross validation score of the lasso solver

    cv_xgb: float
        cross validation score of the xgboost solver

    cv_rf: float
        cross validation score of the random forest solver

    """
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

    y = y.to_numpy().astype('float')

    # Lasso
    lasso = Lasso(lmbda=lmbda, epsilon=epsilon, f=f, n_epochs=n_epochs,
                  screening=screening, store_history=store_history)
    pipe_lasso = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', lasso)])
    cv_lasso = cross_val_score(pipe_lasso, X, y, cv=n_splits).mean()

    # XGBoost
    xgb = XGBRegressor()
    pipe_xgb = Pipeline(steps=[('preprocessor', rf_preprocessor),
                               ('regressor', xgb)])
    cv_xgb = cross_val_score(pipe_xgb, X, y, cv=n_splits).mean()

    # Random Forest
    rf = RandomForestRegressor()
    pipe_rf = Pipeline(steps=[('preprocessor', rf_preprocessor),
                              ('regressor', rf)])
    cv_rf = cross_val_score(pipe_rf, X, y, cv=n_splits).mean()

    return cv_lasso, cv_xgb, cv_rf


def main():
    data_dir = "./Datasets"
    # data_dir = "/home/mrivoire/Documents/M2DS_Polytechnique/Stage_INRIA/Datasets"
    fname_train = data_dir + "/housing_prices_train"
    # fname_test = data_dir + "/housing_prices_test"
    X_train = read_csv(fname_train)
    # X_test = read_csv(fname_test)

    X = X_train  # keep only train data
    y = X['SalePrice']
    X = X.drop('SalePrice', axis=1)

    lmbda = 1.
    epsilon = 1e-7
    f = 10
    n_splits = 5
    screening = True
    store_history = True
    n_epochs = 10000

    cv_lasso, cv_xgb, cv_rf = compute_cv(X=X, y=y,
                                         n_splits=n_splits, lmbda=lmbda,
                                         epsilon=epsilon, f=f,
                                         n_epochs=n_epochs,
                                         screening=screening,
                                         store_history=store_history)

    print("cv lasso = ", cv_lasso)
    print("cv xgb = ", cv_xgb)
    print("cv rf = ", cv_rf)

    # Plots
    # matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    # prices = pd.DataFrame({"price":train_set["SalePrice"],
    #                        "log(price + 1)":np.log1p(train_set["SalePrice"])})
    # prices.hist()


if __name__ == "__main__":
    main()
