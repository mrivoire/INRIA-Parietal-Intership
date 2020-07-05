import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from numba import njit
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from scipy.sparse import issparse
from sklearn.linear_model import Lasso as sklearn_Lasso
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV

from cd_solver_lasso_numba import Lasso, cyclic_coordinate_descent, sparse_cd


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


def compute_cv(X_train, y_train, n_splits, lmbda, epsilon, f, n_epochs,
               screening, store_history):

   
    # Pipeline
    numeric_feats = numeric_features(X_train)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('binning', KBinsDiscretizer(n_bins=3, encode='onehot',
                                     strategy='quantile'))])

    categorical_feats = categorical_features(X_train)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)

    scores_lasso = []
    scores_xgb = []
    scores_rf = []
    X_train = X_train.to_records()
    print(X_train.array.shape)
    y_train = y_train.to_numpy()
    for fold in kf.split(X_train):
        
        X_tr = X_train[fold[0]]
        y_tr = y_train[fold[0]]
        X_te = X_train[fold[1]]
        y_te = y_train[fold[1]]

        lasso = Lasso(lmbda=lmbda, epsilon=epsilon, f=f, n_epochs=n_splits,
                      screening=screening,
                      store_history=store_history)

        pipe_lasso = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', lasso)])

        print(X_tr.shape)                                     

        pipe_lasso.fit(X_tr, y_tr)

        scores_lasso.append(pipe_lasso.score(X_te, y_te))

        xgb = XGBRegressor()

        pipe_xgb = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', xgb)])

        pipe_xgb.fit(X_tr, y_tr)

        scores_xgb.append(pipe_xgb.score(X_te, y_te))

        rf = RandomForestRegressor()

        pipe_rf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', rf)])

        pipe_rf.fit(X_tr, y_tr)

        scores_rf.append(pipe_rf.score(X_te, y_te))

    cv_lasso = scores_lasso.mean()
    cv_xgb = scores_xgb.mean()
    cv_rf = scores_rf.mean()

    return cv_lasso, cv_xgb, cv_rf


def main():
    # data_dir = "../Datasets"
    data_dir = "/home/mrivoire/Documents/M2DS_Polytechnique/Stage_INRIA/Datasets"
    fname_train = data_dir + "/housing_prices_train"
    fname_test = data_dir + "/housing_prices_test"
    train_set = read_csv(fname_train)
    head_train = train_set.head()
    test_set = read_csv(fname_test)
    head_test = test_set.head()

    lmbda = 1.
    epsilon = 1e-15
    f = 10
    n_splits = 5
    screening = True
    store_history = True
    n_epochs = 1000
    # print("Housing Prices Training Set Header : ", head_train)
    # print("Housing Prices Testing Set Header : ", head_test)

    all_data = concat(train_set, test_set)
    head_all_data = all_data.head()
    # print("All data : ", head_all_data)

    log_sale_price = log_transform(train_set['SalePrice'])
    # print("log target : ", log_sale_price)

    # skewed_feats = skewness(train_set)
    # # print("skewed features : ", skewed_feats)

    # numeric_feats = numeric_features(all_data)
    # # print("numeric features : ", numeric_feats)

    # numeric_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='median')),
    #     ('scaler', StandardScaler()),
    #     ('binning', KBinsDiscretizer(n_bins=3, encode='onehot',
    #                                  strategy='quantile'))])

    # print("numeric transformer = ", numeric_transformer)

    # categorical_feats = categorical_features(all_data)
    # categorical_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    #     ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # print("categorical transformer = ", categorical_transformer)

    # preprocessor = ColumnTransformer(transformers=[
    #     ('num', numeric_transformer, numeric_feats),
    #     ('cat', categorical_transformer, categorical_feats)])

    # print("preprocessor : ", preprocessor)

    # n_features = len(numeric_feats) + len(categorical_feats)

    # # Append Lasso to preprocessing pipeline.
    # # Now we have a full prediction pipeline.
    # pipeline = Pipeline(steps=[('preprocessor', preprocessor),
    #                     ('regressor', sklearn_Lasso(alpha=1,
    #                                     fit_intercept=False,
    #                                     normalize=False,
    #                                     max_iter=2000,
    #                                     tol=1e-5))])

    X_train, X_test, y_train = split_train_test(all_data, train_set)

    # pipeline.fit(X_train, y_train)

    # pipeline.predict(X_test)

    cv_lasso, cv_xgb, cv_rf = compute_cv(X_train=X_train, y_train=y_train, 
                                         n_splits=n_splits, lmbda=lmbda, 
                                         epsilon=epsilon, f=f, 
                                         n_epochs=n_epochs,
                                         screening=screening, 
                                         store_history=store_history)

    # lasso.fit(X_train, y_train)
    # print("model score: %.3f" % clf.score(X_test, y_test))

    # all_data = onehot_encoding(all_data)
    # # print("onehot data : ", all_data)

    # prop_NaN = proportion_NaN(all_data)
    # # print("proportion NaN values :", prop_NaN)

    # all_data = fill_NaN(all_data)
    # # print(all_data.head())

    # X_train, X_test, y_train = split_train_test(all_data, train_set)

    # print("X_train :", X_train.head())
    # print("X_test : ", X_test.head())
    # print("y_train : ", y_train.head())

    # Tests with dense features matrices

    # X = X_train.to_numpy()
    # y = y_train.to_numpy()

    # print("shape of X_train : ", X.shape)
    # print("columns of X_train", X_train.columns)

    # lmbda = 1.
    # f = 10
    # epsilon = 1e-14
    # n_epochs = 100000
    # screening = True
    # store_history = True

    # lasso = Lasso()
    # scores = cross_val_score(lasso, X, y, cv=5)
    # print(scores)

    # dense_lasso = Lasso(lmbda=lmbda, epsilon=epsilon, f=f,
    #                     n_epochs=n_epochs, screening=screening,
    #                     store_history=store_history).fit(X, y)

    # dense_cv_score = dense_lasso.score(X, y)

    # print("dense cv score : ", dense_cv_score)

    # dense_lasso_sklearn = sklearn_Lasso(alpha=lmbda / len(X),
    #                                     fit_intercept=False,
    #                                     normalize=False, max_iter=n_epochs,
    #                                     tol=1e-15).fit(X, y)

    # dense_cv_sklearn = dense_lasso_sklearn.score(X_train, y)
    # print("dense cv score sklearn : ", dense_cv_sklearn)

    # n_bins = 3
    # encode = 'onehot'
    # strategy = 'quantile'
    # enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    # X_binned = enc.fit_transform(X)
    # X_binned = X_binned.tocsc()

    # sparse_lasso_sklearn = sklearn_Lasso(alpha=lmbda / len(X),
    #                                      fit_intercept=False,
    #                                      normalize=False, max_iter=n_epochs,
    #                                      tol=1e-15).fit(X_binned, y)

    # sparse_cv_sklearn = sparse_lasso_sklearn.score(X_binned, y)

    # print("sparse cv score sklearn : ", sparse_cv_sklearn)

    # sparse_lasso = Lasso(lmbda=lmbda, epsilon=epsilon, f=f,
    #                      n_epochs=n_epochs,
    #                      screening=screening,
    #                      store_history=store_history).fit(X, y)

    # sparse_cv_score = sparse_lasso.score(X_binned, y)

    # print("sparse crossval score : ", sparse_cv_score)

    # print("X type : ", type(X))

    # Plots
    # matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    # prices = pd.DataFrame({"price":train_set["SalePrice"],
    #                        "log(price + 1)":np.log1p(train_set["SalePrice"])})
    # prices.hist()


if __name__ == "__main__":
    main()
