import numpy as np
import pandas as pd
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
    # data_dir = "./Datasets"
    data_dir = "/home/mrivoire/Documents/M2DS_Polytechnique/Stage_INRIA/Datasets"
    fname_train = data_dir + "/Housing_Prices/housing_prices_train"
    # fname_test = data_dir + "/housing_prices_test"
    X_train = read_csv(fname_train)
    # X_test = read_csv(fname_test)

    X = X_train # keep only train data
    y = X['SalePrice']
    X = X.drop('SalePrice', axis=1)

    lmbda = 1.
    epsilon = 1e-7
    f = 10
    n_splits = 5
    screening = True
    store_history = True
    n_epochs = 10000
    n_jobs = 4

    numeric_feats = numeric_features(X_train)
    print("numeric feats :", numeric_feats)
    print("type : ", type(numeric_feats))
    numeric_feats = numeric_feats.to_numpy()
    print("new type : ", type(numeric_feats))

    for feat in enumerate(numeric_feats):
        print("feat = ", X_train[feat[1]])
    1/0

    # models, tuned_parameters = get_models(X, lmbda=lmbda, epsilon=epsilon, f=f,
    #                                       n_epochs=n_epochs,
    #                                       screening=screening,
    #                                       store_history=store_history)

    # cv_scores = compute_cv(X=X, y=y, models=models, n_splits=n_splits,
    #                        n_jobs=n_jobs)
    # list_cv_scores = []

    # for k, v in cv_scores.items():
    #     print(f'{k}: {v}')
    #     list_cv_scores.append(v)

    # print("cv_scores without tuning params = ", cv_scores)

    # gs_scores = compute_gs(X=X, y=y, models=models, n_splits=n_splits,
    #                        tuned_parameters=tuned_parameters, n_jobs=n_jobs)

    # list_gs_scores = []
    # for k, v in gs_scores.items():
    #     print(f'{k} -- best params = {v.best_params_}')
    #     print(f'{k} -- cv scores = {v.best_score_}')
    #     list_gs_scores.append(v.best_score_)

# Bar Plots For CV Scores

    # labels = ['Lasso', 'Lasso_cv', 'Ridge_cv', 'XGB', 'RF']

    # x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x, list_cv_scores, width)
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('CV Scores')
    # ax.set_title('Crossval Scores By Predictive Model With Tuning')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # ax.legend()

    # def autolabel(rects, scale):
    #     """Attach a text label above each bar in *rects*, displaying its
    #     height.
    #     """

    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(round(height * scale, 0)/scale),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')

    # autolabel(rects1, 1000)

    # fig.tight_layout()

    # plt.show()


#     lmbdas = np.logspace(-4, -0.5, 30)
#     tuned_parameters = [{'alpha': lmbdas}]
#     # lmbdas_list = [1]
#     cv_scores_list = list()
#     for lmbda in lmbdas_list:
#         (cv_scores,
#          pipe_lasso,
#          pipe_lasso_cv,
#          pipe_ridge_cv,
#          pipe_xgb,
#          pipe_rf) = compute_cv(X=X, y=y,
#                                n_splits=n_splits, lmbda=lmbda,
#                                epsilon=epsilon, f=f,
#                                n_epochs=n_epochs,
#                                screening=screening,
#                                store_history=store_history)

#         cv_scores_list.append(cv_scores)

#     # clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
#     # clf.fit(X, y)
#     # scores = clf.cv_results_['mean_test_score']
#     # scores_std = clf.cv_results_['std_test_score']
#     # plt.figure().set_size_inches(8, 6)
#     # plt.semilogx(alphas, scores)

#     cv_lasso = []
#     cv_lasso_sk = []
#     cv_ridge_sk = []
#     cv_xgb = []
#     cv_rf = []

#     for i in range(len(cv_scores_list)):
#         cv_lasso.append(cv_scores_list[i]['lasso'])
#         cv_lasso_sk.append(cv_scores_list[i]['lasso_cv'])
#         cv_ridge_sk.append(cv_scores_list[i]['ridge_cv'])
#         cv_xgb.append(cv_scores_list[i]['xgb'])
#         cv_rf.append(cv_scores_list[i]['rf'])

#     print("cv lasso = ", cv_lasso)
#     print("cv lasso sk = ", cv_lasso_sk)
#     print("cv ridge sk = ", cv_ridge_sk)
#     print("cv xgb = ", cv_xgb)
#     print("cv rf = ", cv_rf)

#     # def autolabel(rects, scale):
#     #     """Attach a text label above each bar in *rects*, displaying its
#     #     height.
#     #     """

#     #     for rect in rects:
#     #         height = rect.get_height()
#     #         ax.annotate('{}'.format(round(height * scale, 0)/scale),
#     #                     xy=(rect.get_x() + rect.get_width() / 2, height),
#     #                     xytext=(0, 3),  # 3 points vertical offset
#     #                     textcoords="offset points",
#     #                     ha='center', va='bottom')

#     # labels = ['0.1', '0.3', '0.5', '0.7', '0.9', '1', '1.5', '2', '2.5', '3']
#     # x = np.arange(len(labels))
#     # width = 0.35  # the width of the bars
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # rects1 = ax.bar(x - width, cv_lasso, width,
#     #                 label='cv lasso')
#     # # rects2 = ax.bar(x - width*(1/5), cv_lasso_sk, width,
#     #                 label='cv lasso sk')
#     # rects3 = ax.bar(x, cv_ridge_sk, width,
#     #                 label='cv ridge sk')
#     # rects4 = ax.bar(x + width*(1/5), cv_xgb, width,
#     #                 label='cv xgb')
#     # rects5 = ax.bar(x + width*(2/5), cv_rf, width,
#     #                 label='cv rf')
#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     # ax.set_xlabel('lambda')
#     # ax.set_ylabel('crossval score')
#     # ax.set_title('crossval score vs lambdas')
#     # ax.set_xticks(x)
#     # ax.set_xticklabels(labels)
#     # ax.legend()

#     # autolabel(rects1, 1000)
#     # autolabel(rects2, 1000)
#     # autolabel(rects3, 1000)
#     # autolabel(rects4, 1000)
#     # autolabel(rects5, 1000)

#     # fig.tight_layout()
#     # plt.show()

#     # cv_scores_list = list()
#     # for n_epochs in range(1000, 10000, 1000):
#     #     (cv_scores,
#     #      pipe_lasso,
#     #      pipe_lasso_cv,
#     #      pipe_ridge_cv,
#     #      pipe_xgb,
#     #      pipe_rf) = compute_cv(X=X, y=y,
#     #                            n_splits=n_splits, lmbda=lmbda,
#     #                            epsilon=epsilon, f=f,
#     #                            n_epochs=n_epochs,
#     #                            screening=screening,
#     #                            store_history=store_history)

#     #     cv_scores_list.append(cv_scores)

#     # cv_lasso_2 = []
#     # cv_lasso_sk_2 = []
#     # cv_ridge_sk_2 = []
#     # cv_xgb_2 = []
#     # cv_rf_2 = []

#     # for i in range(len(cv_scores_list)):
#     #     cv_lasso_2.append(cv_scores_list[i]['lasso'])
#     #     cv_lasso_sk_2.append(cv_scores_list[i]['lasso_cv'])
#     #     cv_ridge_sk_2.append(cv_scores_list[i]['ridge_cv'])
#     #     cv_xgb_2.append(cv_scores_list[i]['xgb'])
#     #     cv_rf_2.append(cv_scores_list[i]['rf'])

#     # x = np.arange(len(range(1000, 10000, 1000)))
#     # labels = range(1000, 10000, 1000)
#     # width = 0.35  # the width of the bars
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # rects1 = ax.bar(x - width*(2/5), cv_lasso_2, width,
#     #                 label='cv lasso')
#     # rects2 = ax.bar(x - width*(1/5), cv_lasso_sk_2, width,
#     #                 label='cv lasso sk')
#     # rects3 = ax.bar(x, cv_ridge_sk_2, width,
#     #                 label='cv ridge sk')
#     # rects4 = ax.bar(x + width*(1/5), cv_xgb_2, width,
#     #                 label='cv xgb')
#     # rects5 = ax.bar(x + width*(2/5), cv_rf_2, width,
#     #                 label='cv rf')
#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     # ax.set_xlabel('n_epochs')
#     # ax.set_ylabel('crossval score')
#     # ax.set_title('crossval score vs number of epochs')
#     # ax.set_xticks(x)
#     # ax.set_xticklabels(labels)
#     # ax.legend()

#     # autolabel(rects1, 100000)
#     # autolabel(rects2, 1000)
#     # autolabel(rects3, 1000)
#     # autolabel(rects4, 1000)
#     # autolabel(rects5, 1000)

#     # fig.tight_layout()
#     # plt.show()


#     # Plots
#     # matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
#     # prices = pd.DataFrame({"price":train_set["SalePrice"],
#     #                        "log(price + 1)":np.log1p(train_set["SalePrice"])})
#     # prices.hist()


if __name__ == "__main__":
    main()
