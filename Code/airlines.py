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


################################################################
#                        Grid Search
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
    # tuned_parameters['xgb'] = {'regressor__alpha': alphas,
    #                            'regressor__n_estimators': [30, 100]}

    tuned_parameters['xgb'] = {'regressor__n_estimators': [30, 100]}

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
    fname_train = data_dir + "/Airlines/airlinedelaycauses_DelayedFlights"
    X_train = read_csv(fname_train)

    # print("X_train : ", X_train)
    # print("X_train shape = ", np.shape(X_train))

    X = X_train.dropna(axis=0, subset=['LateAircraftDelay']).head(200)  # keep only train data
    y = X['LateAircraftDelay']
    X = X.drop(['LateAircraftDelay', 'Unnamed: 0'], axis=1)

    # print("X : ", X)
    # print("X shape = ", np.shape(X))
    # print("y :", y)
    # print("y shape :", np.shape(y))

    lmbda = 1.
    epsilon = 1e-7
    f = 10
    n_splits = 5
    screening = True
    store_history = True
    n_epochs = 10000
    n_jobs = 4

    models, tuned_parameters = get_models(X, lmbda=lmbda, epsilon=epsilon, f=f,
                                          n_epochs=n_epochs,
                                          screening=screening,
                                          store_history=store_history)

    cv_scores = compute_cv(X=X, y=y, models=models, n_splits=n_splits,
                           n_jobs=n_jobs)
    list_cv_scores = []

    for k, v in cv_scores.items():
        print(f'{k}: {v}')
        list_cv_scores.append(v)

    print("cv_scores without tuning params = ", list_cv_scores)

    gs_scores = compute_gs(X=X, y=y, models=models, n_splits=n_splits,
                           tuned_parameters=tuned_parameters, n_jobs=n_jobs)

    list_gs_scores = []
    for k, v in gs_scores.items():
        print(f'{k} -- best params = {v.best_params_}')
        print(f'{k} -- cv scores = {v.best_score_}')
        list_gs_scores.append(v.best_score_)

    print("cv_scores with tuning params = ", list_gs_scores)

    # Bar Plots 
    labels = ['Lasso', 'Lasso_cv', 'Ridge_cv', 'XGB', 'RF']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, list_gs_scores, width)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('CV Scores')
    ax.set_title('Crossval Scores By Predictive Model With Tuning For 200 Samples')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

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

    autolabel(rects1, 1000)

    fig.tight_layout()

    plt.show()
    

if __name__ == "__main__":
    main()
