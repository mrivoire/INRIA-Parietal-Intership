"""
Experiments on real data with categorical variables

"""
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
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
from SPP import SPPRegressor
from sklearn.model_selection import KFold
# from pandas_profiling import ProfileReport

#################################################
#          Preprocess Numeric Features
#################################################


def numeric_features(X):
    numeric_feats = X.dtypes[
        (dtype.kind in 'if' for dtype in X.dtypes)
    ].index
    return numeric_feats


##################################################
#          Preprocess Categorical Features
##################################################


def categorical_features(X):
    categorical_feats = X.dtypes[((X.dtypes == "object")
                                  | (X.dtypes == "category"))].index
    return categorical_feats


##################################################
#          Preprocess Time Features
##################################################


def time_features(X):
    time_feats = X.dtypes[(X.dtypes == "datetime64[ns]")].index
    return time_feats


#######################################################
#                   Time Conversion
#######################################################


def time_convert(X):
    X = X.copy()

    for col, dtype in zip(X.columns, X.dtypes):
        if X[col].dtypes == "datetime64[ns]":
            col_name = str(col) + '_'
            X[col_name + 'year'] = X[col].dt.year
            X[col_name + 'weekday'] = X[col].dt.dayofweek
            X[col_name + 'yearday'] = X[col].dt.dayofyear
            X[col_name + 'time'] = X[col].dt.minute
            X = X.drop(columns=[col], axis=1)

    return X


################################################################
#                       Grid Search
################################################################


def get_models(X, **kwargs):
    # Pipeline

    time_feats = time_features(X)
    time_transformer = FunctionTransformer(time_convert)

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
        ('cat', categorical_transformer, categorical_feats),
        ('time', time_transformer, time_feats)]
    )

    rf_preprocessor = ColumnTransformer(transformers=[
        ('num', rf_numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats),
        ('time', time_transformer, time_feats)]
    )

    models = {}
    tuned_parameters = {}

    # Lasso
    lasso = Lasso(**kwargs)
    models['lasso'] = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', lasso)])
    lmbdas = np.logspace(-4, -0.5, 3)
    tuned_parameters['lasso'] = \
        {'regressor__lmbda': lmbdas, 'preprocessor__num__binning__n_bins': [2, 3, 4, 5]}

    # LassoCV
    models['lasso_cv'] = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', linear_model.LassoCV())])
    tuned_parameters['lasso_cv'] = {'preprocessor__num__binning__n_bins': [2, 3, 4, 5]}

    # RidgeCV
    models['ridge_cv'] = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', linear_model.RidgeCV())])
    tuned_parameters['ridge_cv'] = {'preprocessor__num__binning__n_bins': [2, 3, 4, 5]}

    # XGBoost
    xgb = XGBRegressor()
    models['xgb'] = Pipeline(steps=[('preprocessor', rf_preprocessor),
                                    ('regressor', xgb)])
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    tuned_parameters['xgb'] = {'regressor__alpha': alphas,
                               'regressor__n_estimators': [30, 100]}
    
    # Random Forest
    rf = RandomForestRegressor()
    models['rf'] = Pipeline(steps=[('preprocessor', rf_preprocessor),
                                   ('regressor', rf)])
    tuned_parameters['rf'] = {'regressor__max_depth': [2, 3, 4, 5]}

    # SPP Regressor
    spp_reg = SPPRegressor(**kwargs)
    models['spp_reg'] = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', spp_reg)])

    tuned_parameters['spp_reg'] = \
        {'preprocessor__num__binning__n_bins': [2, 3, 4, 5],
         'regressor__max_depth': [2, 3, 4, 5]}

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

    n_splits: intprint("X before = ", X)
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
        cv_scores[name] = \
            cross_val_score(model, X, y, cv=n_splits, n_jobs=n_jobs).mean()
    return cv_scores


def compute_gs(X, y, models, tuned_parameters, n_splits, n_jobs=1, **kwargs):
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
        if model != 'spp_reg':
            gs = \
                GridSearchCV(model, cv=n_splits,
                             param_grid=tuned_parameters[name], n_jobs=n_jobs)

            gs.fit(X, y)

        else:
            kf = KFold(n_splits=n_splits, random_state=None, shuffle=False)
            for train_index, test_index in kf.split(X):
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                spp_reg = SPPRegressor(**kwargs)
                spp_reg.fit(X_train, y_train)
                y_hat = spp_reg.predict(X_test)
                R_square = spp_reg.score(X_test, y_test)

                gs = {'cv_scores': [], 'y_hat': []}
                gs['cv_scores'].append(R_square)
                gs['y_hat'].append(y_hat)
        
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

    datasets = ['auto_prices', 'lacrimes', 'black_friday', 
                'nyc_taxi', 'housing_prices']

    start = time.time()
    for i in range(len(datasets)):
        if datasets[i] == 'auto_prices':
            X, y = load_auto_prices()
            data_name = 'auto_prices'
        if datasets[i] == 'lacrimes':
            X, y = load_lacrimes()
            data_name = 'lacrimes'
        if datasets[i] == 'black_friday':
            X, y = load_black_friday()
            data_name = 'black_friday'
        if datasets[i] == 'nyc_taxi':
            X, y = load_nyc_taxi()
            data_name = 'nyc_taxi'
        if datasets[i] == 'housing_prices':
            load_housing_prices()
            data_name = 'housing_prices'

        X = X[:10]
        y = y[:10]
        models, tuned_parameters = get_models(
                X,
                lmbda=lmbda,
                epsilon=epsilon,
                f=f, n_epochs=n_epochs,
                screening=screening,
                store_history=store_history)

        print('models = ', models)

        cv_scores = compute_cv(X=X, y=y, models=models, n_splits=n_splits, 
                               n_jobs=n_jobs)

        print("cv_scores = ", cv_scores)

        list_cv_scores = []

        for k, v in cv_scores.items():
            print(f'{k}: {v}')
            list_cv_scores.append(v)

        print("cv_scores without tuning params = ", list_cv_scores)

        gs_scores = compute_gs(X=X, y=y, models=models, n_splits=n_splits,
                               tuned_parameters=tuned_parameters, n_jobs=n_jobs)

        list_gs_scores = []
        scores = pd.DataFrame({'model': [],
                            'best_cv_score': [],
                            'best_param': []})

        execution_time_list = []
        for k, v in gs_scores.items():
            print(f'{k} -- best params = {v.best_params_}')
            print(f'{k} -- cv scores = {v.best_score_}')
            list_gs_scores.append(v.best_score_)
            end = time.time()
            delay = end - start
            execution_time_list.append(delay)
            scores = scores.append(pd.DataFrame({'model': [k],
                                                'best_cv_score': [v.best_score_],
                                                'best_param': [v.best_params_],
                                                'delay': [delay]}))

        print("Housing Prices Dataset with 100 samples")
        print("cv_score with tuning params = ", list_gs_scores)

        print(scores)
        scores.to_csv('/home/mrivoire/Documents/M2DS_Polytechnique/INRIA-Parietal-Intership/Code/' + data_name + '.csv', index=False)

        #######################################################################
        #                         Bar Plots CV Scores
        #######################################################################

        labels = ['Lasso', 'Lasso_cv', 'Ridge_cv', 'XGB', 'RF']

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x, list_gs_scores, width)
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('CV Scores')
        ax.set_title('Crossval Scores By Predictive Model With Tuning For 100 Samples')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        autolabel(rects1, 1000)

        fig.tight_layout()

        plt.show()

        #######################################################################
        #                         Bar Plots CV Time
        #######################################################################

        labels = ['Lasso', 'Lasso_cv', 'Ridge_cv', 'XGB', 'RF']

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x, execution_time_list, width)
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Running Time')
        ax.set_title('Running Time By Predictive Model With Tuning For 100 Samples')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        autolabel(rects1, 1000)

        fig.tight_layout()

        plt.show()


if __name__ == "__main__":
    main()
