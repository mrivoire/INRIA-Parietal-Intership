"""
Experiments on real data with categorical variables

"""
import itertools
import numbers
from re import error
import sys
import pandas as pd
from dataset import (
    load_auto_prices,
    load_lacrimes,
    load_black_friday,
    load_nyc_taxi,
    load_housing_prices,
)

from grid_search import get_models, compute_gs
from bar_plots import bar_plots
from estimation_n_interfeats import estimate_n_interfeats
from SPP import max_val
from sklearn.preprocessing import KBinsDiscretizer


def main():

    data = sys.argv[1]
    if len(sys.argv) > 2:
        n_samples = sys.argv[2]
    else:
        n_samples = None

    ####################################################################
    #                           Parameters
    ####################################################################

    lmbda_lasso = 1.0
    epsilon = 1e-7
    f = 10
    n_splits = 2
    screening = True
    store_history = True
    n_epochs = 10000
    n_jobs = 1
    # encode = "onehot"
    # strategy = "quantile"
    n_bins = 3
    max_depth = 2
    tol = 1e-08
    n_lambda = 100
    lambda_max_ratio = (1 / 1000)
    # lambdas = [1, 0.5, 0.2, 0.1, 0.01]
    lambdas = None
    # lambdas = [0.1]
    n_active_max = 200

    kwargs_spp = {
        "n_lambda": n_lambda,
        "lambdas": lambdas,
        "max_depth": max_depth,
        "epsilon": epsilon,
        "f": f,
        "n_epochs": n_epochs,
        "tol": tol,
        "lambda_max_ratio": lambda_max_ratio,
        "n_active_max": n_active_max,
        "screening": screening,
        "store_history": store_history,
    }

    kwargs_lasso = {
        "lmbda": lmbda_lasso,
        "epsilon": epsilon,
        "f": f,
        "n_epochs": n_epochs,
        "screening": screening,
        "store_history": store_history,
    }

    n_interfeats = estimate_n_interfeats(n_features=15, n_bins=3, max_depth=6)
    print('n_interfeats = ', n_interfeats)

    if data == 'auto_prices':
        X, y = load_auto_prices()
        n_bins_more_bins = [10]
        max_depth_more_bins = [1, 2, 3, 4]
        n_bins_less_bins = [3]
        max_depth_less_bins = [1, 2, 3, 4, 5, 6]
    elif data == 'black_friday':
        X, y = load_black_friday()
        n_bins_more_bins = [10]
        max_depth_more_bins = [1, 2, 3, 4, 5]
        n_bins_less_bins = [3]
        max_depth_less_bins = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif data == 'housing_prices':
        X, y = load_housing_prices()
        n_bins_more_bins = [10]
        max_depth_more_bins = [1, 2]
        n_bins_less_bins = [3]
        max_depth_less_bins = [1, 2, 3]
    elif data == 'la_crimes':
        X, y = load_lacrimes()
        n_bins_more_bins = [10]
        max_depth_more_bins = [1, 2, 3, 4]
        n_bins_less_bins = [3]
        max_depth_less_bins = [1, 2, 3, 4, 5]
    elif data == 'NYC_taxis':
        X, y = load_nyc_taxi()
        n_bins_more_bins = [10]
        max_depth_more_bins = [1, 2, 3, 4]
        n_bins_less_bins = [3]
        max_depth_less_bins = [1, 2, 3, 4, 5, 6]
    else:
        raise ValueError("Dataset not implemented for " + data)

    if n_samples is not None:
        n_samples = int(n_samples)
        X = X[:n_samples]
        y = y[:n_samples]
    else:
        n_samples = X.shape[0]

    models, tuned_parameters = get_models(
        X=X,
        n_bins=n_bins,
        n_bins_less_bins=n_bins_less_bins,
        max_depth_less_bins=max_depth_less_bins,
        n_bins_more_bins=n_bins_more_bins,
        max_depth_more_bins=max_depth_more_bins,
        kwargs_lasso=kwargs_lasso,
        kwargs_spp=kwargs_spp
    )

    # del models['lasso']
    # del models['lasso_cv']
    # del models['ridge_cv']
    # del models['rf']
    # del models['xgb']

    # del tuned_parameters['lasso']
    # del tuned_parameters['lasso_cv']
    # del tuned_parameters['ridge_cv']
    # del tuned_parameters['rf']
    # del tuned_parameters['xgb']

    gs_models = compute_gs(
        X=X,
        y=y,
        models=models,
        tuned_parameters=tuned_parameters,
        n_splits=n_splits,
        n_lambda=n_lambda,
        lambdas=lambdas,
        max_depth=max_depth,
        epsilon=epsilon,
        f=f,
        n_epochs=n_epochs,
        tol=tol,
        lambda_max_ratio=lambda_max_ratio,
        n_active_max=n_active_max,
        screening=screening,
        store_history=store_history,
        n_jobs=n_jobs,
    )

    # print("gs_models = ", gs_models)
    # best_score_spp_more_bins = gs_models["spp_reg_more_bins"]["best_score"]
    # best_params_spp_more_bins = gs_models["spp_reg_more_bins"]["best_params"]

    # print('best_score_spp_more_bins = ', best_score_spp_more_bins)
    # print('best_params_spp_more_bins = ', best_params_spp_more_bins)

    # best_score_spp_less_bins = gs_models["spp_reg_less_bins"]["best_score"]
    # best_params_spp_less_bins = gs_models["spp_reg_less_bins"]["best_params"]
    # print('best_score_spp_less_bins = ', best_score_spp_less_bins)
    # print('best_params_spp_less_bins = ', best_params_spp_less_bins)

    list_df = []
    for name, model in gs_models.items():
        df = pd.io.json.json_normalize(model, sep='_')
        df["model"] = name
        list_df.append(df)

    results = pd.concat(list_df)
    results['n_features'] = X.shape[1]
    results['n_samples'] = X.shape[0]
    results['data'] = data

    print('results = ', results)

    results.to_csv('/home/mrivoire/Documents/M2DS_Polytechnique/INRIA-Parietal-Intership/Code/' +
                   data + '_results.csv', index=False)

    df = pd.read_csv('/home/mrivoire/Documents/M2DS_Polytechnique/INRIA-Parietal-Intership/Code/' +
                     data + '_results.csv')

    df.head()

    bar_plots(df=df)

    #######################################################################
    #                         Bar Plots CV Scores
    #######################################################################

    ######################################################################
    #                  Auto Label Function For Bar Plots
    ######################################################################

    # def autolabel(rects, scale):
    #     """Attach a text label above each bar in *rects*, displaying its
    #     height.
    #     """

    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate(
    #             "{}".format(round(height * scale, 0) / scale),
    #             xy=(rect.get_x() + rect.get_width() / 2, height),
    #             xytext=(0, 3),  # 3 points vertical offset
    #             textcoords="offset points",
    #             ha="center",
    #             va="bottom",
    #         )

    # labels = ["Lasso", "Lasso_cv", "Ridge_cv", "XGB", "RF"]

    # x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x, list_gs_scores, width)
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel("CV Scores")
    # ax.set_title("Crossval Scores By Predictive Model With Tuning For 100 Samples")
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # ax.legend()

    # autolabel(rects1, 1000)

    # fig.tight_layout()

    # plt.show()

    #######################################################################
    #                         Bar Plots CV Time
    #######################################################################

    # labels = ["Lasso", "Lasso_cv", "Ridge_cv", "XGB", "RF"]

    # x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(x, execution_time_list, width)
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel("Running Time")
    # ax.set_title("Running Time By Predictive Model With Tuning For 100 Samples")
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    # ax.legend()

    # autolabel(rects1, 1000)

    # fig.tight_layout()

    # plt.show()


if __name__ == "__main__":
    main()
