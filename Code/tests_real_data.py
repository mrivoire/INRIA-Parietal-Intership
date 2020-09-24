"""
Experiments on real data with categorical variables

"""
import pandas as pd
from dataset import (
    load_auto_prices,
    load_lacrimes,
    load_black_friday,
    load_nyc_taxi,
    load_housing_prices,
)

from grid_search import get_models, compute_gs


def main():

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
    n_lambda = 10
    lambda_max_ratio = (1 / 20)
    # lambdas = [1, 0.5, 0.2, 0.1, 0.01]
    lambdas = None
    # lambdas = [0.1]
    n_active_max = 100

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

    X, y = load_auto_prices()

    X = X[:1000]
    y = y[:1000]

    models, tuned_parameters = get_models(
        X=X,
        n_bins=n_bins,
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

    print("gs_models = ", gs_models)
    best_score_spp = gs_models["spp_reg"]["best_score"]
    best_params_spp = gs_models["spp_reg"]["best_params"]

    print('best_score_spp = ', best_score_spp)
    print('best_params = ', best_params_spp)

    list_df = []

    for model in gs_models.keys():
        df = pd.DataFrame(gs_models[model])
        df["model"] = model
        list_df.append(df)

    results = pd.concat(list_df)

    print('gs_models_df = ', results)

    # gs_models_df.columns = [
    #     'best_params_spp', 'best_score_spp']

    results.to_csv(
        r'/home/mrivoire/Documents/M2DS_Polytechnique/INRIA-Parietal-Intership/Code/results_auto_prices.csv', index=False)

    # list_gs_scores = []
    # scores = pd.DataFrame(
    #     {"model": [], "best_cv_score": [], "best_param": []}
    # )

    # for k, v in gs_scores.items():
    #     print(f"{k} -- best params = {v.best_params_}")
    #     print(f"{k} -- cv scores = {v.best_score_}")
    #     list_gs_scores.append(v.best_score_)
    #     scores = scores.append(
    #         pd.DataFrame(
    #             {
    #                 "model": [k],
    #                 "best_cv_score": [v.best_score_],
    #                 "best_param": [v.best_params_],
    #             }
    #         )
    #     )

    # print("Housing Prices Dataset with 100 samples")
    # print("cv_score with tuning params = ", list_gs_scores)

    # print(scores)
    # scores.to_csv(
    #     "/home/mrivoire/Documents/M2DS_Polytechnique/INRIA-Parietal-Intership/Code/"
    #     + data_name
    #     + ".csv",
    #     index=False,
    # )

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
