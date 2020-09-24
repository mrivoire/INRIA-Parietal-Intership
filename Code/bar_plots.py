import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def bar_plots(scores_spp, scores_lasso, scores_lassoCV, scores_rf, scores_xgb):

    labels = ['SPPRegressor', 'Lasso', 'LassoCV', 'Random_Forests', 'XGBoost']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 2 * (width/5), scores_spp, width, label='SPPRegressor')
    rects2 = ax.bar(x - (width/5), scores_lasso, width, label='Lasso')
    rects3 = ax.bar(x, scores_lassoCV, width, label='LassoCV')
    rects4 = ax.bar(x + (width/5), scores_rf, width, label='Random_Forests')
    rects5 = ax.bar(x + 2 * (width/5), scores_xgb, width, label='XGBoost')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Best CV Scores')
    ax.set_title('Performances of the different solvers')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    fig.tight_layout()

    plt.show()
