import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def autolabel(rects, ax, scale):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height * scale, 0) / scale),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def bar_plots(best_scores, labels, dataset_name, n_samples, n_features):

    plot_title = 'Dataset : ' + dataset_name + \
        ' (n_samples: ' + str(n_samples) + \
        ', n_features : ' + str(n_features) + ' )'

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    barlist = plt.bar(x, best_scores, width, label=labels)
    barlist[0].set_color('r')
    barlist[1].set_color('y')
    barlist[2].set_color('g')
    barlist[3].set_color('b')
    barlist[4].set_color('c')
    barlist[5].set_color('m')

    ax.set_ylabel('MSE')
    ax.set_title(plot_title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.legend()

    fig.tight_layout()

    plt.show()
