import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def autolabel(rects, ax, scale):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height * scale, 0) / scale),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def bar_plots(df):

    best_scores = list(- df['best_score'])
    models = list(df['model'])
    dataset_name = df['data'].unique()[0]
    n_samples = df['n_samples'].unique()[0]
    n_features = df['n_features'].unique()[0]

    plot_title = 'Dataset : ' + dataset_name + \
        ' (n_samples: ' + str(n_samples) + \
        ', n_features : ' + str(n_features) + ' )'

    x = np.arange(len(models))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    barlist = plt.bar(x, best_scores, width, label=models)
    color_range = sns.color_palette("tab10")
    barlist[0].set_color(color_range[0])
    barlist[1].set_color(color_range[1])
    barlist[2].set_color(color_range[2])
    barlist[3].set_color(color_range[3])
    barlist[4].set_color(color_range[4])
    barlist[5].set_color(color_range[5])

    ax.set_ylabel('MSE')
    ax.set_title(plot_title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)

    fig.tight_layout()

    plt.show()


def main():

    dataset_name = 'black_friday'

    df = pd.read_csv('/home/mrivoire/Documents/M2DS_Polytechnique/INRIA-Parietal-Intership/Code/' +
                     dataset_name + '_results.csv')

    print('df = ', df)

    bar_plots(df=df)


if __name__ == "__main__":
    main()
