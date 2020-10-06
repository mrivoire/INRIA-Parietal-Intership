import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import constants_plots as cst
import sys


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

    best_scores = list(- df['best_test_score'])
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

    barlist_test = plt.bar(x, best_scores, width, label=models)
    color_range = sns.color_palette("tab10")
    barlist_test[0].set_color(color_range[0])
    barlist_test[1].set_color(color_range[1])
    barlist_test[2].set_color(color_range[2])
    barlist_test[3].set_color(color_range[3])
    barlist_test[4].set_color(color_range[4])
    barlist_test[5].set_color(color_range[5])
    barlist_test[6].set_color(color_range[6])
    barlist_test[7].set_color(color_range[7])
    barlist_test[8].set_color(color_range[8])

    ax.set_ylabel('MSE')
    ax.set_title(plot_title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    plt.show()


def bar_plots_test_train(df):

    best_test_scores = list(- df['best_test_score'])
    best_train_scores = list(- df['best_train_score'])
    models = list(df['model'])
    dataset_name = df['data'].unique()[0]
    n_samples = df['n_samples'].unique()[0]
    n_features = df['n_features'].unique()[0]

    plot_title = 'Dataset : ' + dataset_name + \
        ' (n_samples: ' + int(n_samples) + \
        ', n_features : ' + int(n_features) + ' )'

    x = np.arange(len(models))  # the label locations
    width = 0.05  # the width of the bars

    fig, ax = plt.subplots()

    barlist_test = ax.bar(x - width/2, best_test_scores, label=models)
    barlist_train = ax.bar(x + width/2, best_train_scores, label=models)

    ax.set_ylabel('MSE')
    ax.set_title(plot_title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend([barlist_train, barlist_test], ['train', 'test'])

    # autolabel(barlist_test, ax, 1000)
    # autolabel(barlist_train, ax, 1000)

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    plt.show()


def main():

    dataset_name = sys.argv[1]
    if dataset_name not in cst.dataset_names_dict.keys():
        raise ValueError(dataset_name+' is not in : ' +
                         str(list(cst.dataset_names_dict.keys())))
    if len(sys.argv) > 2:
        n_samples = sys.argv[2]
    else:
        n_samples = cst.n_samples_dict[dataset_name]

    dataset_name = 'black_friday'
    n_samples = 10000

    plot_title = 'Dataset : ' + cst.dataset_names_dict[dataset_name] + \
        '\n (n_samples: ' + str(n_samples) + \
        ', n_features : ' + str(cst.n_features_dict[dataset_name]) + ' )'

    df = pd.read_csv('/home/mrivoire/Documents/M2DS_Polytechnique/Final_Results/black_friday/' + dataset_name + '_' + str(n_samples) +
                     '_results.csv')
    # df = pd.read_csv('./' + dataset_name + '_' +
    #                  str(n_samples) + '_results.csv')

    sns.set_theme(style="ticks", font_scale=1.5)
    df = df[
        ['best_test_score', 'best_train_score', 'model']
    ].set_index('model').unstack().reset_index()
    df = df.loc[df['model'] != 'lasso']
    df['model_name'] = df['model'].map(cst.models_dict)
    df.columns = ['data', 'model', 'MSE', 'Model']
    df.MSE *= -1
    df.data = df.data.map(
        cst.legend_dict)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(True, which="both")
    sns.barplot(x="Model", y="MSE", hue="data",
                data=df, palette="Paired")
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(loc='upper right')
    ax.set_title(plot_title, fontsize=16)
    plt.tight_layout()
    plt.show()

    print('df = ', df)

    # bar_plots_test_train(df=df)


if __name__ == "__main__":
    main()
