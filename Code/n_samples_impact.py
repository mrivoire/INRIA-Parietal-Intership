import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import constants_plots as cst


params = {'axes.labelsize': 12,
          'font.size': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12}
plt.rcParams.update(params)


def n_samples_impact_plot(df, plot_title):

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.grid(True, which="both")
    sns.lineplot(x="log_n_samples", y="MSE", hue="Model",
                 data=df, palette="Paired")
    # ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.tick_params(axis='both', which='major')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(plot_title)
    plt.tight_layout()
    plt.xlabel(r'$\log(n_{samples})$')
    plt.show()


def main():
    # df_1000 = pd.read_csv(
    #     '/home/mrivoire/Documents/M2DS_Polytechnique/Final_Results/black_friday/black_friday_1000_results.csv')
    # df_10000 = pd.read_csv(
    #     '/home/mrivoire/Documents/M2DS_Polytechnique/Final_Results/black_friday/black_friday_10000_results.csv')
    # df_100000 = pd.read_csv(
    #     '/home/mrivoire/Documents/M2DS_Polytechnique/Final_Results/black_friday/black_friday_100000_results.csv')
    # df_166821 = pd.read_csv(
    #     '/home/mrivoire/Documents/M2DS_Polytechnique/Final_Results/black_friday/black_friday_166821_results.csv')

    df_1000 = pd.read_csv(
        './black_friday_1000_results.csv')
    df_10000 = pd.read_csv(
        './black_friday_10000_results.csv')
    df_100000 = pd.read_csv(
        './black_friday_100000_results.csv')
    df_166821 = pd.read_csv(
        './black_friday_166821_results.csv')

    df = pd.concat([df_1000, df_10000, df_100000, df_166821])

    df = df.loc[df['model'] != 'lasso']
    df['model_name'] = df['model'].map(cst.models_dict)
    df = df[['best_test_score', 'model_name', 'n_samples']]
    df.columns = ['MSE', 'Model', 'n_samples']
    df['log_n_samples'] = np.log10(df['n_samples'])
    df.MSE *= -1

    plot_title = r'Impact of $n_{samples}$ for the Black Friday dataset'

    n_samples_impact_plot(df=df, plot_title=plot_title)


if __name__ == "__main__":
    main()
