import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import constants_plots as cst
import sys


def n_samples_impact_plot(df, plot_title):

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(True, which="both")
    sns.lineplot(x="log_n_samples", y="MSE", hue="Model",
                 data=df, palette="Paired")
    # ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(bbox_to_anchor=(1.25, 1.25), loc='upper right', fontsize=10)
    ax.set_title(plot_title, fontsize=16)
    plt.tight_layout()
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

    plot_title = 'Impact of the number of samples,\n on the predictive performances for the Black Friday dataset'

    n_samples_impact_plot(df=df, plot_title=plot_title)


if __name__ == "__main__":
    main()
