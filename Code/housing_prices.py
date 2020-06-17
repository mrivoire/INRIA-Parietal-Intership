import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score


#######################################
#              Read CSV
#######################################


def read_csv(filePath):
    data = pd.read_csv(filePath + ".csv")
    return data


########################################
#               All Data
########################################


def concat(X_train, X_test):

    all_data = pd.concat((X_train.loc[:, 'MSSubClass':'SaleCondition'],
                          X_test.loc[:, 'MSSubClass':'SaleCondition']))

    return all_data


#################################################
#           Log transform of the target
#################################################


def log_transform(feature):
    log_feat = np.log1p(feature)

    return log_feat


#################################################
#           Numeric Features
#################################################


def numeric_features(dataset):
    numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index

    return numeric_feats


###################################################
#           Skewed Features
###################################################


def skewness(dataset):
    numeric_feats = numeric_features(dataset)
    skewed_feats = dataset[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    return skewed_feats


##########################################################
#     One-hot encoding of the categorical features
##########################################################


def onehot_encoding(dataset):
    onehot_data = pd.get_dummies(dataset)

    return onehot_data


###########################################################
#                       Nan Values 
###########################################################


def proportion_NaN(dataset):
    prop = dataset.isna().mean()
    
    return prop


def fill_NaN(dataset):
    filled_data = dataset.fillna(dataset.mean())

    return filled_data


def main():

    data_dir = "/home/mrivoire/Documents/M2DS_Polytechnique/Stage_INRIA/Code/Datasets"
    fname_train = data_dir + "/housing_prices_train"
    fname_test = data_dir + "/housing_prices_test"
    train_set = read_csv(fname_train)
    head_train = train_set.head()
    test_set = read_csv(fname_test)
    head_test = test_set.head()
    print("Housing Prices Training Set Header : ", head_train)
    print("Housing Prices Testing Set Header : ", head_test)

    all_data = concat(train_set, test_set)
    head_all_data = all_data.head()
    print("All data : ", head_all_data)

    log_sale_price = log_transform(train_set['SalePrice'])
    print("log target : ", log_sale_price)

    numeric_feats = numeric_features(all_data)
    print("numeric features : ", numeric_feats)

    skewed_feats = skewness(train_set)
    print("skewed features : ", skewed_feats)

    all_data = onehot_encoding(all_data)
    print("onehot data : ", all_data)

    prop_NaN = proportion_NaN(all_data)
    print("proportion NaN values :", prop_NaN)

    all_data = fill_NaN(all_data)
    print(all_data.head())

    # Plots
    # matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
    # prices = pd.DataFrame({"price":train_set["SalePrice"],
    #                        "log(price + 1)":np.log1p(train_set["SalePrice"])})
    # prices.hist()


if __name__ == "__main__":
    main()
