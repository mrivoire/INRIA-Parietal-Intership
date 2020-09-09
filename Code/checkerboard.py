import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from mlxtend.plotting import checkerboard_plot
from SPP import SPPRegressor
from scipy.sparse import csc_matrix, hstack
from sklearn.preprocessing import KBinsDiscretizer


def decision_function(x1, x2, dim1, dim2, n_bins):
    """
    Parameters
    ----------
    x1: float
        coordinate of the first dimension of the sample x

    x2: float
        cooridnate of the second dimension of the sample x

    dim1: int
        range of values for the first dimension

    dim2: int
        range of values for the second dimension

    n_bins: int
        number of bins created in each dimension

    Returns
    -------
    y: int
        target variable with binary values : 
        1 if the floor in the Euclidean division of the coordinates in the 
        first and in the second dimension have the same parity 
        (both even or both odds)
        0 otherwise
    """

    # We want to create a number n_bins of bins in each dimension
    # For that purpose we divide the range of values in each dimension by the 
    # number of bins that we want to create in each dimension
    # Thus we obtain the width of each bin in each dimension

    bin_width_dim1 = dim1 / n_bins
    bin_width_dim2 = dim2 / n_bins

    # Given a sample x for which the coordinate in the first dimension is 
    # denoted as x1 and the coordinate in the second dimension is denoted as x2
    # We want to know to locate the sample x in the checkerboard i.e. we want 
    # to know to which bins both in the first dimension and in the second 
    # dimension the sample x belongs
    # For that purpose, we have to divide each coordinate in each dimension 
    # by the bin width in the given dimension, this allows to look for the 
    # maximum number of bin width that there exists in each coordinate i.e. the 
    # maximum multiple of bin width that there exists in each coordinate in each 
    # dimension, then in each dimension, the sample x belongs to the bin whose 
    # the number corresponds to the maximum multiple of the bin width that 
    # there exists in the coordinate of the given dimension (if we count the 
    # number of bins starting from 0) and to the maximum multiple of the bin 
    # width that there exists in the coordinate of the given dimension plus 1 
    # (if we count the number of bins starting from 1)

    # Once we have locate the sample x in the checkboard we have to classify it 
    # between two classes 0 (white) or 1 (black), this classification is 
    # performed thanks to the decision function corresponding to the 
    # checkerboard, this decision function proceeds as follows
    # if the numbers of the bins in both dimensions to which the given sample 
    # belongs have the same parity, then the given sample belongs to the 
    # class 1 (black) otherwise it belongs to the class 0

    if (int(x1 / bin_width_dim1) % 2) == 0 and ((int(x2 / bin_width_dim2) % 2) == 0):
        y = 1
    elif ((int(x1 / bin_width_dim1) % 2) == 1) and ((int(x2 / bin_width_dim2) % 2) == 1):
        y = 1
    else:
        y = 0

    return y 


def checkerboard(dim1, dim2, n_samples, n_bins):
    """
    Parameters
    ----------
    dim1: int
        range of values for the first dimension

    dim2: int 
        range of values for the second dimension

    n_samples: int
        number of samples to classify

    n_bins: int
        number of bins created in each dimension

    Returns
    -------
    data: dictionary under the form of pandas DataFrame
        this dictionary contains 3 lists which are the following :
        'x1': list containing the coordinates of the samples in the 
        first dimension
        'x2': list containing the coordinates of the samples in the 
        second dimension
        'y': list containing the binary values of the target variable 
    """

    # We randomly generate a number n_samples of samples in each dimension
    x1 = np.random.rand(n_samples) * dim1
    x2 = np.random.rand(n_samples) * dim2

    data = {'x1': [],
            'x2': [],
            'y': []}

    # From the list of coordinates of the 2 dimensions we create samples by 
    # attributing to each sample the coordinates of same index in the lists
    # Then for each sample of coordinates (x1, x2), we classify the sample 
    # between the two potential classes thanks to the decision function of 
    # the checkerboard
    # Therefore, the samples have been randomly generated but they all are 
    # classified in a non-random way thanks to the decision function of the 
    # checkerboard
    # If the number of generated samples is small then the classification of 
    # the samples will respect the decision boundaries of the checkerboard but 
    # we will not be able to detect the structure of a checkerboard
    # nevertheless, when the number of generated samples considerably 
    # increases, we can easily detect after classification by the decision 
    # function of all the samples, the structure of a checkerboard

    # Finally, we store in a dataframe the coordinates of the samples and the 
    # predicted labels (0 or 1) cooresponding to the target values
    # Therefore, the generated samples with their corresopnding target labels 
    # form the training set on which we can train our spp solver which is 
    # intended to classify the samples of the training set in the same way 
    # that the decision function proceeds

    for idx in range(n_samples):
        x1coor = x1[idx]
        x2coor = x2[idx]

        y = decision_function(x1=x1coor, x2=x2coor, dim1=dim1, 
                              dim2=dim2, n_bins=n_bins)

        data['x1'].append(x1coor)
        data['x2'].append(x2coor)
        data['y'].append(y) 

    data = pd.DataFrame(data)

    return data


def main():
    dim1 = 10
    dim2 = 10
    n_bins = 5
    n_samples = 100000

    data = checkerboard(dim1=dim1, dim2=dim2, n_samples=n_samples, n_bins=n_bins)
    # fig = plt.figure(figsize=(10, 10))
    # sns.scatterplot(data['x1'], data['x2'], data['y'])
    # plt.show()

    f = 10
    epsilon = 1e-14
    n_epochs = 100000
    screening = True
    store_history = True
    max_depth = 2
    n_val_gs = 10
    tol = 1e-08
    best_lmbda_idx = 0
    encode = 'onehot'
    strategy = 'quantile'

    spp_reg = SPPRegressor(best_lmbda_idx=best_lmbda_idx, n_val_gs=n_val_gs,
                           max_depth=max_depth,
                           epsilon=epsilon, f=f, n_epochs=n_epochs, tol=tol,
                           screening=screening, store_history=store_history)


    y_train = data['y']
    X = data.drop(y_train).to_numpy()
    enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)

    y_train = y_train.to_numpy()
    print('y_train = ', y_train)
    print('type y_train = ', type(y_train))
    X_binned = enc.fit_transform(X)
    X_binned = X_binned.tocsc()

    spp_reg.fit(X_binned=X_binned, y=y_train)
    

if __name__ == "__main__":
    main()