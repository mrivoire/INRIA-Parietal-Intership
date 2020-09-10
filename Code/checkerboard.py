import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from SPP import SPPRegressor


def compute_quantiles(x1, x2, n_bins):
    """
    Parameters
    ----------
    x1: list of floats
        list of coordinates of the observations in the first dimension

    x2: list of floats
        list of coordinates of the observations in the second dimension

    n_bins: int
        number of bins (and then of quantiles) that we want to create in each
        dimension

    Returns
    -------
    quantiles_dim1_idx: list of ints
        list containing the indices of the cut points delineating the quantiles 
        in the series of coordinates in the first dimension
    
    quantiles_dim2_idx: list of ints
        list containing the indices of the cut points delineating the quantiles
        in the series of coordinates in the second dimension

    quantiles_dim1_val: list of floats
        list containing the values of the cut points delineating the quantiles 
        in the series of coordinates in the first dimension

    quantiles_dim2_val: list of floats
        list containing the values of the cut points delineating the quantiles 
        int the series of coordinates in the second dimension

    """

    sorted_x1 = x1.sort()
    sorted_x2 = x2.sort()

    n_obs_quantile_dim1 = int(len(x1) / n_bins)
    n_obs_quantile_dim2 = int(len(x2) / n_bins)

    quantiles_dim1_idx = []
    quantiles_dim2_idx = []
    quantiles_dim1_val = []
    quantiles_dim2_val = []

    for q in range(n_bins):
        q_quantile_dim1_idx = q * n_obs_quantile_dim1 + 1
        q_quantile_dim2_idx = q * n_obs_quantile_dim2 + 1
        print('q_quantile_dim1_idx = ', q_quantile_dim1_idx)
        print('q_quantile_dim2_idx = ', q_quantile_dim2_idx)

        quantiles_dim1_idx.append(q_quantile_dim1_idx)
        quantiles_dim2_idx.append(q_quantile_dim2_idx)

        print('quantiles_dim1_idx = ', quantiles_dim1_idx)
        print('quantiles_dim2_idx = ', quantiles_dim2_idx)
        
        q_quantile_dim1_val = sorted_x1[q_quantile_dim1_idx]
        q_quantile_dim2_val = sorted_x2[q_quantile_dim2_idx]

        print('q_quantile_dim1_val = ', q_quantile_dim1_val)
        print('q_quantile_dim2_val = ', q_quantile_dim2_val)

        quantiles_dim1_val.append(q_quantile_dim1_val)
        quantiles_dim2_val.append(q_quantile_dim2_val)

    return (quantiles_dim1_val, quantiles_dim2_val, quantiles_dim1_idx, 
            quantiles_dim2_idx)


def decision_function_quantilestrat(coord1, coord2, x1, x2, dim1, dim2, n_bins):
    """
    Parameters
    ----------
    coord1: float
        coordinate of the given sample x

    coord2: float
        coordinate of the given sample x

    x1: list of floats 
        list of coordinates of the observations in the first dimension 

    x2: list of floats
        list of coordinates of the observations in the second dimension 

    dim1: int
        range of values for the first dimension

    dim2: int
        range of values for the second dimension

    n_bins: int
        number of bins (and then of quantiles) created in each dimension

    Returns
    -------
    y: int
        target variable with binary values :
        1 if the floor in the Euclidean division of the coordinates in the
        first and in the second dimension have the same parity
        (both even or both odds)
        0 otherwise
    """

    (quantiles_dim1_val, quantiles_dim2_val, quantiles_dim1_idx, 
     quantiles_dim2_idx) = \
        compute_quantiles(x1=x1, x2=x2, n_bins=n_bins)

    quantile_num_coord1 = 0
    quantile_num_coord2 = 0

    for num, quantile_val in enumerate(quantiles_dim1_val):
        if coord1 < quantile_val:
            pass
        else:
            quantile_num_coord1 = num - 1

    for num, quantile_val in enumerate(quantiles_dim2_val):
        if x2 < quantile_val:
            pass
        else:
            quantile_num_coord2 = num - 1

    if (quantile_num_coord1 % 2) == (quantile_num_coord2 % 2):
        y = 1
    else:
        y = -1

    return y


def decision_function_uniformstrat(x1, x2, dim1, dim2, n_bins):
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

    # if (int(x1 / bin_width_dim1) % 2) == 0 and ((int(x2 / bin_width_dim2) % 2) == 0):
    #     y = 1
    # elif ((int(x1 / bin_width_dim1) % 2) == 1) and ((int(x2 / bin_width_dim2) % 2) == 1):
    #     y = 1

    # The same parity of the two quotients can also be expressed by the 
    # equality of the rests in the Euclidean division of each quotient par 2
    if (int(x1 / bin_width_dim1) % 2) == (int(x2 / bin_width_dim2) % 2):
        y = 1
    else:
        y = -1

    return y


def checkerboard(dim1, dim2, n_samples, n_bins, binning_strategy):
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
    rng = np.random.RandomState(42)
    x1 = rng.rand(n_samples) * dim1
    x2 = rng.rand(n_samples) * dim2

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

    if binning_strategy == 'uniform':
        y = [decision_function_uniformstrat(x1=u, x2=v, dim1=dim1, dim2=dim2, 
             n_bins=n_bins) for u, v in zip(x1, x2)]
    elif binning_strategy == 'quantile':
        y = [decision_function_quantilestrat(coord1=u, coord2=v, x1=x1, x2=x2, 
             dim1=dim1, dim2=dim2, n_bins=n_bins) for u, v in zip(x1, x2)] 

    # for idx in range(n_samples):
    #     x1coor = x1[idx]
    #     x2coor = x2[idx]

    #     y = decision_function_uniformstrat(x1=x1coor, x2=x2coor, dim1=dim1,
    #                           dim2=dim2, n_bins=n_bins)

    #     data['x1'].append(x1coor)
    #     data['x2'].append(x2coor)
    #     data['y'].append(y)

    data['x1'] = x1
    data['x2'] = x2
    data['y'] = y

    data = pd.DataFrame(data)
    X = data[['x1', 'x2']].values
    y = data['y'].values.astype(float)

    return X, y


def plot_est(est, X, y, X_train, y_train, X_test, y_test):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    h = .1  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    _, ax = plt.subplots(1, 1)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    est.fit(X_train, y_train)
    Z = est.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    plt.show()


def main():
    dim1 = 10
    dim2 = 10
    n_bins = 5
    n_samples = 1000
    binning_strategy = 'uniform'

    rng = np.random.RandomState(42)
    x1 = rng.rand(n_samples) * dim1
    x2 = rng.rand(n_samples) * dim2

    x1 = list(x1)
    x2 = list(x2)

    (quantiles_dim1_val, quantiles_dim2_val, quantiles_dim1_idx, 
     quantiles_dim2_idx) = compute_quantiles(x1=x1, x2=x2, n_bins=n_bins)

    # X, y = checkerboard(dim1=dim1, dim2=dim2, n_samples=n_samples, n_bins=n_bins)

    # X_train, X_test, y_train, y_test = \
    #     train_test_split(X, y, test_size=.4, random_state=42)

    # f = 10
    # epsilon = 1e-14
    # n_epochs = 1000
    # screening = True
    # store_history = True
    # max_depth = 2
    # n_val_gs = 10
    # tol = 1e-08
    # lmbda = 0.00001
    # encode = 'onehot'
    # strategy = 'quantile'

    # spp_reg = SPPRegressor(lmbda=lmbda, n_val_gs=n_val_gs,
    #                        max_depth=max_depth,
    #                        epsilon=epsilon, f=f, n_epochs=n_epochs, tol=tol,
    #                        screening=screening, store_history=store_history)

    # enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)

    # X_binned = enc.fit_transform(X_train).tocsc()

    # spp_reg.fit(X_binned=X_binned, y=y_train)

    # print(f'R2 score: {spp_reg.score(X_binned, y_train)}')

    # est = make_pipeline(enc, spp_reg)
    # plot_est(est, X, y, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
