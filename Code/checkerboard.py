import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib.pyplot import savefig
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from SPP import SPPRegressor, max_val


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
    rng = np.random.RandomState(42)
    x1 = rng.rand(n_samples) * dim1
    x2 = rng.rand(n_samples) * dim2

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

    y = [decision_function_uniformstrat(x1=u, x2=v, dim1=dim1, dim2=dim2, 
         n_bins=n_bins) for u, v in zip(x1, x2)]

    return np.c_[x1, x2], y


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
    print("Z = ", Z)
    print("Z length = ", np.count_nonzero(Z))

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

# save fig en pdf 
def main():
    dim1 = 10
    dim2 = 10
    n_bins = 5
    n_samples = 100

    X, y = checkerboard(dim1=dim1, dim2=dim2, n_samples=n_samples,
                        n_bins=n_bins)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    f = 10
    epsilon = 1e-14
    n_epochs = 1000
    screening = True
    store_history = True
    max_depth = 2
    n_lambda = 10
    tol = 1e-08
    lmbda = 0.00001
    encode = 'onehot'
    strategy = 'quantile'
    lambda_max_ratio = 0.5
    n_active_max = 100
    lambdas = [0.5]
    
    enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    X_binned_train = enc.fit_transform(X_train)

    spp_reg = SPPRegressor(n_lambda=n_lambda,
                           lambdas=lambdas,
                           max_depth=max_depth,
                           epsilon=epsilon, f=f, n_epochs=n_epochs, tol=tol,
                           lambda_max_ratio=lambda_max_ratio, 
                           n_active_max=n_active_max, screening=screening, 
                           store_history=store_history)

    solutions = spp_reg.fit(X_binned=X_binned_train, y=y_train).solutions_
    print('solutions = ', solutions)
    print('solutions length = ', len(solutions))
    slopes = solutions[0]['spp_lasso_slopes']
    print('slopes = ', slopes)
    print('slopes length = ', len(slopes))

    X_binned_test = enc.transform(X_test)
    y_hats = spp_reg.predict(X_binned=X_binned_test)
    print('y_hats = ', y_hats)
    print('y_hats length = ', len(y_hats))

    print(f'R2 score (train): {spp_reg.score(X_binned_train, y_train)}')
    print(f'R2 score (test): {spp_reg.score(X_binned_test, y_test)}')

    est = make_pipeline(enc, spp_reg)
    plot_est(est, X, y, X_train, y_train, X_test, y_test)

    # est = make_pipeline(
    #     enc,
    #     PolynomialFeatures(order=2, include_bias=False, interaction_only=True),
    #     Lasso(alpha=0.01)
    # )
    # plot_est(est, X, y, X_train, y_train, X_test, y_test)
    

if __name__ == "__main__":
    main()
