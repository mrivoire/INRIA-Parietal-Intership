import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from mlxtend.plotting import checkerboard_plot


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

    bin_width_dim1 = dim1 / n_bins
    bin_width_dim2 = dim2 / n_bins

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

    x1 = np.random.rand(n_samples) * dim1
    x2 = np.random.rand(n_samples) * dim2

    data = {'x1': [],
            'x2': [],
            'y': []}

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
    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(data['x1'], data['x2'], data['y'])
    plt.show()
    

if __name__ == "__main__":
    main()