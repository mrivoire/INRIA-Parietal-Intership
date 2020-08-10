import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipdb
import time
import matplotlib
import matplotlib.pyplot as plt


from numpy.random import randn
from scipy.linalg import toeplitz

from numba import njit, objmode
from numba import jit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso as sklearn_Lasso
from sklearn.utils import check_random_state
from scipy.sparse import csc_matrix
from scipy.sparse import issparse

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from numba.typed import List
from numba import typeof


def simu(beta, n_samples=1000, corr=0.5, for_logreg=False, random_state=None):
    n_features = len(beta)
    cov = toeplitz(corr ** np.arange(0, n_features))

    rng = check_random_state(random_state)

    # Features Matrix
    X = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X = np.asfortranarray(X)

    # Target labels vector with noise
    y = np.dot(X, beta) + rng.randn(n_samples)

    if for_logreg:
        y = sign(y)

    return X, y


@njit
def sign(x):
    """
    Parameters
    ----------
    x: float

    Returns
    -------
    s: sign int
      (-1) if x < 0,, (+1) if x > 0, 0 if x = 0
    """

    if x > 0:
        s = 1
    elif x < 0:
        s = -1
    else:
        s = 0

    return s


@njit
def soft_thresholding(u, x):
    """
    Parameters
    ----------
    u: float
      threshold

    x: float

    Returns
    -------
    ST: float
        0 between -u and +u or slope of the straight line x - u otherwise

    """

    ST = sign(x) * max(abs(x) - u, 0)

    return ST


@njit
def sign(x):
    """
    Parameters
    ----------
    x: float

    Returns
    -------
    s: sign int
      (-1) if x < 0,, (+1) if x > 0, 0 if x = 0
    """

    if x > 0:
        s = 1
    elif x < 0:
        s = -1
    else:
        s = 0

    return s


@njit
def soft_thresholding(u, x):
    """
    Parameters
    ----------
    u: float
      threshold

    x: float

    Returns
    -------
    ST: float
        0 between -u and +u or slope of the straight line x - u otherwise

    """

    ST = sign(x) * max(abs(x) - u, 0)

    return ST


@njit
def sparse_cd(
    X_data,
    X_indices,
    X_indptr,
    y,
    lmbda,
    epsilon,
    f,
    n_epochs,
    screening,
    store_history,
):
    """Solver : sparse cyclic coordinate descent

    Parameters
    ----------

    X: numpy.ndarray, shape (n_samples, n_features)
        features matrix

    X_data: csr format, data array of the features matrix
        data is an array containing all the non-zero elements of the sparse
        matrix

    X_indices: csr format, index array of the features matrix
        indices is an array mapping each element in data to its rows in the
        sparse matrix

    X_indptr: csr format, index array pointer of the features matrix
        indptr is an array mapping the elements of data and indices to the
        columns of the sparse matrix

    y: numpy.array, shape (n_samples, )
        target labels vector

    lmbda: float
        regularization parameter

    epsilon: float
        stopping criterion

    f: int
    frequency

    n_epochs: int,

    screening: bool, default = True
        defines whether or not one adds screening to the solver

    store_history: bool, default = True
        defines whether or not one stores the values of the parameters
        while the solver is running

    Returns
    -------
    beta: numpy.array, shape(n_features,)
        primal parameters vector

    theta: numpy.array, shape(n_samples, )
        dual parameters vector

    P_lmbda: float
        primal value

    D_lmbda: float
        dual value

    primal_hist: numpy.array, shape(n_epochs / f, )
        store the primal values during the whole solving process

    dual_hist: numpy.array, shape(n_epochs / f, )
        store the dual values during the whole solving process

    gap_hist: numpy.array, shape(n_epochs / f, )
        store the duality gap values during the whole solving process

    r_list: numpy.array, shape(n_epochs / f, )
        store the values of the radius of the safe sphere during
        the screening process

    n_active_features: numpy.array, shape(n_epochs / f, )
        store the number of active features in the active set
        during the screening process

    """

    n_features = len(X_indptr) - 1
    n_samples = len(X_indices) + 1
    beta = np.zeros(n_features)
    theta = np.zeros(n_samples)

    y_norm2 = np.linalg.norm(y, ord=2) ** 2
    residuals = np.copy(y)
    n_active_features = []
    r_list = []
    primal_hist = []
    dual_hist = []
    gap_hist = []
    theta_hist = []
    P_lmbda = 0
    D_lmbda = 0
    G_lmbda = 0
    # y_norm2 = 0

    # for i in range(n_samples):
    #     y_norm2 += y[i]**2

    safeset_membership = np.ones(n_features)

    L = np.zeros(n_features)
    # L = List([int(x) for x in range(0)])
    # L = [0] * (n_features)
    for j in range(n_features):
        start, end = X_indptr[j: j + 2]
        start = np.int64(start)
        end = np.int64(end)

        for ind in range(start, end):
            L[j] += X_data[ind] ** 2

    for k in range(n_epochs):
        for j in range(n_features):
            if L[j] == 0.0:
                continue

            if safeset_membership[j] == 0:
                continue

            old_beta_j = beta[j]

            # Matrix product between the features matrix X and the residuals
            start, end = X_indptr[j: j + 2]
            grad = 0.0
            for ind in range(start, end):
                grad += X_data[ind] * residuals[X_indices[ind]]

            # Update of the parameters
            step = 1 / L[j]
            beta[j] += step * grad

            # Apply proximal operator
            beta[j] = soft_thresholding(step * lmbda, beta[j])
            diff = old_beta_j - beta[j]

            if diff != 0:
                for ind in range(start, end):
                    residuals[X_indices[ind]] += diff * X_data[ind]

        if k % f == 0:
            # Computation of theta
            XTR_absmax = 0
            # Matrix product between the features matrix X and the residuals
            for j in range(n_features):
                if safeset_membership[j] == 1:
                    start, end = X_indptr[j: j + 2]
                    dot = 0.0
                    for ind in range(start, end):
                        dot += X_data[ind] * residuals[X_indices[ind]]
                    XTR_absmax = max(abs(dot), XTR_absmax)

            theta = residuals / max(XTR_absmax, lmbda)

            # Computation of the primal problem
            P_lmbda = 0.5 * residuals.dot(residuals)
            P_lmbda += lmbda * np.linalg.norm(beta, 1)

            # Computation of the dual problem
            D_lmbda = 0.5 * y_norm2
            D_lmbda -= ((lmbda ** 2) / 2) * np.linalg.norm(
                theta - y / lmbda, ord=2
            ) ** 2

            # Computation of the dual gap
            G_lmbda = P_lmbda - D_lmbda

            # Objective function related to the primal
            if store_history:
                theta_hist.append(theta)
                primal_hist.append(P_lmbda)
                dual_hist.append(D_lmbda)
                gap_hist.append(G_lmbda)

            if screening:
                # Computation of the radius of the gap safe sphere
                r = np.sqrt(2 * np.abs(G_lmbda)) / lmbda
                if store_history:
                    r_list.append(r)

                # Computation of the active set
                for j in range(n_features):
                    if safeset_membership[j] == 1:
                        start, end = X_indptr[j : j + 2]
                        dot = 0.0
                        norm = 0.0

                        for ind in range(start, end):
                            dot += X_data[ind] * theta[X_indices[ind]]
                            norm += X_data[ind] ** 2

                        norm = np.sqrt(norm)
                        dot = np.abs(dot)

                        mu = np.abs(dot) + r * norm

                    if mu < 1:
                        safeset_membership[j] = 0

                if store_history:
                    n_active_features.append(np.sum(safeset_membership))

                if np.abs(G_lmbda) <= epsilon:
                    break

    return (
        beta,
        primal_hist,
        dual_hist,
        gap_hist,
        r_list,
        n_active_features,
        theta,
        P_lmbda,
        D_lmbda,
        G_lmbda,
        safeset_membership,
    )


def main():
    rng = check_random_state(0)
    n_samples, n_features = 100, 100
    beta = rng.randn(n_features)
    lmbda = 1.
    f = 10.
    epsilon = 1e-15
    n_epochs = 100000
    screening = True
    store_history = True
    encode = "onehot"
    strategy = "quantile"

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False, 
                random_state=rng)

    enc = KBinsDiscretizer(n_bins=2, encode=encode, strategy=strategy)
    X_binned = enc.fit_transform(X)
    X_binned = X_binned.tocsc()
    X_binned_data = X_binned.data
    X_binned_indices = X_binned.indices
    X_binned_indptr = X_binned.indptr
    X_binned_data = List([int(x) for x in range(0)])
    # X_binned_indices = List([int(x) for x in range(0)])
    # X_binned_indptr = List([int(x) for x in range(0)])

    (beta, 
     primal_hist, 
     dual_hist, gap_hist, 
     r_list, n_active_features, 
     theta, P_lmbda, 
     D_lmbda, 
     G_lmbda, 
     safeset_membership, 
     ) = sparse_cd(X_data=X_binned_data, X_indices=X_binned_indices, 
                   X_indptr=X_binned_indptr, y=y, 
                   lmbda=lmbda, epsilon=epsilon, f=f, n_epochs=n_epochs, 
                   screening=screening, store_history=store_history)


if __name__ == "__main__":
    main()