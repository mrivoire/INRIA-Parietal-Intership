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

######################################################################
#     Iterative Solver With Gap Safe Rules
######################################################################

"""
One chooses Coordinate Descent as iterative solver.
The idea of coordinate descent is to decompose a large optimisation
problem into a sequence of one-dimensional optimisation problems.
Coordinate descent methods have become unavoidable
in machine learning because they are very efficient for key problems,
namely Lasso, Logistic Regression and Support Vector Machines.
Moreover, the decomposition into small subproblems means that
only a small part of the data is processed at each iteration and
this makes coordinate descent easily scalable to high dimensions.
The idea of coordinate gradient descent is to perform
one iteration of gradient in the 1-dimensional problem
instead of solving it completely. In general it reduces drastically
the cost of each iteration while keeping the same convergence behaviour.
"""

#############################################################################
#                           Data Simulation
#############################################################################


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


############################################################################
#                   Cyclic CD For Dense Features Matrix
############################################################################


@njit
def cyclic_coordinate_descent(
    X, y, lmbda, epsilon, f, n_epochs, screening, store_history
):
    """Solver : dense cyclic coordinate descent

    Parameters
    ----------

    X: numpy.ndarray, shape (n_samples, n_features)
        features matrix

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

    # Initialisation of the parameters

    n_samples, n_features = X.shape

    beta = np.zeros(n_features)
    theta = np.zeros(n_samples)

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

    y_norm2 = np.linalg.norm(y, ord=2) ** 2
    # for i in range(n_samples):
    #     y_norm2 += y[i]**2

    residuals = y.copy()

    # Computation of the lipschitz constants vector
    L = (X ** 2).sum(axis=0)

    safeset_membership = np.ones(n_features)

    # Iterations of the algorithm
    for k in range(n_epochs):
        for j in range(n_features):
            if safeset_membership[j] == 0:
                continue

            # One cyclicly updates the i^{th} coordinate corresponding to
            # the rest in the Euclidean division by the number of features
            # This allows to always selecting an index between 1 and
            # n_features

            old_beta_j = beta[j]
            step = 1 / L[j]
            grad = np.dot(X[:, j], residuals)

            # Update of the parameters
            beta[j] += step * grad

            # Apply proximal operator
            beta[j] = soft_thresholding(step * lmbda, beta[j])

            # Update of the residuals
            delta_beta_j = old_beta_j - beta[j]
            if delta_beta_j != 0.0:
                for i in range(n_samples):
                    residuals[i] += delta_beta_j * X[i, j]
                # residuals += delta_beta_j * X[:, j]

        if k % f == 0:
            # Computation of theta
            XTR_absmax = 0
            for j in range(n_features):
                if safeset_membership[j]:
                    XTR_absmax = max(abs(X[:, j].dot(residuals)), XTR_absmax)

            theta = residuals / max(XTR_absmax, lmbda)

            # Computation of the primal problem
            # beta_norm1 = 0
            # for i in range(n_features):
            #     beta_norm1 += abs(beta[i])

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
                        mu = np.abs(X[:, j].T.dot(theta)) + r * np.sqrt(L[j])

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


#########################################################################
#                 Cyclic CD For Sparse Features Matrix
#########################################################################


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
    n_samples = max(X_indices) + 1
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

    for j in range(n_features):
        start, end = X_indptr[j : j + 2]
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
            start, end = X_indptr[j : j + 2]
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
                    start, end = X_indptr[j : j + 2]
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


# https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr


###########################################################################
#                           Class Dense Lasso
###########################################################################


class Lasso(BaseEstimator, RegressorMixin):
    def __init__(self, lmbda, epsilon, f, n_epochs, screening, store_history):

        self.lmbda = lmbda
        self.epsilon = epsilon
        self.f = f
        self.n_epochs = n_epochs
        self.screening = screening
        self.store_history = store_history

        assert epsilon > 0

    def fit(self, X, y):
        """Fit the data (X,y) based on the solver of the Lasso class

        Parameters
        ----------

        X: numpy.ndarray, shape = (n_samples, n_features)
            features matrix

        y: numpy.array, shape = (n_samples, )
            target vector

        Returns
        -------
        self: Lasso object
        """

        if issparse(X):
            X = X.tocsc()
            X_data = X.data
            X_indices = X.indices
            X_indptr = X.indptr

            (
                beta_hat_cyclic_cd_true,
                primal_hist,
                dual_hist,
                gap_hist,
                r_list,
                n_active_features_true,
                theta_hat_cyclic_cd,
                P_lmbda,
                D_lmbda,
                G_lmbda,
                safe_set,
            ) = sparse_cd(
                X_data,
                X_indices,
                X_indptr,
                y,
                self.lmbda,
                self.epsilon,
                self.f,
                self.n_epochs,
                self.screening,
                self.store_history,
            )
        else:
            (
                beta_hat_cyclic_cd_true,
                primal_hist,
                dual_hist,
                gap_hist,
                r_list,
                n_active_features_true,
                theta_hat_cyclic_cd,
                P_lmbda,
                D_lmbda,
                G_lmbda,
                safe_set,
            ) = cyclic_coordinate_descent(
                X,
                y,
                self.lmbda,
                self.epsilon,
                self.f,
                self.n_epochs,
                self.screening,
                self.store_history,
            )

        self.slopes = beta_hat_cyclic_cd_true
        self.G_lmbda = G_lmbda
        self.P_lmbda = P_lmbda
        self.r_list = r_list
        self.safe_set = safe_set

        return self

    def predict(self, X):
        """Predict the target from the observations matrix

        Parameters
        ----------
        X: numpy.ndarray, shape = (n_samples, n_features)
            features matrix

        Returns
        -------
        y_hat: numpy.array, shape = (n_samples, )
            predicted target vector
        """

        y_hat = X.dot(self.slopes)

        return y_hat

    def score(self, X, y):
        """Compute the cross-validation score to assess the performance of the
           model (use negative mean absolute error)

        Parameters
        ----------
        X: numpy.ndarray, shape = (n_samples, n_features)
            features matrix

        y: numpy.array, shape = (n_samples, )
            target vector

        Returns
        -------
        score: float
            negative mean absolute error (MAE)
            negative to keep the semantic that higher is better
        """

        u = ((y - self.predict(X)) ** 2).sum()
        v = ((y - np.mean(y)) ** 2).sum()
        score = 1 - u / v

        return score


##########################################################
#                    Sign Function
##########################################################


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


######################################
#    Soft-Thresholding Function
######################################


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


#######################################
#              Read CSV
#######################################


def read_csv(filePath):
    data = pd.read_csv(filePath + ".csv")
    return data


def main():
    # Data Simulation
    rng = check_random_state(0)
    n_samples, n_features = 100, 100
    beta = rng.randn(n_features)
    lmbda = 1.0
    f = 10
    epsilon = 1e-15
    n_epochs = 100000
    screening = True
    store_history = True
    encode = "onehot"
    strategy = "quantile"

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False, 
                random_state=rng)

    # X = csc_matrix(X)
    # X_data = X.data
    # X_indices = X.indices
    # X_indptr = X.indptr
    enc = KBinsDiscretizer(n_bins=2, encode=encode, strategy=strategy)
    X_binned = enc.fit_transform(X)
    X_binned = X_binned.tocsc()
    X_binned_data = X_binned.data
    X_binned_indices = X_binned.indices
    X_binned_indptr = X_binned.indptr

    
    (beta_hat_cyclic_cd_true_sparse,
        primal_hist_sparse,
        dual_hist_sparse,
        gap_hist_sparse,
        r_list_sparse,
        n_active_features_true_sparse,
        theta_hat_cyclic_cd_sparse,
        P_lmbda_sparse,
        D_lmbda_sparse,
        G_lmbda_sparse,
        safe_set_sparse) = sparse_cd(X_data=X_binned_data,
                                     X_indices=X_binned_indices,
                                     X_indptr=X_binned_indptr, y=y,
                                     lmbda=lmbda,
                                     epsilon=epsilon, f=f,
                                     n_epochs=n_epochs, screening=screening,
                                     store_history=store_history)

    print("safe set sparse : ", safe_set_sparse)

    sparse_lasso_sklearn = sklearn_Lasso(alpha=(lmbda / X_binned.shape[0]),
                                         fit_intercept=False,
                                         normalize=False,
                                         max_iter=n_epochs,
                                         tol=1e-14).fit(X_binned, y)

    slopes = sparse_lasso_sklearn.coef_
    intercept = sparse_lasso_sklearn.intercept_

    primal_function = ((1 / 2)
                       * np.linalg.norm(y - X_binned.dot(slopes.T), 2)**2
                       + lmbda * np.linalg.norm(slopes, 1))

    print("primal sklearn : ", primal_function)
    X_binned = X_binned.toarray()

    (beta_hat_cyclic_cd_true,
        primal_hist,
        dual_hist,
        gap_hist,
        r_list,
        n_active_features_true,
        theta_hat_cyclic_cd,
        P_lmbda,
        D_lmbda,
        G_lmbda,
        safe_set_dense) = cyclic_coordinate_descent(X_binned, y, lmbda=lmbda,
                                                    epsilon=epsilon,
                                                    f=f, n_epochs=n_epochs,
                                                    screening=screening,
                                                    store_history=True)

    print("safe set dense = ", safe_set_dense)

    # # Plot primal objective function (=primal_hist)
    # obj = primal_hist

    # x = np.arange(1, len(obj) + 1)

    # plt.plot(x, obj, label='cyclic_cd', color='blue')
    # plt.yscale('log')
    # plt.title("Cyclic CD Objective")
    # plt.xlabel('n_iter')
    # plt.ylabel('f obj')
    # plt.legend(loc='best')
    # plt.show()

    # # List of abscissa to plot the evolution of the parameters
    # list_epochs = []
    # for i in range(len(dual_hist)):
    #     list_epochs.append(10 * i)

    # print("length list epochs : ", len(list_epochs))
    # print("length r list : ", len(r_list))

    # # Plot history of the radius
    # plt.plot(list_epochs, r_list, label='radius', color='red')
    # plt.yscale('log')
    # plt.title("Convergence of the radius of the safe sphere")
    # plt.xlabel("n_epochs")
    # plt.ylabel("Radius")
    # plt.legend(loc='best')
    # plt.show()

    # # Plot Dual history vs Primal history
    # plt.plot(list_epochs, dual_hist, label='dual', color='red')
    # plt.plot(list_epochs, obj, label='primal', color='blue')
    # plt.yscale('log')
    # plt.title("Primal VS Dual Monitoring")
    # plt.xlabel('n_epochs')
    # plt.ylabel('optimization problem')
    # plt.legend(loc='best')
    # plt.show()

    # # Plot Dual gap
    # plt.plot(list_epochs, gap_hist, label='dual gap', color='cyan')
    # plt.yscale('log')
    # plt.title("Convergence of the Duality Gap")
    # plt.xlabel('n_epochs')
    # plt.ylabel('Duality gap')
    # plt.legend(loc='best')
    # plt.show()

    # # # Plot number of features in active set
    # plt.plot(list_epochs, n_active_features_true,
    #          label='number of active features', color='magenta')
    # plt.yscale('log')
    # plt.title("Evolution of the number of active features")
    # plt.xlabel('n_epochs')
    # plt.ylabel('Number of active features')
    # plt.legend(loc='best')
    # plt.show()

    # Comparison between the linear regression on the original dataset (before
    # discretization, taking continuous values) and the discretized dataset
    # (taking only a certain number of discrete values defined by the bins)

    # 1. Prediction on the original dataset

    # Defines the setting of the subplots
    # n_epochs = 500
    # fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 4))
    # # Linear regression on the original dataset
    # reg = LinearRegression().fit(X, y)
    # # Lasso regression on the original dataset
    # lasso = sklearn_Lasso(alpha=lmbda / len(X), fit_intercept=False,
    #                       normalize=False, max_iter=n_epochs,
    #                       tol=1e-15).fit(X, y)

    # ax1.plot(X[:, 0], reg.predict(X), linewidth=2, color='blue',
    #          label='linear regression')

    # # Decision Tree on the original dataset
    # tree_reg = DecisionTreeRegressor(min_samples_split=3,
    #                                  random_state=0).fit(X, y)
    # ax1.plot(X[:, 0], tree_reg.predict(X), linewidth=2, color='red',
    #          label='decision tree')

    # ax1.plot(X[:, 0], y, 'o', c='k')
    # ax1.legend(loc="best")
    # ax1.set_ylabel("Regression output")
    # ax1.set_xlabel("Input feature")
    # ax1.set_title("Result before discretization")

    # # 2. Prediction on the discretized dataset
    # n_bins = 10
    # encode = 'onehot'
    # strategy = 'quantile'
    # enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    # X_binned = enc.fit_transform(X)
    # X_binned = X_binned.tocsc()
    # # Linear Regression on the discretized dataset
    # binning_reg = LinearRegression().fit(X_binned, y)
    # # Lasso onthe discretized dataset
    # binning_lasso = sklearn_Lasso(alpha=lmbda / X_binned.shape[0],
    #                               fit_intercept=False, normalize=False,
    #                               max_iter=n_epochs,
    #                               tol=1e-15).fit(X_binned, y)

    # ax2.plot(X[:, 0], binning_reg.predict(X_binned), linewidth=2, color='blue',
    #          linestyle='-', label='linear regression')
    # enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    # X_binned = enc.fit_transform(X)
    # X_binned = X_binned.tocsc()

    # sparse_lasso_sklearn = sklearn_Lasso(alpha=(lmbda / X_binned.shape[0]),
    #                                      fit_intercept=False,
    #                                      normalize=False,
    #                                      max_iter=n_epochs,
    #                                      tol=1e-14).fit(X_binned, y)
    # # Decision Tree on the discretized dataset
    # binning_tree_reg = DecisionTreeRegressor(min_samples_split=3,
    #                                          random_state=0).fit(X_binned, y)
    # ax2.plot(X[:, 0], binning_tree_reg.predict(X_binned), linewidth=2,
    #          color='red', linestyle=':', label='decision tree')

    # ax2.plot(X[:, 0], y, 'o', c='k')
    # ax2.vlines(enc.bin_edges_[0], *plt.gca().get_ylim(), linewidth=1,
    #            alpha=0.2)
    # ax2.legend(loc="best")
    # ax2.set_xlabel("Input features")
    # ax2.set_title("Result after discretization")
    # plt.tight_layout()
    # plt.show()

    # # Assessment of the model by computing the crossval score
    # dense_lasso = Lasso(lmbda=lmbda, epsilon=epsilon, f=f,
    #                     n_epochs=n_epochs, screening=screening,
    #                     store_history=store_history).fit(X, y)

    # original_dense_lasso_cv_score = dense_lasso.score(X, y)
    # print("original dense lasso crossval score : ",
    #       original_dense_lasso_cv_score)

    # sparse_lasso = Lasso(lmbda=lmbda, epsilon=epsilon, f=f, n_epochs=n_epochs,
    #                      screening=screening,
    #                      store_history=store_history).fit(X_binned, y)

    # binning_lasso_cv_score = sparse_lasso.score(X_binned, y)
    # print("binning lasso crossval score : ", binning_lasso_cv_score)

    # sklearn_dense_cv_score = lasso.score(X, y)
    # print("sklearn dense lasso crossval score : ", sklearn_dense_cv_score)

    # sklearn_binning_cv_score = binning_lasso.score(X_binned, y)
    # print("sklearn binning lasso crossval score : ", sklearn_binning_cv_score)

    # # PCA : Principal Component Analysis on the original dataset
    # pca = PCA(n_components=2, svd_solver='full')
    # pca.fit(X)
    # svd = pca.singular_values_
    # var_ratio = pca.explained_variance_ratio_
    # PCs = pca.transform(X)
    # print("Variances of the principal components : ", svd)
    # print("Explained variance ratio : ", var_ratio)
    # print("Principal Components : ", PCs)

    # # 3. Prediction on the original dataset after having carried out a PCA

    # # Defines the setting of the subplots
    # fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 4))
    # # Linear regression on the original dataset
    # reg = LinearRegression().fit(X, y)
    # ax1.plot(PCs[:, 0], reg.predict(X), linewidth=2, color='blue',
    #          label='linear regression')

    # # Decision Tree on the original dataset
    # tree_reg = DecisionTreeRegressor(min_samples_split=3,
    #                                  random_state=0).fit(X, y)
    # ax1.plot(PCs[:, 0], tree_reg.predict(X), linewidth=2, color='red',
    #          label='decision tree')

    # ax1.plot(PCs[:, 0], y, 'o', c='k')
    # ax1.legend(loc="best")
    # ax1.set_ylabel("Regression output")
    # ax1.set_xlabel("Input feature")
    # ax1.set_title("Result before discretization")

    # # 4. Prediction on the discretized dataset after having carried out a PCA
    # n_bins = 10
    # encode = 'onehot'
    # strategy = 'quantile'
    # enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    # X_binned = enc.fit_transform(X)
    # # Linear Regression on the discretized dataset
    # binning_reg = LinearRegression().fit(X_binned, y)
    # ax2.plot(PCs[:, 0], binning_reg.predict(X_binned), linewidth=2,
    #          color='blue', linestyle='-', label='linear regression')
    # # Decision Tree on the discretized dataset
    # binning_tree_reg = DecisionTreeRegressor(min_samples_split=3,
    #                                          random_state=0).fit(X_binned, y)
    # ax2.plot(PCs[:, 0], binning_tree_reg.predict(X_binned), linewidth=2,
    #          color='red', linestyle=':', label='decision tree')

    # ax2.plot(PCs[:, 0], y, 'o', c='k')
    # ax2.vlines(enc.bin_edges_[0], *plt.gca().get_ylim(), linewidth=1,
    #            alpha=0.2)
    # ax2.legend(loc="best")
    # ax2.set_xlabel("Input features")
    # ax2.set_title("Result after discretization")
    # plt.tight_layout()
    # plt.show()

    # Results
    # Before discretization, the predictions are different whether we use
    # linear regression or decision tree : decision tree is able to build
    # models which are much more complex than linear regression
    # After discretization, the predictions are exactly the same whatever we
    # use linear regression or decision tree.

    # Bar Plots
    encode = 'onehot'
    strategy = 'quantile'
    time_list = [0 for i in range(3)]
    time_list_sklearn = [0 for i in range(3)]
    scores_list = [0 for i in range(3)]
    sklearn_scores = [0 for i in range(3)]

    for rep in range(1):
        X, y = simu(beta, n_samples=10, corr=0.5, for_logreg=False,
                    random_state=rep)
        for n_bins in range(2, 5):
            enc = KBinsDiscretizer(n_bins=n_bins, encode=encode,
                                   strategy=strategy)
            X_binned = enc.fit_transform(X)
            X_binned = X_binned.tocsc()

            start1 = time.time()

            sparse_lasso = Lasso(lmbda=lmbda, epsilon=epsilon, f=f,
                                 n_epochs=n_epochs,
                                 screening=True,
                                 store_history=False).fit(X_binned, y)

            end1 = time.time()
            delay_sparse = end1 - start1

            cv_score = sparse_lasso.score(X_binned, y)
            scores_list[n_bins - 2] += cv_score
            start2 = time.time()

            sparse_lasso_sklearn = sklearn_Lasso(alpha=(lmbda
                                                        / X_binned.shape[0]),
                                                 fit_intercept=False,
                                                 normalize=False,
                                                 max_iter=n_epochs,
                                                 tol=1e-15).fit(X_binned, y)

            end2 = time.time()
            delay_sklearn = end2 - start2

            cv_score_sklearn = sparse_lasso_sklearn.score(X_binned, y)
            sklearn_scores[n_bins - 2] += cv_score_sklearn

            time_list[n_bins - 2] += delay_sparse
            time_list_sklearn[n_bins - 2] += delay_sklearn

    sklearn_scores = [i / 1 for i in sklearn_scores]
    scores_list = [i / 1 for i in scores_list]
    time_list = [i / 1 for i in time_list]
    time_list_sklearn = [i / 1 for i in time_list_sklearn]

    print("list sklearn sparse cv scores : ", sklearn_scores)
    print("list time sklearn : ", time_list_sklearn)
    print("list sparse cv scores : ", scores_list)
    print("list time : ", time_list)

    bins = ['2', '3', '4']

    x = np.arange(len(bins))  # the label locations
    width = 0.35  # the width of the bars
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(x - width/2, time_list, width,
                    label='our sparse lasso solver')
    rects2 = ax.bar(x + width/2, time_list_sklearn, width,
                    label='sklearn sparse lasso solver')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('execution time')
    ax.set_title('time by bins and solver')
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.legend()

    def autolabel(rects, scale):
        """Attach a text label above each bar in *rects*, displaying its
        height.
        """

        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height * scale, 0)/scale),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1, 10000)
    autolabel(rects2, 10000)

    fig.tight_layout()
    plt.show()
    
    bins = ['2', '3', '4']
    x = np.arange(len(bins))  # the label locations
    width = 0.35  # the width of the bars

    print("scores list shape : ", len(scores_list))
    print("sklearn scores list shape : ", len(sklearn_scores))
    print("first element scores list : ", type(scores_list[0]))
    print("first element scores list sklearn : ", type(sklearn_scores[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1_score = ax.bar(x - width/2, scores_list, width,
                          label='our sparse lasso solver')
    rects2_score = ax.bar(x + width/2, sklearn_scores, width,
                          label='sklearn sparse lasso solver')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('crossval score')
    ax.set_title('crossval score by bins and solver')
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.legend()

    autolabel(rects1_score, 1000)
    autolabel(rects2_score, 1000)

    fig.tight_layout()
    plt.show()

    # Performance figures

    delay_list_dense = list()
    delay_list_dense_sklearn = list()

    primal_dense_list = list()
    primal_dense_list_sklearn = list()
    dense_cv_list = list()
    dense_cv_list_sklearn = list()

    for n_epochs in range(1000, 100000, 1000):

        start_dense = time.time()

        dense_lasso = Lasso(
            lmbda=lmbda,
            epsilon=epsilon,
            f=f,
            n_epochs=n_epochs,
            screening=screening,
            store_history=store_history,
        ).fit(X_binned, y)

        end_dense = time.time()
        delay_dense = end_dense - start_dense
        delay_list_dense.append(delay_dense)

        primal_dense = dense_lasso.P_lmbda
        primal_dense_list.append(primal_dense)

        dense_cv_score = dense_lasso.score(X_binned, y)
        dense_cv_list.append(dense_cv_score)

        start_dense_sklearn = time.time()

        dense_lasso_sklearn = sklearn_Lasso(
            alpha=(lmbda / X_binned.shape[0]),
            fit_intercept=False,
            normalize=False,
            max_iter=n_epochs,
            tol=1e-15,
        ).fit(X_binned, y)

        end_dense_sklearn = time.time()
        delay_dense_sklearn = end_dense_sklearn - start_dense_sklearn
        delay_list_dense_sklearn.append(delay_dense_sklearn)

        slopes = dense_lasso_sklearn.coef_

        primal_function = (1 / 2) * np.linalg.norm(
            y - X_binned.dot(slopes.T), 2
        ) ** 2 + lmbda * np.linalg.norm(slopes, 1)

        primal_dense_list_sklearn.append(primal_function)

        dense_cv_score_sklearn = dense_lasso_sklearn.score(X_binned, y)
        dense_cv_list_sklearn.append(dense_cv_score_sklearn)

    bins = ['2', '3', '4']

    print("shape = ", len(delay_list_dense))

    x = np.arange(len(bins))  # the label locations
    width = 0.35  # the width of the bars
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(x - width/2, delay_list_dense, width,
                    label='our dense lasso solver')
    rects2 = ax.bar(x + width/2, delay_list_dense_sklearn, width,
                    label='sklearn dense lasso solver')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('execution time')
    ax.set_title('time by bins and solver in dense')
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.legend()

    autolabel(rects1, 10000)
    autolabel(rects2, 10000)

    fig.tight_layout()
    plt.show()
    
    bins = ['2', '3', '4']
    x = np.arange(len(bins))  # the label locations
    width = 0.35  # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1_score = ax.bar(x - width/2, dense_cv_list, width,
                          label='our dense lasso solver')
    rects2_score = ax.bar(x + width/2, dense_cv_list_sklearn, width,
                          label='sklearn dense lasso solver')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('crossval score')
    ax.set_title('crossval score by bins and solver in dense')
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.legend()

    autolabel(rects1_score, 1000)
    autolabel(rects2_score, 1000)

    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
