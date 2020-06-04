import numpy as np

import numba
from numba import njit

from numpy.random import randn
from numpy.random import multivariate_normal
from scipy.linalg import toeplitz
from sklearn.linear_model import Lasso as sklearn_Lasso


##########################################################
#                    Sign Function
##########################################################


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

    ST = np.sign(x) * max(abs(x) - u, 0)

    return ST


#############################################################################
#                           Data Simulation
#############################################################################


def simu(beta, n_samples=1000, corr=0.5, for_logreg=False):
    n_features = len(beta)
    cov = toeplitz(corr ** np.arange(0, n_features))

    # Features Matrix
    X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)

    # Target labels vector with noise
    y = np.dot(X, beta) + randn(n_samples)

    if for_logreg:
        y = sign(y)

    return X, y


################################################
#    Active set and zero set : Equation 7
################################################


def active_set_vs_zero_set(X, c, r):
    """
    Parameters
    ----------
    X: numpy.ndarray, shape = (n_samples, n_features)
       features matrix

    c: ndarray, shape = (n_samples, )
       center of the safe sphere

    r: float
       radius of the safe sphere

    Returns
    -------
    A_C: ndarray, shape = (n_idx_active_features, )
         active set : contains the indices of the relevant features

    Z_C: ndarray, shape = (n_idx_zero_features, )
         zero set : contains the indices of the irrelevant features
    """
    A_C = []
    Z_C = []
    n_features = X.shape[1]
    for j in range(n_features):
        x_j = X[:, j]
        mu = np.abs(np.dot(x_j.T, c)) + r * np.linalg.norm(x_j)
        if mu >= 1:
            A_C.append(j)
        else:
            Z_C.append(j)

    return A_C, Z_C


##############################################################################
#   Minimization of the Primal Problem with Coordinate Descent Algorithm
##############################################################################


def cyclic_coordinate_descent(X, y, lmbda, epsilon, f, n_epochs=5000,
                              screening=True):
    """Solver : cyclic coordinate descent

    Parameters
    ----------

    X: numpy.ndarray, shape (n_samples, n_features)
        features matrix

    y: ndarray, shape (n_samples, )
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

    Returns
    -------
    beta: ndarray, shape(n_features,)
        primal parameters vector

    theta: ndarray, shape(n_samples, )
        dual parameters vector

    all_objs: ndarray, shape(n_features)
        residuals vector
    """

    # Initialisation of the parameters

    n_samples, n_features = X.shape
    all_objs = []

    beta = np.zeros(n_features)
    theta = np.zeros(n_samples)
    residuals = y - X.dot(beta)

    # Computation of the lipschitz constants vector

    lips_const = np.linalg.norm(X, axis=0)**2

    A_c = list(range(n_features))

    # Iterations of the algorithm
    for k in range(n_epochs):
        for i in A_c:
            # One cyclicly updates the i^{th} coordinate corresponding to the
            # rest in the Euclidean division by the number of features
            # This allows to always selecting an index between 1 and n_features
            old_beta_i = beta[i].copy()
            step = 1 / lips_const[i]
            grad = np.dot(X[:, i], residuals)

            # Update of the parameters
            beta[i] += step * grad

            # Apply proximal operator
            if beta[i] > 0:
                beta[i] = max(abs(beta[i]) - step * lmbda, 0)
            elif beta[i] < 0:
                beta[i] = -max(abs(beta[i]) - step * lmbda, 0)
            else:
                beta[i] = 0

            # Update of the residuals
            if old_beta_i != beta[i]:
                residuals += np.dot(X[:, i], old_beta_i - beta[i])

        if k % f == 0:
            # Computation of theta
            theta = ((y - np.dot(X, beta))
                     / (lmbda*max(np.max(np.abs(residuals)), 1)))

            # Computation of the primal problem
            P_lmbda = 0.5 * np.linalg.norm(np.dot(X, beta) - y, 2)**2
            P_lmbda += lmbda * np.linalg.norm(beta, 1)
            # Objective function related to the primal
            all_objs.append(P_lmbda)

            # Computation of the dual problem
            D_lmbda = 0.5*np.linalg.norm(y, ord=2)**2
            D_lmbda -= (((lmbda**2) / 2)
                        * np.linalg.norm(theta - y / lmbda, ord=2)**2)

            # Computation of the dual gap
            G_lmbda = P_lmbda - D_lmbda

            if screening:
                # Computation of the radius of the gap safe sphere
                r = np.sqrt(2*np.abs(G_lmbda))/lmbda

                # Computation of the active set
                for j in A_c:
                    x_j = X[:, j]
                    mu = np.abs(np.dot(x_j.T, theta)) + r * np.linalg.norm(x_j)
                    if mu < 1:
                        A_c.remove(j)

                if np.abs(G_lmbda) <= epsilon:
                    break

    return (beta, theta, all_objs)


def main():
    # Data Simulation
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 30
    beta = rng.randn(n_features)
    lmbda = 1.

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False)

    epsilon = 1e-14
    f = 10
    n_epochs = 100000

    beta_hat, theta_hat, all_objs = \
        cyclic_coordinate_descent(X,
                                  y,
                                  lmbda,
                                  epsilon,
                                  f,
                                  n_epochs,
                                  screening=True)


if __name__ == "__main__":
    main()
