import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randn
from numpy.random import multivariate_normal
from scipy.linalg import toeplitz

from numba import njit

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


def simu(beta, n_samples=1000, corr=0.5, for_logreg=False):
    n_features = len(beta)
    cov = toeplitz(corr ** np.arange(0, n_features))

    # Features Matrix
    X = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X = np.asfortranarray(X)

    # Target labels vector with noise
    y = np.dot(X, beta) + randn(n_samples)

    if for_logreg:
        y = sign(y)

    return X, y


##############################################################################
#   Minimization of the Primal Problem with Coordinate Descent Algorithm
##############################################################################

@njit
def cyclic_coordinate_descent(X, y, lmbda, epsilon, f, n_epochs=5000,
                              screening=True, store_history=True):
    """Solver : cyclic coordinate descent

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

    residuals = y - X.dot(beta)

    # Computation of the lipschitz constants vector

    L = (X**2).sum(0)

    A_c = list(range(n_features))

    # Iterations of the algorithm
    for k in range(n_epochs):
        for i in A_c:
            # One cyclicly updates the i^{th} coordinate corresponding to the
            # rest in the Euclidean division by the number of features
            # This allows to always selecting an index between 1 and n_features
            old_beta_i = beta[i]
            step = 1 / L[i]
            grad = np.dot(X[:, i], residuals)

            # Update of the parameters
            beta[i] += step * grad

            # Apply proximal operator
            beta[i] = soft_thresholding(step * lmbda, beta[i])

            # Update of the residuals
            if old_beta_i != beta[i]:
                residuals += (old_beta_i - beta[i]) * X[:, i]

        if k % f == 0:
            # Computation of theta
            theta = (residuals
                     / (lmbda * max(np.max(np.abs(residuals/lmbda)), 1)))

            # Computation of the primal problem
            P_lmbda = 0.5 * residuals.dot(residuals)
            P_lmbda += lmbda * np.linalg.norm(beta, 1)

            # Computation of the dual problem
            D_lmbda = 0.5*np.linalg.norm(y, ord=2)**2
            D_lmbda -= (((lmbda**2) / 2)
                        * np.linalg.norm(theta - y / lmbda, ord=2)**2)

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
                r = np.sqrt(2*np.abs(G_lmbda))/lmbda
                # r_list.append(r)

                # Computation of the active set
                for j in A_c:
                    # mu = mu_B(X[:, j], theta, r)
                    mu = (np.abs(np.dot(X[:, j].T, theta))
                          + r * np.linalg.norm(X[:, j]))
                    if mu < 1:
                        A_c.remove(j)
                if store_history:
                    n_active_features.append(len(A_c))
                    r_list.append(r)

                if np.abs(G_lmbda) <= epsilon:
                    break

    return (beta, primal_hist, dual_hist, gap_hist, r_list,
            n_active_features, theta, P_lmbda, D_lmbda, G_lmbda)


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


def main():
    # Data Simulation
    np.random.seed(0)
    n_samples, n_features = 10, 30
    beta = np.random.randn(n_features)
    lmbda = 0.1

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False)

    # Minimization of the Primal Problem with Coordinate Descent Algorithm
    epsilon = 10**(-20)
    f = 10

    (beta_hat_cyclic_cd_false,
        primal_hist,
        dual_hist,
        gap_hist,
        r_list,
        n_active_features_true,
        theta_hat_cyclic_cd,
        P_lmbda,
        D_lmbda,
        G_lmbda) = cyclic_coordinate_descent(X,
                                             y,
                                             lmbda,
                                             epsilon,
                                             f,
                                             n_epochs=10000,
                                             screening=False,
                                             store_history=True)

    (beta_hat_cyclic_cd_true,
        primal_hist,
        dual_hist,
        gap_hist,
        r_list,
        n_active_features_true,
        theta_hat_cyclic_cd,
        P_lmbda,
        D_lmbda,
        G_lmbda) = cyclic_coordinate_descent(X,
                                             y,
                                             lmbda,
                                             epsilon,
                                             f,
                                             n_epochs=10000,
                                             screening=True,
                                             store_history=True)

    # Plot primal objective function (=primal_hist)
    obj = primal_hist

    x = np.arange(1, len(obj)+1)

    plt.plot(x, obj, label='cyclic_cd', color='blue')
    plt.yscale('log')
    plt.title("Cyclic CD Objective")
    plt.xlabel('n_iter')
    plt.ylabel('f obj')
    plt.legend(loc='best')
    plt.show()

    # List of abscissa to plot the evolution of the parameters
    list_epochs = []
    for i in range(len(dual_hist)):
        list_epochs.append(10*i)

    # Plot history of the radius
    plt.plot(list_epochs, r_list, label='radius', color='red')
    plt.yscale('log')
    plt.title("Convergence of the radius of the safe sphere")
    plt.xlabel("n_epochs")
    plt.ylabel("Radius")
    plt.legend(loc='best')
    plt.show()

    # Plot Dual history vs Primal history
    plt.plot(list_epochs, dual_hist, label='dual', color='red')
    plt.plot(list_epochs, obj, label='primal', color='blue')
    plt.yscale('log')
    plt.title("Primal VS Dual Monitoring")
    plt.xlabel('n_epochs')
    plt.ylabel('optimization problem')
    plt.legend(loc='best')
    plt.show()

    # Plot Dual gap
    plt.plot(list_epochs, gap_hist, label='dual gap', color='cyan')
    plt.yscale('log')
    plt.title("Convergence of the Duality Gap")
    plt.xlabel('n_epochs')
    plt.ylabel('Duality gap')
    plt.legend(loc='best')
    plt.show()

    # Plot number of features in active set
    plt.plot(list_epochs, n_active_features_true,
             label='number of active features', color='magenta')
    plt.yscale('log')
    plt.title("Evolution of the number of active features")
    plt.xlabel('n_epochs')
    plt.ylabel('Number of active features')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()
