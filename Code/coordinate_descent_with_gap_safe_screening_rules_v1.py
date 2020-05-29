import numpy as np
import matplotlib.pyplot as plt
# from scipy import linalg
from numpy.random import multivariate_normal
from scipy.linalg.special_matrices import toeplitz
from numpy.random import randn
# import seaborn as sns

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

    # Target labels vector with noise
    y = np.dot(X, beta) + randn(n_samples)

    if for_logreg:
        y = sign(y)

    return X, y


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

    y: numpy.array, shape (n_samples, )
       target labels vector

    lmbda: float
           regularization parameter

    epsilon: float
             stopping criterion

    f: int
       frequency

    n_epochs: int, default = 5000
              number of iterations

    screening: bool, default = True
               defines whether or not one adds screening to the solver

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

    all_objs: numpy.array, shape(n_features)
              residuals vector

    """

    # Initialisation of the parameters

    n_samples, n_features = X.shape
    all_objs = []

    beta = np.zeros(n_features)
    theta = np.zeros(n_samples)
    A_C = np.zeros(n_features)

    nb_active_features = []
    r_list = []
    primal_hist = []
    dual_hist = []
    gap_hist = []
    theta_hist = []
    A_C_hist = []

    residuals = y - X.dot(beta)

    # Computation of the lipschitz constants vector

    lips_const = np.linalg.norm(X, axis=0)**2

    A_c = range(n_features)

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
            beta[i] = soft_thresholding(step * lmbda, beta[i])

            # Update of the residuals
            if old_beta_i != beta[i]:
                residuals += np.dot(X[:, i], old_beta_i - beta[i])

        if (k % f == 1) or (k == 0):
            # Computation of theta
            theta = compute_theta_k(X, y, beta, lmbda)
            theta_hist.append(theta)

            # Computation of the primal problem
            P_lmbda = primal_pb(X, y, beta, lmbda)
            primal_hist.append(P_lmbda)

            # Computation of the dual problem
            D_lmbda = dual_pb(y, theta, lmbda)
            dual_hist.append(D_lmbda)

            # Computation of the dual gap
            G_lmbda = duality_gap(P_lmbda, D_lmbda)
            gap_hist.append(G_lmbda)

            # Objective function related to the primal
            all_objs.append(P_lmbda)

            if screening:
                # Computation of the radius of the gap safe sphere
                r = radius(G_lmbda, lmbda)
                r_list.append(r)

                # Computation of the active set
                A_C, _ = active_set_vs_zero_set(X, theta, r)
                nb_active_features.append(len(A_C))
                A_C_hist.append(A_C)

                if np.abs(G_lmbda) <= epsilon:
                    break

    return (beta, A_C_hist, primal_hist, dual_hist, gap_hist, theta_hist,
            r_list, nb_active_features,
            all_objs, theta, P_lmbda, D_lmbda, G_lmbda)


###########################################################################
#                     Computation of R hat : Theorem 2
###########################################################################


def R_primal(X, y, beta, lmbda):
    """
    Parameters
    ----------

    X: numpy.ndarray, shape=(n_samples, n_features)
       features matrix

    y: numpy.array, shape=(n_samples, )
       target labels vector

    beta: numpy.array, shape=(n_features, )
          primal optimal parameters vector

    lmbda: float
           regularization parameter

    Returns
    -------

    R_hat_lmbda: float
                 primal radius of the dome
    """

    R_hat_lmbda = ((1 / lmbda)*np.max(np.linalg.norm(y)**2
                   - np.linalg.norm(np.dot(X, beta) - y)**2
                   - 2*lmbda*np.linalg.norm(beta, 1), 0)**(1/2))

    return R_hat_lmbda


###########################################################################
#                     Computation of R chech : Theorem 2
###########################################################################


def R_dual(y, theta, lmbda):
    """
    Parameters
    ----------
    y: numpy.array, shape=(n_samples, )
        target labels vector

    theta: numpy.array, shape=(n_features, )
        dual optimal parameters vector

    lmbda: float
        regularization parameter

    Returns
    -------
    R_inv_hat_lmbda: float
       dual radius of the dome
    """
    R_inv_hat_lmbda = np.linalg.norm(theta - y / lmbda)

    return R_inv_hat_lmbda


##################################################################
#    Radius of the safe sphere in closed form : Theorem 2
##################################################################

def radius_thm2(R_hat_lmbda, R_inv_hat_lmbda):
    """Compute the radius of the safe sphere region

    Parameters
    ----------

    R_hat_lmbda: float
        primal radius of the safe dome region

    R_inv_hat_lmbda: float
        dual radius of the safe dome region

    Returns
    -------
    r_lmbda: float
        radius of the safe sphere region
    """

    r_lmbda = np.sqrt(R_inv_hat_lmbda**2 - R_hat_lmbda**2)

    return r_lmbda


def radius(G_lmbda, lmbda):
    """
    Parameters
    ----------
    G_lmbda: float
             duality gap

    lmbda: float
           regularization parameter

    Returns
    -------
    r: float
       radius of the safe sphere
    """

    r = np.sqrt(2*np.abs(G_lmbda))/lmbda
    return r


############################################################################
#   Mu Function applied to the safe sphere in closed form : Equation 9
############################################################################


def mu_B(x_j, c, r):
    """Function mu applied to the sphere of center c and radius r
       for the jth feature X[:,j]

    Parameters
    ----------
    x_j: numpy.array, shape=(n_samples, )
         jth feature X[:,j]

    c: float
       center of the sphere

    r: float
       radius of the sphere

    Returns
    -------

    mu: float
        maximum value between the scalar products of theta and x_j
        and theta and -x_j
    """

    mu = np.abs(np.dot(x_j.T, c)) + r * np.linalg.norm(x_j)

    return mu


###############################################################
#    compute dual point
###############################################################


def compute_theta_k(X, y, beta_hat, lmbda):
    """Maximization of the dual problem
       Orthogonal projection of the center of the safe sphere
       onto the feasible set

    Parameters
    ----------
    X: numpy.ndarray, shape = (n_samples, n_features)
       features matrix

    y: numpy.array, shape = (n_samples, )
       target labels vector

    beta_hat: numpy.array shape = (n_features, )
              current primal optimal parameters vector

    lmbda: float
           regularization parameter

    Returns
    -------

    theta_hat: numpy.array, shape = (n_samples, )
               dual optimal parameters vector

    """

    # Link equation : Equation 3
    residus = (y - np.dot(X, beta_hat))/lmbda

    # Orthogonal projection of theta_hat onto the feasible set
    theta_hat = residus / max(np.max(np.abs(residus)), 1)

    return theta_hat


################################################
#    Active set and zero set : Equation 7
################################################


def active_set_vs_zero_set(X, c, r):
    """
    Parameters
    ----------
    X: numpy.ndarray, shape = (n_samples, n_features)
       features matrix

    c: numpy.array, shape = (n_samples, )
       center of the safe sphere

    r: float
       radius of the safe sphere

    Returns
    -------
    A_C: numpy.array, shape = (n_idx_active_features, )
         active set : contains the indices of the relevant features

    Z_C: numpy.array, shape = (n_idx_zero_features, )
         zero set : contains the indices of the irrelevant features
    """
    A_C = []
    Z_C = []
    n_features = X.shape[1]
    for j in range(n_features):
        x_j = X[:, j]
        mu = mu_B(x_j, c, r)
        if mu >= 1:
            A_C.append(j)
        else:
            Z_C.append(j)

    return A_C, Z_C


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


##########################################################
#                  Primal Problem : Equation 1
##########################################################


def primal_pb(X, y, beta, lmbda):
    """
    Parameters
    ----------
    X: numpy.ndarray, shape = (n_samples, n_features)
       features matrix

    y: numpy.array, shape = (n_samples, )
       target labels vector

    beta: numpy.array, shape = (n_features, )
          initial vector of primal parameters

    lmbda: float
           regularization parameter

    Returns
    -------

    P_lmbda: float
             value of the primal problem for a given beta vector
    """

    P_lmbda = 0.5 * np.linalg.norm(np.dot(X, beta) - y, 2)**2
    P_lmbda += lmbda * np.linalg.norm(beta, 1)

    return P_lmbda


##########################################################
#                  Dual Problem : Equation 2
##########################################################


def dual_pb(y, theta, lmbda):
    """
    Parameters
    ----------
    y: numpy.array, shape = (n_features, )

    theta: numpy.array, shape = (n_samples, )
           initial vector of dual parameters

    lmbda: float
           regularization parameter

    Returns
    -------
    D_lmbda: float
             value of the dual problem for a given theta vector
    """
    D_lmbda = 0.5*np.linalg.norm(y, ord=2)**2
    D_lmbda -= ((lmbda**2) / 2) * np.linalg.norm(theta - y / lmbda, ord=2)**2

    return D_lmbda


##########################################################
#                  Duality Gap : Equation 2
##########################################################


def duality_gap(P_lmbda, D_lmbda):
    """
    Parameters
    ----------

    P_lmbda: float
             value of the primal problem at the optimum beta_hat

    D_lmbda: float
             value of the dual problem at the optimum theta_hat

    Returns
    -------
    G_lmbda: float
             duality gap between the primal optimal and the dual optimal

    """

    # Duality gap
    # If it is equal to 0 then the primal optimal is equal to the dual optimal
    # and the strong duality holds
    # If there exists a gap between the primal optimal and the dual optimal
    # then one only has the weak duality with P_lmbda >= D_lmbda
    G_lmbda = P_lmbda - D_lmbda

    return G_lmbda


##########################################################
#              Gap Safe Sphere : Equation 18
##########################################################


def gap_safe_sphere(X, c, r):
    """
    Parameters
    ----------

    X: numpy.ndarray, shape = (n_samples, n_features)
       features matrix

    c: numpy.array, shape = (n_samples,)
       center of the sphere

    r: float
       radius of the sphere

    Returns
    -------

    C_k: interval
         sphere of center c and of radius r
    """

    _, p = X.shape

    C_k = []

    for j in range(p):
        if np.linalg.norm(X[:, j] - c) <= r:
            C_k.append(j)

    return C_k


def main():
    # Data Simulation
    np.random.seed(0)
    n_samples, n_features = 10, 30
    beta = np.random.randn(n_features)
    lmbda = 0.1

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False)

    # Minimization of the Primal Problem with Coordinate Descent Algorithm
    epsilon = 10**(-14)
    f = 10

    (beta_hat_cyclic_cd_false,
        A_C_hist_false,
        primal_hist,
        dual_hist,
        gap_hist,
        theta_hist_false,
        r_list,
        n_active_features_true,
        objs_cyclic_cd,
        theta_hat_cyclic_cd,
        P_lmbda,
        D_lmbda,
        G_lmbda) = cyclic_coordinate_descent(X,
                                             y,
                                             lmbda,
                                             epsilon,
                                             f,
                                             n_epochs=1000,
                                             screening=False)

    print("Beta without screening : ", beta_hat_cyclic_cd_false)
    # Test KKT for theta without screening
    # kkt_list = []
    # for j in range(X.shape[1]):
    #     kkt = np.dot(X[:, j].T, theta_hat_cyclic_cd)
    #     kkt_list.append(kkt)

    # print("kkt list : ", kkt_list)
    # print("beta :", beta_hat_cyclic_cd_false)

    (beta_hat_cyclic_cd_true,
        A_C_hist_true,
        primal_hist,
        dual_hist,
        gap_hist,
        theta_hist_true,
        r_list,
        n_active_features_true,
        objs_cyclic_cd,
        theta_hat_cyclic_cd,
        P_lmbda,
        D_lmbda,
        G_lmbda) = cyclic_coordinate_descent(X,
                                             y,
                                             lmbda,
                                             epsilon,
                                             f,
                                             n_epochs=1000,
                                             screening=True)

    print("Beta with screening : ", beta_hat_cyclic_cd_true)

    # Test KKT with screening
    # kkt_list_true = []
    # for j in range(X.shape[1]):
    #     kkt = np.dot(X[:, j].T, theta_hat_cyclic_cd)
    #     kkt_list_true.append(kkt)

    # print("kkt list true : ", kkt_list_true)
    # print("beta :", beta_hat_cyclic_cd_true)

    obj = objs_cyclic_cd

    x = np.arange(1, len(obj)+1)

    plt.plot(x, obj, label='cyclic_cd', color='blue')
    plt.yscale('log')
    plt.title("Cyclic CD Objective")
    plt.xlabel('n_iter')
    plt.ylabel('f obj')
    plt.legend(loc='best')
    plt.show()

    # Plots
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