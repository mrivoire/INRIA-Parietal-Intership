import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy.random import randn
from numpy.random import multivariate_normal
from scipy.linalg import toeplitz

from numba import njit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso as sklearn_Lasso
from scipy.sparse import csc_matrix

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


############################################################################
#                   Cyclic CD For Dense Features Matrix
############################################################################


@njit
def cyclic_coordinate_descent(X, y, lmbda, epsilon, f, n_epochs, screening,
                              store_history):
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

    n_samples, n_features = np.shape(X)

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
            # One cyclicly updates the i^{th} coordinate corresponding to
            # the rest in the Euclidean division by the number of features
            # This allows to always selecting an index between 1 and
            # n_features

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
            theta = (residuals / (lmbda
                                  * max(np.max(np.abs(residuals
                                                      / lmbda)), 1)))

            # Computation of the primal problem
            P_lmbda = 0.5 * residuals.dot(residuals)
            P_lmbda += lmbda * np.linalg.norm(beta, 1)

            # Computation of the dual problem
            D_lmbda = 0.5 * np.linalg.norm(y, ord=2)**2
            D_lmbda -= (((lmbda**2) / 2)
                        * np.linalg.norm(theta - y
                                         / lmbda, ord=2)**2)

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
                for j in A_c:
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


#########################################################################
#                 Cyclic CD For Sparse Features Matrix
#########################################################################


def sparse_cd(X_data, X_indices, X_indptr, y, lmbda, epsilon, f, n_epochs,
              screening, store_history):
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

    # print(y)
    residuals = np.copy(y) 
    n_active_features = []
    r_list = []
    primal_hist = []
    dual_hist = []
    gap_hist = []
    theta_hist = []

    A_c = list(range(n_features))

    L = np.zeros(n_features)

    for k in range(n_epochs):
        for i in A_c:  
            start, end = X_indptr[i:i+2] 
            for ind in range(start, end):
                L[i] += X_data[ind] * X_data[X_indptr[ind]]

            if L[i] == 0.:
                continue

            old_beta_i = beta[i]
            scal = 0.

            # Matrix product between the features matrix X and the residuals
            for ind in range(start, end):
                scal += X_data[ind] * residuals[X_indices[ind]]

            beta[i] = soft_thresholding(beta[i] + scal / L[i], lmbda / L[i])
            diff = old_beta_i - beta[i]

            if diff != 0:
                for ind in range(start, end):
                    residuals[X_indices[ind]] += diff * X_data[ind]

        if k % f == 0:
            # Computation of theta
            theta = (residuals / (lmbda
                                  * max(np.max(np.abs(residuals
                                                      / lmbda)), 1)))

            # Computation of the primal problem
            P_lmbda = 0.5 * residuals.dot(residuals)
            P_lmbda += lmbda * np.linalg.norm(beta, 1)

            # Computation of the dual problem
            D_lmbda = 0.5 * np.linalg.norm(y, ord=2)**2
            D_lmbda -= (((lmbda**2) / 2)
                        * np.linalg.norm(theta - y
                                         / lmbda, ord=2)**2)

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
                for j in A_c:
                    start, end = X_indptr[j:j+2]
                    dot = 0.
                    norm = 0.

                    for ind in range(start, end):
                        dot += X_data[ind] * theta[X_indices[ind]] 
                        norm += X_data[ind]**2
                    
                    mu = np.abs(dot) + r * np.linalg.norm(X_data[ind])

                    # mu = (np.abs(np.dot(X_data[X_indptr[j]].T, theta))
                    #       + r * np.linalg.norm(X_data[X_indptr[j]]))
                    if mu < 1:
                        A_c.remove(j)
                if store_history:
                    n_active_features.append(len(A_c))
                    r_list.append(r)

                if np.abs(G_lmbda) <= epsilon:
                    break

    return (beta, primal_hist, dual_hist, gap_hist, r_list,
            n_active_features, theta, P_lmbda, D_lmbda, G_lmbda)

# https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr


###########################################################################
#                           Class Dense Lasso
###########################################################################

# @njit
class DenseLasso:

    # @njit
    def __init__(self, lmbda, epsilon, f, n_epochs, screening, store_history):

        self.lmbda = lmbda
        self.epsilon = epsilon
        self.f = f
        self.n_epochs = n_epochs
        self.screening = screening
        self.store_history = store_history

        assert epsilon > 0

    # @njit
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

        (beta_hat_cyclic_cd_true,
         primal_hist,
         dual_hist,
         gap_hist,
         r_list,
         n_active_features_true,
         theta_hat_cyclic_cd,
         P_lmbda,
         D_lmbda,
         G_lmbda) = cyclic_coordinate_descent(
            X, y, self.lmbda, self.epsilon, self.f, self.n_epochs,
            self.screening, self.store_history)

        self.slopes = beta_hat_cyclic_cd_true
        self.G_lmbda = G_lmbda
        self.r_list = r_list

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

        y_hat = np.dot(X, self.slopes)

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
        score = -np.mean(np.abs(y - self.predict(X)))

        return score


############################################################################
#                         Class Sparse Lasso
############################################################################


class SparseLasso:

    def __init__(self, lmbda, epsilon, f, n_epochs, screening, store_history):

        self.lmbda = lmbda
        self.epsilon = epsilon
        self.f = f
        self.n_epochs = n_epochs
        self.screening = screening
        self.store_history = store_history

        assert epsilon > 0

    def fit(self, X_data, X_indices, X_indptr, y):
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

        (beta_hat_cyclic_cd_true,
         primal_hist,
         dual_hist,
         gap_hist,
         r_list,
         n_active_features_true,
         theta_hat_cyclic_cd,
         P_lmbda,
         D_lmbda,
         G_lmbda) = sparse_cd(X_data, X_indices, X_indptr, y, self.lmbda,
                              self.epsilon, self.f, self.n_epochs,
                              self.screening, self.store_history)

        self.slopes = beta_hat_cyclic_cd_true
        self.G_lmbda = G_lmbda
        self.r_list = r_list

        return self

    def predict(self, X_data, X_indices, X_indptr):
        """Predict the target from the observations matrix

        Parameters
        ----------
        X: numpy.ndarray, shape = (n_samples, n_features)
        features matrix

        X_data: csr format, data array of the features matrix
        data is an array containing all the non-zero elements of the sparse
        matrix

        X_indices: csr format, index array of the features matrix
        indices is an array mapping each element in data to its column in the
        sparse matrix

        X_indptr: csr format, index array pointer of the features matrix
        indptr is an array mapping the elements of data and indices to the rows
        of the sparse matrix

        Returns
        -------
        y_hat: numpy.array, shape = (n_samples, )
        predicted target vector
        """
        n_features = len(X_indptr) - 1
        n_samples = max(X_indices) + 1

        y_hat = np.zeros(n_features)

        for i in range(n_samples):
            for j in range(n_features):
                y_hat[i] += X_data[X_indices[j]] * self.slopes[X_indptr[j]]

        return y_hat

    def score(self, X_data, X_indices, X_indptr, y):
        """Compute the cross-validation score to assess the performance of the
           model (use negative mean absolute error)

        Parameters
        ----------
        X: numpy.ndarray, shape = (n_samples, n_features)
            features matrix

        X_data: csr format, data array of the features matrix
        data is an array containing all the non-zero elements of the sparse
        matrix

        X_indices: csr format, index array of the features matrix
        indices is an array mapping each element in data to its column in the
        sparse matrix

        X_indptr: csr format, index array pointer of the features matrix
        indptr is an array mapping the elements of data and indices to the rows
        of the sparse matrix

        y: numpy.array, shape = (n_samples, )
            target vector

        Returns
        -------
        score: float
            negative mean absolute error (MAE)
            negative to keep the semantic that higher is better
        """
        score = -np.mean(np.abs(y - self.predict(X_data, X_indices, X_indptr)))

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
    np.random.seed(0)
    n_samples, n_features = 10, 30
    beta = np.random.randn(n_features)
    lmbda = 1.
    f = 10
    epsilon = 1e-14
    n_epochs = 100000
    screening = True
    store_history = True

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False)
   
    X = csc_matrix(X)
    X_data = X.data
    X_indices = X.indices
    X_indptr = X.indptr

    print("X_data : ", X_data)
    print("X_indices : ", X_indices)
    print("X_indptr : ", X_indptr)

    beta_hat = sparse_cd(X_data=X_data, X_indices=X_indices, X_indptr=X_indptr,
                         y=y, lmbda=lmbda, epsilon=epsilon, f=f,
                         n_epochs=n_epochs, screening=screening,
                         store_history=store_history)

    print("beta hat : ", beta_hat)

    sparse_lasso = SparseLasso(lmbda=lmbda, epsilon=epsilon, f=f,
                               n_epochs=n_epochs, screening=screening,
                               store_history=store_history).fit(X_data,
                                                                X_indices,
                                                                X_indptr, y)

    print("sparse lasso : ", sparse_lasso)


    """
    (beta_hat_cyclic_cd_true,
        primal_hist,
        dual_hist,
        gap_hist,
        r_list,
        n_active_features_true,
        theta_hat_cyclic_cd,
        P_lmbda,
        D_lmbda,
        G_lmbda) = cyclic_coordinate_descent(
            X, y, lmbda=lmbda, epsilon=lmbda, f=f, n_epochs=n_epochs,
            screening=True, store_history=True)

    # Plot primal objective function (=primal_hist)
    obj = primal_hist

    x = np.arange(1, len(obj) + 1)

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
        list_epochs.append(10 * i)

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

    # Comparison between the linear regression on the original dataset (before
    # discretization, taking continuous values) and the discretized dataset
    # (taking only a certain number of discrete values defined by the bins)

    # 1. Prediction on the original dataset

    # Defines the setting of the subplots
    n_epochs = 500
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 4))
    # Linear regression on the original dataset
    reg = LinearRegression().fit(X, y)
    # Lasso regression on the original dataset
    lasso = sklearn_Lasso(alpha=lmbda / len(X), fit_intercept=False,
                          normalize=False, max_iter=n_epochs,
                          tol=1e-15).fit(X, y)

    ax1.plot(X[:, 0], reg.predict(X), linewidth=2, color='blue',
             label='linear regression')

    # Decision Tree on the original dataset
    tree_reg = DecisionTreeRegressor(min_samples_split=3,
                                     random_state=0).fit(X, y)
    ax1.plot(X[:, 0], tree_reg.predict(X), linewidth=2, color='red',
             label='decision tree')

    ax1.plot(X[:, 0], y, 'o', c='k')
    ax1.legend(loc="best")
    ax1.set_ylabel("Regression output")
    ax1.set_xlabel("Input feature")
    ax1.set_title("Result before discretization")

    # 2. Prediction on the discretized dataset
    n_bins = 10
    encode = 'onehot'
    strategy = 'quantile'
    enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    X_binned = enc.fit_transform(X)
    # Linear Regression on the discretized dataset
    binning_reg = LinearRegression().fit(X_binned, y)
    # Lasso onthe discretized dataset
    binning_lasso = sklearn_Lasso(alpha=lmbda / X_binned.shape[0],
                                  fit_intercept=False, normalize=False,
                                  max_iter=n_epochs,
                                  tol=1e-15).fit(X_binned, y)

    ax2.plot(X[:, 0], binning_reg.predict(X_binned), linewidth=2, color='blue',
             linestyle='-', label='linear regression')
    # Decision Tree on the discretized dataset
    binning_tree_reg = DecisionTreeRegressor(min_samples_split=3,
                                             random_state=0).fit(X_binned, y)
    ax2.plot(X[:, 0], binning_tree_reg.predict(X_binned), linewidth=2,
             color='red', linestyle=':', label='decision tree')

    ax2.plot(X[:, 0], y, 'o', c='k')
    ax2.vlines(enc.bin_edges_[0], *plt.gca().get_ylim(), linewidth=1,
               alpha=0.2)
    ax2.legend(loc="best")
    ax2.set_xlabel("Input features")
    ax2.set_title("Result after discretization")
    plt.tight_layout()
    plt.show()

    # Assessment of the model by computing the crossval score
    dense_lasso = DenseLasso(lmbda=lmbda, epsilon=epsilon, f=f,
                             n_epochs=n_epochs, screening=screening,
                             store_history=store_history).fit(X, y)

    original_dense_lasso_cv_score = dense_lasso.score(X, y)
    print("original dense lasso crossval score", original_dense_lasso_cv_score)

    # PCA : Principal Component Analysis on the original dataset
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(X)
    svd = pca.singular_values_
    var_ratio = pca.explained_variance_ratio_
    PCs = pca.transform(X)
    print("Variances of the principal components : ", svd)
    print("Explained variance ratio : ", var_ratio)
    print("Principal Components : ", PCs)

    # 3. Prediction on the original dataset after having carried out a PCA

    # Defines the setting of the subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 4))
    # Linear regression on the original daaset
    reg = LinearRegression().fit(X, y)
    ax1.plot(PCs[:, 0], reg.predict(X), linewidth=2, color='blue',
             label='linear regression')

    # Decision Tree on the original dataset
    tree_reg = DecisionTreeRegressor(min_samples_split=3,
                                     random_state=0).fit(X, y)
    ax1.plot(PCs[:, 0], tree_reg.predict(X), linewidth=2, color='red',
             label='decision tree')

    ax1.plot(PCs[:, 0], y, 'o', c='k')
    ax1.legend(loc="best")
    ax1.set_ylabel("Regression output")
    ax1.set_xlabel("Input feature")
    ax1.set_title("Result before discretization")

    # 4. Prediction on the discretized dataset after having carried out a PCA
    n_bins = 10
    encode = 'onehot'
    strategy = 'quantile'
    enc = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    X_binned = enc.fit_transform(X)
    # Linear Regression on the discretized dataset
    binning_reg = LinearRegression().fit(X_binned, y)
    ax2.plot(PCs[:, 0], binning_reg.predict(X_binned), linewidth=2,
             color='blue', linestyle='-', label='linear regression')
    # Decision Tree on the discretized dataset
    binning_tree_reg = DecisionTreeRegressor(min_samples_split=3,
                                             random_state=0).fit(X_binned, y)
    ax2.plot(PCs[:, 0], binning_tree_reg.predict(X_binned), linewidth=2,
             color='red', linestyle=':', label='decision tree')

    ax2.plot(PCs[:, 0], y, 'o', c='k')
    ax2.vlines(enc.bin_edges_[0], *plt.gca().get_ylim(), linewidth=1,
               alpha=0.2)
    ax2.legend(loc="best")
    ax2.set_xlabel("Input features")
    ax2.set_title("Result after discretization")
    plt.tight_layout()
    plt.show()

    # Results
    # Before discretization, the predictions are different whether we use
    # linear regression or decision tree : decision tree is able to build
    # models which are much more complex than linear regression
    # After discretization, the predictions are exactly the same whatever we
    # use linear regression or decision tree.

    # Read CSV : Housing Prices Dataset
    # data_dir = "./Datasets"
    # fname_train = data_dir + "/housing_prices_train"
    # fname_test = data_dir + "/housing_prices_test"
    # train_set = read_csv(fname_train)
    # head_train = train_set.head()
    # test_set = read_csv(fname_test)
    # head_test = test_set.head()
    # print("Housing Prices Training Set Header : ", head_train)
    # print("Housing Prices Testing Set Header : ", head_test)
    """


if __name__ == "__main__":
    main()
