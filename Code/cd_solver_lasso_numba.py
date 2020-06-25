import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ipdb

from numpy.random import randn
from scipy.linalg import toeplitz

from numba import njit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso as sklearn_Lasso
from sklearn.utils import check_random_state
from scipy.sparse import csc_matrix
from scipy.sparse import issparse

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


def simu(beta, n_samples=1000, corr=0.5, for_logreg=False,
         random_state=None):
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
    P_lmbda = 0
    D_lmbda = 0
    G_lmbda = 0

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
                                  * max(np.max(np.abs(X.T @ residuals
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
                    mu = (np.abs(X[:, j].T.dot(theta))
                          + r * np.linalg.norm(X[:, j]))

                    if mu < 1:
                        A_c.remove(j)

                if store_history:
                    n_active_features.append(len(A_c))

                if np.abs(G_lmbda) <= epsilon:
                    break

    return (beta, primal_hist, dual_hist, gap_hist, r_list,
            n_active_features, theta, P_lmbda, D_lmbda, G_lmbda, A_c)


#########################################################################
#                 Cyclic CD For Sparse Features Matrix
#########################################################################


@njit
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
    P_lmbda = 0
    D_lmbda = 0
    G_lmbda = 0

    A_c = list(range(n_features))

    L = np.zeros(n_features)

    for i in A_c:
        start, end = X_indptr[i:i + 2]
        for ind in range(start, end):
            L[i] += X_data[ind] ** 2

    for k in range(n_epochs):
        for i in A_c:
            if L[i] == 0.:
                continue

            old_beta_i = beta[i]

            # Matrix product between the features matrix X and the residuals
            start, end = X_indptr[i:i + 2]
            grad = 0.
            for ind in range(start, end):
                grad += X_data[ind] * residuals[X_indices[ind]]

            # Update of the parameters
            step = 1 / L[i]
            beta[i] += step * grad

            # Apply proximal operator
            beta[i] = soft_thresholding(step * lmbda, beta[i])
            diff = old_beta_i - beta[i]

            # import ipdb; ipdb.set_trace()
            if diff != 0:
                for ind in range(start, end):
                    residuals[X_indices[ind]] += diff * X_data[ind]

        if k % f == 0:
            # Computation of theta
            XTR_absmax = 0
            # Matrix product between the features matrix X and the residuals
            for i in A_c:
                start, end = X_indptr[i:i + 2]
                dot = 0.
                for ind in range(start, end):
                    dot += X_data[ind] * residuals[X_indices[ind]]
                XTR_absmax = max(abs(dot), XTR_absmax)

            theta = residuals / max(XTR_absmax, lmbda)

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
                    start, end = X_indptr[j:j + 2]
                    dot = 0.
                    norm = 0.

                    for ind in range(start, end):
                        dot += X_data[ind] * theta[X_indices[ind]]
                        norm += X_data[ind]**2

                    norm = np.sqrt(norm)
                    dot = np.abs(dot)

                    mu = np.abs(dot) + r * norm
                    if mu < 1:
                        A_c.remove(j)

                if store_history:
                    n_active_features.append(len(A_c))

                if np.abs(G_lmbda) <= epsilon:
                    break

    return (beta, primal_hist, dual_hist, gap_hist, r_list,
            n_active_features, theta, P_lmbda, D_lmbda, G_lmbda, A_c)

# https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr


###########################################################################
#                           Class Dense Lasso
###########################################################################

class Lasso:

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
            X_data = X.data
            X_indices = X.indices
            X_indptr = X.indptr

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
             A_c) = sparse_cd(X_data, X_indices, X_indptr, y, self.lmbda,
                              self.epsilon, self.f, self.n_epochs,
                              self.screening, self.store_history)
        else:
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
             A_c) = cyclic_coordinate_descent(X, y, self.lmbda, self.epsilon,
                                              self.f, self.n_epochs,
                                              self.screening,
                                              self.store_history)

        self.slopes = beta_hat_cyclic_cd_true
        self.G_lmbda = G_lmbda
        self.r_list = r_list
        self.A_c = A_c

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
        score = -np.mean(np.abs(y - self.predict(X)))

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
    n_samples, n_features = 10, 30
    beta = rng.randn(n_features)
    lmbda = 1.
    f = 10
    epsilon = 1e-14
    n_epochs = 100000
    screening = True
    store_history = True

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False,
                random_state=rng)

    X = csc_matrix(X)
    X_data = X.data
    X_indices = X.indices
    X_indptr = X.indptr

    print("shape sparse X : ", X.shape)
    print("shape y : ", y.shape)

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
        A_c_sparse) = sparse_cd(X_data=X_data, X_indices=X_indices,
                                X_indptr=X_indptr, y=y, lmbda=lmbda,
                                epsilon=epsilon, f=f,
                                n_epochs=n_epochs, screening=screening,
                                store_history=store_history)

    X = X.toarray()

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
        A_c_dense) = cyclic_coordinate_descent(X, y, lmbda=lmbda, 
                                               epsilon=epsilon,
                                               f=f, n_epochs=n_epochs,
                                               screening=screening,
                                               store_history=True)

    print("shape X dense : ", X.shape)
    print("shape y : ", y.shape)

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

    print("length list epochs : ", len(list_epochs))
    print("length r list : ", len(r_list))

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
    X_binned = X_binned.tocsc()
    print("type X_binned : ", type(X_binned))
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
    dense_lasso = Lasso(lmbda=lmbda, epsilon=epsilon, f=f,
                        n_epochs=n_epochs, screening=screening,
                        store_history=store_history).fit(X, y)

    original_dense_lasso_cv_score = dense_lasso.score(X, y)
    print("original dense lasso crossval score : ", 
          original_dense_lasso_cv_score)

    sparse_lasso = Lasso(lmbda=lmbda, epsilon=epsilon, f=f, n_epochs=n_epochs, 
                         screening=screening, 
                         store_history=store_history).fit(X_binned, y)

    sparse_pred = sparse_lasso.predict(X_binned)
    print("shape sparse pred : ", sparse_pred.shape)
    
    binning_lasso_cv_score = sparse_lasso.score(X_binned, y)
    print("binning lasso crossval score : ", binning_lasso_cv_score)

    sklearn_dense_cv_score = lasso.score(X, y)
    print("sklearn dense lasso crossval score : ", sklearn_dense_cv_score)

    sklearn_binning_cv_score = binning_lasso.score(X_binned, y)
    print("sklearn binning lasso crossval score : ", sklearn_binning_cv_score)

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


if __name__ == "__main__":
    main()
