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
from sklearn.metrics import mean_absolute_error

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


#########################################
#               RMSE
#########################################

def rmse_cv(model, X, y, scoring="neg_mean_squared_error", cv=5):
    """RMSE : Cross-val score 
    """
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring, cv).mean())
    return(rmse)

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
    lmbda = 0.1

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False)
    print("number of samples :", X.shape[0])
    print("number of features :", X.shape[1])

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

    # cd = cyclic_coordinate_descent(X, y, lmbda, epsilon, f, n_epochs=10000,
    #                                screening=False,
    #                                store_history=True)[0].fit(X, y)

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
    # binning_lasso = sklearn_Lasso(alpha=lmbda / len(X_binned),
    #                               fit_intercept=False, normalize=False,
    #                               max_iter=n_epochs,
    #                               tol=1e-15).fit(X_binned.getnnz(axis=0), y)

    # binning_cd = cyclic_coordinate_descent(X_binned, y, lmbda, epsilon, f,
    #                                        n_epochs=10000, screening=False,
    #                                        store_history=True)[0].fit(X_binned,
    #                                                                   y)

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
    original_crossval_score = cross_val_score(reg, X, y, cv=5).mean()
    std_original_crossval = cross_val_score(reg, X, y, cv=5).std()
    print("Original crossval score : ", original_crossval_score)
    print("Std original crossval score : ", std_original_crossval)
    original_rmse = rmse_cv(reg, X, y, scoring="neg_mean_squared_error", cv=5)
    print("RMSE on original dataset : ", original_rmse)
    # mae_original_score = mean_absolute_error(y, reg)
    # print("MAE original dataset : ", mae_original_score)
    binning_crossval_score = cross_val_score(reg, X_binned, y, cv=5).mean()
    std_binning_crossval = cross_val_score(reg, X_binned, y, cv=5).std()
    print("Binning crossval score : ", binning_crossval_score)
    print("Std binning crossval score : ", std_binning_crossval)
    # mae_binning_score = mean_absolute_error(y, binning_reg)
    # print("MAE binning score : ", mae_binning_score)
    # cd_crossval_score = cross_val_score(cd, X, y, cv=5)
    # print("CD crossval score : ", cd_crossval_score)
    # binning_cd_crossval_score = cross_val_score(binning_cd, X_binned, y, cv=5)
    # print("Binning CD crossval score : ", binning_cd_crossval_score)

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
    fileDir = "/home/mrivoire/Documents/M2DS_Polytechnique/Stage_INRIA/Code/Datasets"
    fileNameTrain = "/housing_prices_train"
    fileNameTest = "/housing_prices_test"
    filePathTrain = fileDir + fileNameTrain
    train_set = read_csv(filePathTrain)
    head_train = train_set.head()
    filePathTest = fileDir + fileNameTest
    test_set = read_csv(filePathTest)
    head_test = test_set.head()
    print("Housing Prices Training Set Header : ", head_train)
    print("Housing Prices Testing Set Header : ", head_test)


if __name__ == "__main__":
    main()
