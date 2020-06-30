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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso as sklearn_Lasso
from sklearn.utils import check_random_state
from scipy.sparse import csc_matrix
from scipy.sparse import issparse


#######################################################################
#                   Safe Pattern Pruning Algorithm
#######################################################################


"""
1. Implement a function allowing to give the required patterns under the
proper format to the SPP Algorithm. In this step, perform a binning process
over the continuous variables so as to obtain discrete one-hot encoded features
from which we can create interactions variables.

2. Implement a function which from a given subtree is able to compute the
value of lambda max.

3. Implement the SPP algorithm which scans the tree built over the whole set of
patterns from the root to the leaf and for each node t that tests the SPPC(t).
If the SPPC(t) is satisfied it prunes the whole subtree, otherwise it leaves
it. For the scan of the tree from the root to the leaves, look at depth at
first search algorithm.
At the end of the pruning process, we form the active set with all the patterns
which remain in the tree i.e. which have not been pruned out.

4. Implement a solver allowing to run an optimization process over the active
set A hat. (sparse_cd)

5. Implement a grid search over the lambdas which runs SPP algorithms for
each value of lambda of the grid. The values of lambda in the grid search start
from lambda max and evenly decrease until a very small lambda to obtain a
solution always more dense.
For each value of lambda in the grid, run a SPP algorithm in order
to find the active set A hat. Once A hat have been determined, run the
optimization process thanks to the implemented Lasso solver.

6. At each iteration of the grid search, we obtain a couple of primal dual
solutions, store for each lambda these solutions and return the whole set of
primal / dual solutions.
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


#############################################################################
#                         Maximum Inner Product
#############################################################################


# def max_val(X_binned, ids_list):
#     """Compute the maximum inner product between X_binned.T and the residuals 
#        Return the feature and its id which allows to maximize the inner product

#     Parameters
#     ----------
#     X_binned: numpy.ndarray shape = (n_samples, n_features * n_bins)
#         discrete features matrix, sparse matrix 

#     ids_list: list of length n_features * n_bins
#         contains the ids of the discrete features

#     Returns
#     -------
#     max_feature: feature allowing to maximize the inner product between 
#         the sparse matrix of features X_binned.T and the residuals vector

#     max_id: int
#         id of the feature allowing to maximize the inner product
#     """

  
#     return max_feature, max_id 


# def max_val_rec(feature, id, residuals):
#     """Compute the inner product value between the given feature and the vector
#         of residuals provided thanks to the pre-solve processing

#     Parameters
#     ----------
#     feature: feature for which we want to compute the inner product value

#     id: int
#         id of the feature given as parameter

#     residuals: numpy.array, shape = (n_samples, )
#         vector of residuals

#     Returns
#     -------

#     inner_prod: float
#         inner product value 

#     id: int
#         id of the feature for which we compute the inner product
#     """
#     # res obtained by pre-solve
#     # compare current_max_inner_prod to inner_product
#     # if equation above algo in spp paper is satisfied then prune the subtree
#     # otherwise visit the subtree

#     return inner_prod, id 


# #############################################################################
# #                           Safe Pattern Pruning
# #############################################################################


# def safe_prune(t):
#     """Safe Patterns Pruning Criterion at node t

#     Parameters
#     ----------
#     list of original variables with their ids 
#     center (feasible theta) and radius of the safe sphere 
#     maximum depth = maximum order of interactions 
#     Returns
#     -------
#     safe set : matrix sparse + list of ids or vector of nodes where the nodes
#     are defined in the class node
#     """

#     # recursif algorithm

#     return sppc_t


# def safe_prune_rec(feature, safe_set_to_update):
#     """
#     take as input a single feature either original or interaction
#     """
#     # recursive function depth first search type
#     # call safe prune rec on a feature i and 
#     # compute the interaction ij 
#     # apply the sppc criterion on the interaction feature ij  


# def SPP(tau):
#     """Safe Patterns Pruning Algorithm
#        Scan the tree from the root to the leaves and prunes out the subtree
#        which statisfie the SPPC(t) criterion

#     Parameters
#     ----------
#     tau: tree built over the whole set of patterns of the database

#     Returns
#     -------

#     A_hat: numpy.array, shape = (nb active features, )
#         contains the active features taking part to the optimal
#         predictive model
#     """
#     # compute lambda_max with max_val with beta = 0 and decrease it with a 
#     # logarithmic step 

#     for lambda_t in lambdas_list:
#         # Pre-solve : solve the optimization problem with the new lambda on the 
#         # previous optimal set of features. (epsilon not too small ~ 10^-8)
#         # use the implemented lasso with as input only the previous 
#         # optimal subset on features 
#         # we obtain a beta_hat which is not the optimal beta but which is a 
#         # better optimization of beta since it is closer to the optimum then 
#         # the screening is better from the beginning 

#         # between pre-solve and safe prune
#         # compute the dual gap (compute the primal and the dual)
#         # compute of a feasible dual solution
#         # rescaling of the solution to make it feasible
#         max_val()
#         # compute feasible theta with the max_val
#         # compute the radius of the safe sphere 

#         # Safe Prune after pre-solve : 
#         safe_prune():   

#             # epoch == 0 => first perform spp then launch the solver with 
#             # screening 
#             # then SPP: we obtain a safe set
#             # launch the solver taking as input the safe set 
#             # the solver will screen even more the features in the safe set
#             # safe set vector which contains all the nodes which have not been 
#             # screened 
#             # safe set contains both the sparse features and an id 
#             # (corresponding to the ancestors of the features)
#             # representation of the id of the features
#             # 2 vectors of same size : 1 with the sparse features 
#             # (vector of vectors = sparse matrix) and 
#             # 1 with the id (vector of tuples)
#             # can be directly given as input to the solver
#             # then remove the ids outside from the solver with regards to the 
#             # 0 coeffs of beta
#             # or implement a class "node" with attribute key (id = tuple) 
#             # and the sparse 
#             # vector which represents the feature 
#             # then make a vector of node

#         # launch the already implemented solver on the safe set 


#     return list_of_sol # non-zero beta and the id


def main():

    rng = check_random_state(0)
    n_samples, n_features = 10, 30
    beta = rng.randn(n_features)
    lmbda = 1.
    f = 10
    epsilon = 1e-14
    n_epochs = 100000
    screening = True
    store_history = True
    encode = 'onehot'
    strategy = 'quantile'

    X, y = simu(beta, n_samples=n_samples, corr=0.5, for_logreg=False,
                random_state=rng)

    # Discretization by binning strategy
    enc = KBinsDiscretizer(n_bins=2, encode=encode, strategy=strategy)
    X_binned = enc.fit_transform(X)
    X_binned = X_binned.tocsc()
    X_binned_data = X_binned.data
    X_binned_indices = X_binned.indices
    X_binned_indptr = X_binned.indptr

    n_samples, n_discrete_features = np.shape(X_binned)

    # Test function for max_val 

    sparse_lasso_sklearn = sklearn_Lasso(alpha=(lmbda / X_binned.shape[0]),
                                         fit_intercept=False,
                                         normalize=False,
                                         max_iter=n_epochs,
                                         tol=1e-14).fit(X_binned, y)
    
    max_inner_prod = 0
    residuals = y - X_binned.dot(sparse_lasso_sklearn.coef_)
    print("residuals : ", residuals)
    for j in range(n_discrete_features):
        start, end = X_binned_indptr[j:j + 2]
        dot = 0.
        for ind in range(start, end):
            dot += X_binned_data[ind] * residuals[X_binned_indices[ind]]
        max_inner_prod = max(abs(dot), max_inner_prod)
    
    print("max inner product : ", max_inner_prod)


if __name__ == "__main__":
    main()
