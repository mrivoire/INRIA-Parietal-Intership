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


#############################################################################
#                         Maximum Inner Product
#############################################################################


def max_val(X_binned_data, X_binned_indices, X_binned_indptr, residuals, 
            max_depth):
    """Compute the maximum inner product between X_binned.T and the residuals
       Return the feature and its id which allows to maximize the inner product

    Parameters
    ----------
    X_binned: numpy.ndarray shape = (n_samples, n_features * n_bins)
        discrete features matrix, sparse matrix

    ids_list: list of length n_features * n_bins
        contains the ids of the discrete features

    max_depth: int
        maximal depth of the tree
        maximal degree of interactions between the original features

    Returns
    -------
    max_feature: feature allowing to maximize the inner product between
        the sparse matrix of features X_binned.T and the residuals vector

    max_id: int
        id of the feature allowing to maximize the inner product
    """

    n_features = len(X_binned_indptr) - 1
    n_samples = max(X_binned_indices) + 1

    max_val = 0
    depth = 0

    parent_data = np.ones(n_samples)
    parent_indices = np.arange(n_samples)

    for i in range(n_features):
        max_val, key = max_val_rec(X_binned_data=X_binned_data, 
                                   X_binned_indices=X_binned_indices, 
                                   X_binned_indptr=X_binned_indptr, 
                                   parent_data=parent_data,
                                   parent_indices=parent_indices,
                                   current_max_val=max_val, j=i,
                                   residuals=residuals,
                                   max_depth=max_depth,
                                   depth=depth)


    # the recursion is used to scan the tree from the root to the leaves ?
    # stopping criterion = when the pruned tree contains no more node for
    # which the sppc criterion is satisfied ?

    # DFS
    # 1. Create a recursive function that takes the index of node and a visited
    # array
    # 2. Mark the current node as visited and print the node
    # 3. Traverse all the adjacent and unmarked nodes and call the recursive
    # function with index of adjacent node.

    return max_val, key


def compute_inner_prod(data1, ind1, residuals):
    """
    Parameters
    ----------

    Returns
    -------
    """

    inner_prod = 0
    inner_prod_pos = 0
    inner_prod_neg = 0

    count1 = 0
    count2 = 0

    while count1 < len(ind1) and count2 < len(residuals):
        if ind1[count1] == count2:
            prod = data1[count1] * residuals[count2]
            inner_prod += prod 

            if residuals[count2] >= 0:
                inner_prod_pos += prod 
            else:
                inner_prod_neg += prod 

            count1 += 1
            count2 += 1
        elif ind1[count1] < count2:
            count1 += 1
        else:
            count2 += 1

    return inner_prod, inner_prod_neg, inner_prod_pos


def compute_interactions(data1, ind1, data2, ind2):
    """
    Parameters
    ----------

    Returns
    -------
    """

    inner_prod = 0

    count1 = 0
    count2 = 0

    inter_feat_ind = list()
    inter_feat_data = list()

    while count1 < len(ind1) and count2 < len(ind2):
        if ind1[count1] == ind2[count2]:
            prod = data1[count1] * data2[count2]
            inner_prod += prod 
            inter_feat_ind.append(ind1[count1])
            inter_feat_data.append(prod)
            count1 += 1
            count2 += 1
        elif ind1[count1] < ind2[count2]:
            count1 += 1
        else:
            count2 += 1

    return inter_feat_data, inter_feat_ind


def max_val_rec(X_binned_data, X_binned_indices, X_binned_indptr, 
                parent_data, parent_indices, current_max_val, j, residuals, 
                max_depth, depth):
    """Compute the maximal inner product value between the given feature and
        the vector of residuals provided thanks to the pre-solve processing

    Parameters
    ----------
    X_binned: numpy.ndarray(), shape = (n_samples, n_features)
        binned features matrix

    parent: numpy.array(), shape = (n_samples, )
        vector of 0 and 1 corresponding to the discrete parent feature

    current_max_val: float
        current maximal inner product between the visited features
        and the residuals

    j (line 254): int
        index of the child node

    residuals: numpy.array(), shape = (n_samples, )
        vector of residuals

    max_depth: int
        maximum degree of interactions between the original features
        maximum depth in the tree

    depth: int
        current depth
        goes from 0 to max_depth if we start from the root to the leaves
        or from max_dpeth to 0 if we start from the leaves to the root

    Returns
    -------

    key: int
        index of the feature which provides the max_val

    current_max_val: float
        maximal inner product between the features and the vector of residuals

    """
    # res obtained by pre-solve
    # compare current_max_inner_prod to inner_product
    # if equation above algo in spp paper is satisfied then prune the subtree
    # otherwise visit the subtree

    # for a given node which is the root of a subtree, returns the max_val
    # if the max_val of the subtree is greater it returns the maxval of the
    # subtree otherwise it returns the previous maxval
    # for a given node t, compute the criterion with the
    # negative and positive parts

    # Compute the current feature
    # element wise product between parent and feature j to obtain the
    # interaction result feature x

    n_features = len(X_binned_indptr) - 1
    n_samples = max(X_binned_indices) + 1

    child_data = np.zeros(n_samples)
    child_indices = np.zeros(n_samples)

    start, end = X_binned_indptr[j, j+2]
    for ind in range(start, end):
        child_indices[X_binned_indices[ind]] = X_binned_indices[ind]
        child_data[X_binned_indices[ind]] = X_binned_data[ind] 


    # How to scan arrays from different sizes in the same time ?
    # How to move in the two vectors of indices (child and parent) in the same 
    # time whereas we have not to move at the same speed ?
    (inter_feat_data, 
     inter_feat_ind) = compute_interactions(child_data, child_indices, 
                                            parent_data, parent_indices)

    (inner_prod, 
     inner_prod_neg, 
     inner_prod_pos) = compute_inner_prod(parent_data, 
                                          parent_indices, 
                                          residuals)

    # If the criterion is verified
        # return the current maxval

    upper_bound = max(inner_prod_pos, -inner_prod_neg)

    if upper_bound <= current_max_val:
        return current_max_val, key

    else:
    # If the criterion is not satisfied:
        # Check the current node :
        # compute max_val
        # update max_val if required

        current_max_val = inner_prod

        # If depth < max_depth:
            # for loop over the child nodes (number of children = n_features)
            # for ind = j+1 ... p if the parent node is j
                # maxval = max_val_rec(depth = depth + 1)

        if depth < max_depth: # start from 0 to max_depth
            # recursive call of the function on the following stage
            for ind in range(j+1, n_features):
                current_max_val, key = max_val_rec(X_binned_data, 
                                                   X_binned_indices, 
                                                   X_binned_indptr,  
                                                   parent_data, 
                                                   parent_indices, 
                                                   current_max_val, 
                                                   j+1, residuals, 
                                                   max_depth, depth + 1)

        # We keep the same parent node and we change the child nodes ? 
        # How to find the key of the feature providing the maxval ?

        return current_max_val, key

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

    n_features = len(X_binned_indptr) - 1
    n_samples = max(X_binned_indices) + 1

    # Test function for max_val

    sparse_lasso_sklearn = sklearn_Lasso(alpha=(lmbda / X_binned.shape[0]),
                                         fit_intercept=False,
                                         normalize=False,
                                         max_iter=n_epochs,
                                         tol=1e-14).fit(X_binned, y)

    residuals = y - X_binned.dot(sparse_lasso_sklearn.coef_)
    print("residuals : ", residuals)

    parent1_feature = np.zeros(n_samples)
    parent2_feature = np.zeros(n_samples)
    child_feature = np.zeros(n_samples)

    interactions_feat_list = list()

    # Building of the interactions features
    for j in range(n_features):
        start1, end1 = X_binned_indptr[j:j+2]
        for ind in range(start1, end1):
            parent1_feature[X_binned_indices[ind]] = X_binned_data[ind]
        for k in range(j+1, n_features):
            start2, end2 = X_binned_indptr[k:k+2]
            for ind in range(start2, end2):
                parent2_feature[X_binned_indices[ind]] = X_binned_data[ind]

            child_feature = np.zeros(n_samples)
            for i in range(n_samples):
                child_feature[i] = parent1_feature[i] * parent2_feature[i]
            interactions_feat_list.append(child_feature)

    max_val_test = 0
    inner_prod = 0

    for ind, feat in enumerate(interactions_feat_list):
        for i in range(n_samples):
            inner_prod += feat[i] * residuals[i]
            if inner_prod > max_val_test:
                max_val_test = inner_prod

    print("max val = ", max_val_test)


    # Test max_val function
    max_depth = 2

    max_inner_prod, key = max_val(X_binned_data=X_binned_data, 
                                  X_binned_indices=X_binned_indices, 
                                  X_binned_indptr=X_binned_indptr, 
                                  residuals=residuals, max_depth=max_depth)

    print("max inner prod = ", max_inner_prod)
    print("max key = ", key)


if __name__ == "__main__":
    main()
